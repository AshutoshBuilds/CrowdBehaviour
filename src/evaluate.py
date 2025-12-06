import os
import itertools
from datetime import datetime
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize


def _percentile_ci(values, alpha=0.05):
    """
    Compute percentile confidence interval for an array.
    Returns (low, high). Safe for empty arrays.
    """
    if len(values) == 0:
        return (0.0, 0.0)
    lower = 100 * (alpha / 2)
    upper = 100 * (1 - alpha / 2)
    return tuple(np.percentile(values, [lower, upper]))


def _stratified_bootstrap_indices(labels, rng):
    """
    Generate bootstrap indices while preserving class proportions.
    Keeps each class frequency constant per resample to avoid collapsing
    minority classes (helps CI stability on tiny splits).
    """
    labels = np.asarray(labels)
    unique_classes, counts = np.unique(labels, return_counts=True)
    bootstrap_idxs = []
    for cls, count in zip(unique_classes, counts):
        cls_idxs = np.where(labels == cls)[0]
        if len(cls_idxs) == 0:
            continue
        bootstrap_idxs.append(rng.choice(cls_idxs, size=count, replace=True))
    if not bootstrap_idxs:
        return np.array([], dtype=int)
    return np.concatenate(bootstrap_idxs)


def _bootstrap_cis(
    labels, preds, n_classes, iters=200, alpha=0.05, random_state=42, stratified=True
):
    """
    Bootstrap confidence intervals for overall and per-class metrics.
    labels/preds must be numpy arrays.
    stratified=True keeps per-class counts fixed each draw to stabilize CIs
    when supports are small.
    """
    rng = np.random.default_rng(random_state)
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    n = len(labels)
    if n == 0 or iters <= 0:
        return None

    stats = {
        "accuracy": [],
        "precision_weighted": [],
        "recall_weighted": [],
        "f1_weighted": [],
        "per_class": [[] for _ in range(n_classes)],
    }

    for _ in range(iters):
        if stratified:
            idx = _stratified_bootstrap_indices(labels, rng)
            if len(idx) == 0:
                continue
        else:
            idx = rng.integers(0, n, size=n)
        lbl = labels[idx]
        prd = preds[idx]

        stats["accuracy"].append(accuracy_score(lbl, prd))
        p_w, r_w, f_w, _ = precision_recall_fscore_support(
            lbl, prd, labels=range(n_classes), average="weighted", zero_division=0
        )
        stats["precision_weighted"].append(p_w)
        stats["recall_weighted"].append(r_w)
        stats["f1_weighted"].append(f_w)

        p_c, r_c, f_c, _ = precision_recall_fscore_support(
            lbl, prd, labels=range(n_classes), average=None, zero_division=0
        )
        for i in range(n_classes):
            stats["per_class"][i].append((p_c[i], r_c[i], f_c[i]))

    per_class_ci = []
    for cls_stats in stats["per_class"]:
        if not cls_stats:
            per_class_ci.append(
                {
                    "precision": (0.0, 0.0),
                    "recall": (0.0, 0.0),
                    "f1": (0.0, 0.0),
                }
            )
            continue
        p_vals, r_vals, f_vals = zip(*cls_stats)
        per_class_ci.append(
            {
                "precision": _percentile_ci(p_vals, alpha),
                "recall": _percentile_ci(r_vals, alpha),
                "f1": _percentile_ci(f_vals, alpha),
            }
        )

    return {
        "accuracy": _percentile_ci(stats["accuracy"], alpha),
        "precision_weighted": _percentile_ci(stats["precision_weighted"], alpha),
        "recall_weighted": _percentile_ci(stats["recall_weighted"], alpha),
        "f1_weighted": _percentile_ci(stats["f1_weighted"], alpha),
        "per_class": per_class_ci,
    }


def evaluate_model(
    model,
    loader,
    history=None,
    device="cuda",
    class_names=None,
    output_dir="docs",
    model_name="unknown_model",
    model_version="not-specified",
    dataset_name="Crowd Behavior (Violent Flow + Avenue proxy)",
    dataset_version="not-specified",
    split_name=None,
    purpose_audience="Sample predictions for qualitative review and sharing with stakeholders.",
    bootstrap_iters=0,
    ci_alpha=0.05,
    bootstrap_stratified=True,
    return_details=False,
    random_state=42,
    split_meta=None,
):
    if len(loader.dataset) == 0:
        print(f"{Fore.RED}Dataset is empty. Cannot evaluate.")
        return None

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    split_label = split_meta.get("split_name") if split_meta else split_name
    print(f"{Fore.CYAN}Starting evaluation... Split={split_label or 'unspecified'}")
    if split_meta:
        print(
            f"{Fore.CYAN}Split size: {split_meta.get('size', len(loader.dataset))} | "
            f"Class counts: {split_meta.get('class_counts', 'n/a')} | Seed: {split_meta.get('seed', random_state)}"
        )

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels_device = labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, labels_device)
            _, predicted = torch.max(outputs.data, 1)

            total_loss += loss.item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)

    # Metrics
    n_classes = len(class_names) if class_names else len(np.unique(all_labels))
    label_range = list(range(n_classes))
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=label_range, average="weighted", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        all_labels, all_preds, labels=label_range, average=None, zero_division=0
    )
    average_loss = total_loss / max(len(loader), 1)

    n_classes = len(class_names) if class_names else len(per_class[0])
    ci = None
    if bootstrap_iters and len(all_labels) > 0:
        ci = _bootstrap_cis(
            np.array(all_labels),
            np.array(all_preds),
            n_classes=n_classes,
            iters=bootstrap_iters,
            alpha=ci_alpha,
            random_state=random_state,
            stratified=bootstrap_stratified,
        )

    print(f"{Fore.GREEN}Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Weighted): {precision:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")
    print(f"F1-Score (Weighted): {f1:.4f}")
    print(f"Loss: {average_loss:.4f}")

    # Derive metadata helpers for downstream logging/saving.
    resolved_class_names = class_names or [f"Class {i}" for i in range(n_classes)]
    split_value = split_name or os.path.basename(os.path.dirname(output_dir)) or "unspecified"
    evaluation_date = datetime.now().strftime("%Y-%m-%d")

    # Visualizations
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Confusion Matrix Heatmap
    cm = confusion_matrix(all_labels, all_preds, labels=label_range)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    print(f"{Fore.CYAN}Confusion matrix saved to {output_dir}/confusion_matrix.png")

    # 2. ROC Curves (Multiclass)
    if class_names:
        n_classes = len(class_names)
        y_test_bin = label_binarize(all_labels, classes=range(n_classes))

        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            if i < all_probs.shape[1]:
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roc_curves.png"))
        plt.close()
        print(f"{Fore.CYAN}ROC curves saved to {output_dir}/roc_curves.png")

    # 3. Precision-Recall Curves (Multiclass)
    if class_names:
        n_classes = len(class_names)
        y_test_bin = label_binarize(all_labels, classes=range(n_classes))
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            if i < all_probs.shape[1]:
                precision_c, recall_c, _ = precision_recall_curve(y_test_bin[:, i], all_probs[:, i])
                pr_auc = auc(recall_c, precision_c)
                plt.plot(recall_c, precision_c, label=f"{class_names[i]} (AUC = {pr_auc:.2f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pr_curves.png"))
        plt.close()
        print(f"{Fore.CYAN}Precision-Recall curves saved to {output_dir}/pr_curves.png")

    # 4. Training History Plots (if provided)
    if history:
        epochs = range(1, len(history["train_loss"]) + 1)

        # Loss Plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history["train_loss"], "b-", label="Training Loss")
        plt.plot(epochs, history["val_loss"], "r-", label="Validation Loss")
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_curve.png"))
        plt.close()

        # Accuracy Plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history["train_acc"], "b-", label="Training Accuracy")
        plt.plot(epochs, history["val_acc"], "r-", label="Validation Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
        plt.close()
        print(f"{Fore.CYAN}Training curves saved to {output_dir}/")

    # 5. Error Analysis Plot (Bar chart of misclassified counts per class)
    misclassified_counts = np.zeros(len(class_names))
    for label, pred in zip(all_labels, all_preds):
        if label != pred:
            misclassified_counts[label] += 1

    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=misclassified_counts, hue=class_names, legend=False, palette="viridis")
    plt.title("Misclassified Samples per Class")
    plt.ylabel("Count")
    plt.xlabel("Actual Class")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_analysis.png"))
    plt.close()
    print(f"{Fore.CYAN}Error analysis plot saved to {output_dir}/error_analysis.png")

    # 6. Sample Predictions (Save text log of some random samples)
    metadata_lines = [
        "---",
        f"model_name: {model_name}",
        f"model_version: {model_version}",
        f"evaluation_date: {evaluation_date}",
        f"dataset_name: {dataset_name}",
        f"dataset_version: {dataset_version}",
        f"data_split: {split_value}",
        "overall_metrics:",
        f"  accuracy: {accuracy:.4f}",
        f"  precision_weighted: {precision:.4f}",
        f"  recall_weighted: {recall:.4f}",
        f"  f1_weighted: {f1:.4f}",
        "class_metrics:",
    ]
    if split_meta:
        metadata_lines.extend(
            [
                "split_details:",
                f"  seed: {split_meta.get('seed', random_state)}",
                f"  total_dataset_size: {split_meta.get('total_samples', 'n/a')}",
                f"  split_name: {split_label or split_value}",
                f"  split_size: {split_meta.get('size', len(loader.dataset))}",
                f"  split_class_counts: {split_meta.get('class_counts', 'n/a')}",
            ]
        )

    if ci:
        metadata_lines.extend(
            [
                "overall_metrics_ci:",
                f"  accuracy_95_ci: [{ci['accuracy'][0]:.4f}, {ci['accuracy'][1]:.4f}]",
                f"  precision_weighted_95_ci: [{ci['precision_weighted'][0]:.4f}, {ci['precision_weighted'][1]:.4f}]",
                f"  recall_weighted_95_ci: [{ci['recall_weighted'][0]:.4f}, {ci['recall_weighted'][1]:.4f}]",
                f"  f1_weighted_95_ci: [{ci['f1_weighted'][0]:.4f}, {ci['f1_weighted'][1]:.4f}]",
            ]
        )

    for i, cls in enumerate(resolved_class_names):
        metadata_lines.append(f"  - class: {cls}")
        metadata_lines.append(f"    precision: {per_class[0][i]:.4f}")
        metadata_lines.append(f"    recall: {per_class[1][i]:.4f}")
        metadata_lines.append(f"    f1: {per_class[2][i]:.4f}")
        metadata_lines.append(f"    support: {per_class[3][i]}")
        if ci:
            p_ci = ci["per_class"][i]["precision"]
            r_ci = ci["per_class"][i]["recall"]
            f_ci = ci["per_class"][i]["f1"]
            metadata_lines.append(f"    precision_95_ci: [{p_ci[0]:.4f}, {p_ci[1]:.4f}]")
            metadata_lines.append(f"    recall_95_ci: [{r_ci[0]:.4f}, {r_ci[1]:.4f}]")
            metadata_lines.append(f"    f1_95_ci: [{f_ci[0]:.4f}, {f_ci[1]:.4f}]")

    metadata_lines.extend(
        [
            "purpose_audience: >",
            f"  {purpose_audience}",
            "---",
            "Sample Predictions:",
            "===================",
        ]
    )

    with open(os.path.join(output_dir, "sample_predictions.txt"), "w") as f:
        f.write("\n".join(metadata_lines) + "\n")
        indices = np.random.choice(len(all_labels), min(20, len(all_labels)), replace=False)
        for idx in indices:
            actual = resolved_class_names[all_labels[idx]]
            pred = resolved_class_names[all_preds[idx]]
            status = "CORRECT" if actual == pred else "WRONG"
            f.write(f"Sample {idx}: Actual={actual}, Predicted={pred} [{status}]\n")
    print(f"{Fore.CYAN}Sample predictions saved to {output_dir}/sample_predictions.txt")

    # Save detailed metrics
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        acc_line = f"Accuracy: {accuracy:.4f}"
        if ci:
            acc_line += f" (95% CI: {ci['accuracy'][0]:.4f}-{ci['accuracy'][1]:.4f})"
        f.write(acc_line + "\n")

        prec_line = f"Precision (Weighted): {precision:.4f}"
        rec_line = f"Recall (Weighted): {recall:.4f}"
        f1_line = f"F1-Score (Weighted): {f1:.4f}"
        if ci:
            prec_line += f" (95% CI: {ci['precision_weighted'][0]:.4f}-{ci['precision_weighted'][1]:.4f})"
            rec_line += f" (95% CI: {ci['recall_weighted'][0]:.4f}-{ci['recall_weighted'][1]:.4f})"
            f1_line += f" (95% CI: {ci['f1_weighted'][0]:.4f}-{ci['f1_weighted'][1]:.4f})"
        f.write(prec_line + "\n")
        f.write(rec_line + "\n")
        f.write(f1_line + "\n")
        f.write(f"Loss: {average_loss:.4f}\n")
        if split_meta:
            f.write("\nSplit details:\n")
            f.write(f"Split name: {split_label or split_value}\n")
            f.write(f"Split size: {split_meta.get('size', len(loader.dataset))}\n")
            f.write(f"Class counts: {split_meta.get('class_counts', 'n/a')}\n")
            f.write(f"Seed: {split_meta.get('seed', random_state)}\n")
            f.write(f"Total dataset size: {split_meta.get('total_samples', 'n/a')}\n")
        if class_names:
            f.write("\nPer-class metrics (Precision, Recall, F1, Support):\n")
            for i, cls in enumerate(class_names):
                line = (
                    f"{cls}: P={per_class[0][i]:.4f}, R={per_class[1][i]:.4f}, "
                    f"F1={per_class[2][i]:.4f}, Support={per_class[3][i]}"
                )
                if ci:
                    p_ci = ci["per_class"][i]["precision"]
                    r_ci = ci["per_class"][i]["recall"]
                    f_ci = ci["per_class"][i]["f1"]
                    line += (
                        f" | P 95% CI: {p_ci[0]:.4f}-{p_ci[1]:.4f}, "
                        f"R 95% CI: {r_ci[0]:.4f}-{r_ci[1]:.4f}, "
                        f"F1 95% CI: {f_ci[0]:.4f}-{f_ci[1]:.4f}"
                    )
                f.write(line + "\n")

    results = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "loss": average_loss,
        "per_class": per_class,
        "ci": ci,
    }
    if return_details:
        results["labels"] = np.array(all_labels)
        results["preds"] = np.array(all_preds)
    return results
