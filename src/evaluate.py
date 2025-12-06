import os
import itertools
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


def evaluate_model(
    model,
    loader,
    history=None,
    device="cuda",
    class_names=None,
    output_dir="docs",
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

    print(f"{Fore.CYAN}Starting evaluation...")

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
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    average_loss = total_loss / max(len(loader), 1)

    print(f"{Fore.GREEN}Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Weighted): {precision:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")
    print(f"F1-Score (Weighted): {f1:.4f}")
    print(f"Loss: {average_loss:.4f}")

    # Visualizations
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Confusion Matrix Heatmap
    cm = confusion_matrix(all_labels, all_preds)
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
    with open(os.path.join(output_dir, "sample_predictions.txt"), "w") as f:
        f.write("Sample Predictions:\n")
        f.write("===================\n")
        indices = np.random.choice(len(all_labels), min(20, len(all_labels)), replace=False)
        for idx in indices:
            actual = class_names[all_labels[idx]]
            pred = class_names[all_preds[idx]]
            status = "CORRECT" if actual == pred else "WRONG"
            f.write(f"Sample {idx}: Actual={actual}, Predicted={pred} [{status}]\n")
    print(f"{Fore.CYAN}Sample predictions saved to {output_dir}/sample_predictions.txt")

    # Save detailed metrics
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision (Weighted): {precision:.4f}\n")
        f.write(f"Recall (Weighted): {recall:.4f}\n")
        f.write(f"F1-Score (Weighted): {f1:.4f}\n")
        f.write(f"Loss: {average_loss:.4f}\n")
        if class_names:
            f.write("\nPer-class metrics (Precision, Recall, F1, Support):\n")
            for i, cls in enumerate(class_names):
                f.write(
                    f"{cls}: P={per_class[0][i]:.4f}, R={per_class[1][i]:.4f}, "
                    f"F1={per_class[2][i]:.4f}, Support={per_class[3][i]}\n"
                )

    results = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "loss": average_loss,
        "per_class": per_class,
    }
    return results
