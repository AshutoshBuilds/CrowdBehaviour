import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore
import numpy as np
import os
import itertools

def evaluate_model(model, dataset, history=None, device='cuda', batch_size=4, class_names=None, output_dir='docs'):
    if len(dataset) == 0:
        print(f"{Fore.RED}Dataset is empty. Cannot evaluate.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print(f"{Fore.CYAN}Starting evaluation...")
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_probs = np.array(all_probs)
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f"{Fore.GREEN}Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Weighted): {precision:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")
    print(f"F1-Score (Weighted): {f1:.4f}")
    
    # Visualizations
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Confusion Matrix Heatmap
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    print(f"{Fore.CYAN}Confusion matrix saved to {output_dir}/confusion_matrix.png")
    
    # 2. ROC Curves (Multiclass)
    if class_names:
        n_classes = len(class_names)
        y_test_bin = label_binarize(all_labels, classes=range(n_classes))
        
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            # Handle edge case where a class might not be in the batch
            if i < all_probs.shape[1]:
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
        plt.close()
        print(f"{Fore.CYAN}ROC curves saved to {output_dir}/roc_curves.png")

    # 3. Training History Plots (if provided)
    if history:
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss Plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
        plt.close()
        
        # Accuracy Plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        plt.title('Training vs Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'))
        plt.close()
        print(f"{Fore.CYAN}Training curves saved to {output_dir}/")

    # 4. Error Analysis Plot (Bar chart of misclassified counts per class)
    misclassified_counts = np.zeros(len(class_names))
    for label, pred in zip(all_labels, all_preds):
        if label != pred:
            misclassified_counts[label] += 1
            
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=misclassified_counts, hue=class_names, legend=False, palette="viridis")
    plt.title('Misclassified Samples per Class')
    plt.ylabel('Count')
    plt.xlabel('Actual Class')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_analysis.png'))
    plt.close()
    print(f"{Fore.CYAN}Error analysis plot saved to {output_dir}/error_analysis.png")

    # 5. Sample Predictions (Save text log of some random samples)
    with open(os.path.join(output_dir, 'sample_predictions.txt'), 'w') as f:
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
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
