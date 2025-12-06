import argparse
import csv
import os
import random
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
from colorama import Fore, init

from src.dataset import CrowdBehaviorDataset
from src.model import CNNLSTM
from src.train import train_model
from src.evaluate import evaluate_model

# Initialize colorama
init(autoreset=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loaders(dataset, batch_size, val_split=0.15, test_split=0.15, seed=42, use_weighted_sampler=True):
    total_len = len(dataset)
    if total_len == 0:
        return None, None, None

    val_len = int(total_len * val_split)
    test_len = int(total_len * test_split)
    train_len = total_len - val_len - test_len

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    # Compute class counts for weighting without loading frames
    class_counts = np.zeros(len(dataset.class_map))
    for idx in train_set.indices:
        class_counts[dataset.labels[idx]] += 1
    class_weights = None
    sampler = None
    if use_weighted_sampler and class_counts.sum() > 0:
        class_weights = class_counts.sum() / (len(class_counts) * np.maximum(class_counts, 1))
        sample_weights = [class_weights[label] for _, label in train_set]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, shuffle=sampler is None)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def main():
    print(f"{Fore.CYAN}Crowd Behavior Analysis System (Normal, Violent, Panic)")

    parser = argparse.ArgumentParser(description="Train/Evaluate crowd behavior models with backbone comparison.")
    parser.add_argument("--models", nargs="+", default=["resnet50"], help="Backbone models to compare")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training/eval")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--sequence-length", type=int, default=16, help="Frames per clip")
    parser.add_argument("--resize", nargs=2, type=int, default=[224, 224], help="Resize (H W) for frames")
    parser.add_argument("--train-backbone", action="store_true", help="Unfreeze backbone for fine-tuning")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--output-dir", default="docs", help="Base output directory for artifacts")
    args = parser.parse_args()

    set_seed(42)

    # Configuration (can be moved to a config file if needed)
    BASE_DIR = os.getcwd()
    VIOLENT_FLOW_DIR = os.path.join(BASE_DIR, "Violent flow")

    normal_dir = os.path.join(VIOLENT_FLOW_DIR, "training dataset", "non-violence")
    violent_dir = os.path.join(VIOLENT_FLOW_DIR, "training dataset", "violence")
    avenue_dir = os.path.join(BASE_DIR, "Avenue Dataset", "training_videos")

    root_dirs = {
        "Normal": normal_dir,
        "Violent": violent_dir,
    }

    if os.path.exists(avenue_dir):
        root_dirs["Panic"] = avenue_dir
    else:
        print(f"{Fore.YELLOW}Warning: 'Panic' class data (Avenue Dataset) not found at {avenue_dir}.")
        print(f"{Fore.YELLOW}Proceeding with 2 classes: Normal, Violent.")

    # Hyperparameters
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    SEQUENCE_LENGTH = args.sequence_length
    RESIZE = (args.resize[0], args.resize[1])
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{Fore.CYAN}Using device: {DEVICE}")
    print(f"{Fore.CYAN}Training for {NUM_EPOCHS} epochs.")
    if args.augment:
        print(f"{Fore.CYAN}Data augmentation enabled.")
    if args.train_backbone:
        print(f"{Fore.CYAN}Backbone fine-tuning enabled.")

    # Dataset
    print(f"{Fore.CYAN}Loading dataset...")
    dataset = CrowdBehaviorDataset(
        root_dirs,
        sequence_length=SEQUENCE_LENGTH,
        resize=RESIZE,
        apply_normalization=True,
        apply_augmentations=args.augment,
    )

    if len(dataset) == 0:
        print(f"{Fore.RED}No videos found in the specified directories. Exiting.")
        return

    num_classes = len(root_dirs)
    class_names = list(root_dirs.keys())
    print(f"{Fore.GREEN}Classes: {class_names}")

    # DataLoaders with splits
    loaders = build_loaders(dataset, batch_size=BATCH_SIZE, val_split=0.15, test_split=0.15, seed=42)
    if (not loaders) or any(l is None for l in loaders):
        print(f"{Fore.RED}Unable to create data loaders. Exiting.")
        return
    train_loader, val_loader, test_loader = loaders

    os.makedirs(args.output_dir, exist_ok=True)
    benchmarks_dir = os.path.join(args.output_dir, "benchmarks")
    os.makedirs(benchmarks_dir, exist_ok=True)
    summary_rows = []

    for model_name in args.models:
        print(f"{Fore.CYAN}Initializing model backbone: {model_name}")
        model = CNNLSTM(
            num_classes=num_classes,
            backbone=model_name,
            train_backbone=args.train_backbone,
            pretrained_backbone=True,
        )

        checkpoint_path = os.path.join(args.output_dir, f"{model_name}_checkpoint.pth")

        # Train
        print(f"{Fore.CYAN}Starting training for {model_name}...")
        trained_model, history, best_metrics = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            device=DEVICE,
            save_path=checkpoint_path,
            use_amp=True,
        )

        if trained_model is None:
            continue

        # Reload best checkpoint before evaluation to ensure best weights are used.
        if os.path.exists(checkpoint_path):
            trained_model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

        # Evaluate on validation/test with per-model subfolders
        val_dir = os.path.join(args.output_dir, "val", model_name)
        test_dir = os.path.join(args.output_dir, "test", model_name)

        print(f"{Fore.CYAN}Evaluating {model_name} on validation set...")
        val_metrics = evaluate_model(
            trained_model,
            val_loader,
            history=history,
            device=DEVICE,
            class_names=class_names,
            output_dir=val_dir,
        )

        print(f"{Fore.CYAN}Evaluating {model_name} on test set...")
        test_metrics = evaluate_model(
            trained_model,
            test_loader,
            history=None,
            device=DEVICE,
            class_names=class_names,
            output_dir=test_dir,
        )

        if val_metrics:
            summary_rows.append(
                {
                    "model": model_name,
                    "split": "val",
                    "accuracy": val_metrics["accuracy"],
                    "precision_weighted": val_metrics["precision_weighted"],
                    "recall_weighted": val_metrics["recall_weighted"],
                    "f1_weighted": val_metrics["f1_weighted"],
                    "loss": val_metrics["loss"],
                }
            )
        if test_metrics:
            summary_rows.append(
                {
                    "model": model_name,
                    "split": "test",
                    "accuracy": test_metrics["accuracy"],
                    "precision_weighted": test_metrics["precision_weighted"],
                    "recall_weighted": test_metrics["recall_weighted"],
                    "f1_weighted": test_metrics["f1_weighted"],
                    "loss": test_metrics["loss"],
                }
            )

    if summary_rows:
        summary_path = os.path.join(benchmarks_dir, "summary.csv")
        with open(summary_path, "w", newline="") as csvfile:
            fieldnames = ["model", "split", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "loss"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"{Fore.GREEN}Benchmark summary saved to {summary_path}")


if __name__ == "__main__":
    main()
