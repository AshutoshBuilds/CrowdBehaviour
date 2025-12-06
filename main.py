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
    BATCH_SIZE = 4
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    SEQUENCE_LENGTH = 16
    RESIZE = (224, 224)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{Fore.CYAN}Using device: {DEVICE}")
    print(f"{Fore.CYAN}Training for {NUM_EPOCHS} epochs.")

    # Dataset
    print(f"{Fore.CYAN}Loading dataset...")
    dataset = CrowdBehaviorDataset(root_dirs, sequence_length=SEQUENCE_LENGTH, resize=RESIZE, apply_normalization=True)

    if len(dataset) == 0:
        print(f"{Fore.RED}No videos found in the specified directories. Exiting.")
        return

    num_classes = len(root_dirs)
    class_names = list(root_dirs.keys())
    print(f"{Fore.GREEN}Classes: {class_names}")

    # DataLoaders with splits
    loaders = build_loaders(dataset, batch_size=BATCH_SIZE, val_split=0.15, test_split=0.15, seed=42)
    if loaders is None:
        print(f"{Fore.RED}Unable to create data loaders. Exiting.")
        return
    train_loader, val_loader, test_loader = loaders

    # Model
    print(f"{Fore.CYAN}Initializing model...")
    model = CNNLSTM(num_classes=num_classes)

    # Train
    print(f"{Fore.CYAN}Starting training...")
    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        save_path="model_checkpoint.pth",
        use_amp=True,
    )

    if trained_model:
        # Evaluate on validation/test
        print(f"{Fore.CYAN}Evaluating on validation set...")
        evaluate_model(trained_model, val_loader, history=history, device=DEVICE, class_names=class_names, output_dir="docs/val")

        print(f"{Fore.CYAN}Evaluating on test set...")
        evaluate_model(trained_model, test_loader, history=None, device=DEVICE, class_names=class_names, output_dir="docs/test")


if __name__ == "__main__":
    main()
