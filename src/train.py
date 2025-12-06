import torch
import torch.nn as nn
import torch.optim as optim
from colorama import Fore
from tqdm import tqdm
from sklearn.metrics import f1_score


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    learning_rate=0.001,
    device="cuda",
    save_path="model_checkpoint.pth",
    use_amp=True,
    scheduler_step=5,
    scheduler_gamma=0.5,
):
    """
    Train the model using provided train/val loaders and track history.
    """
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        print(f"{Fore.RED}Train/Val datasets are empty. Cannot train.")
        return None, None, None

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.startswith("cuda"))

    best_val_acc = 0.0
    best_val_f1 = 0.0
    history = {"train_loss": [], "train_acc": [], "train_f1": [], "val_loss": [], "val_acc": [], "val_f1": []}

    print(f"{Fore.CYAN}Starting training on {device} for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_preds = []
        train_labels = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp and device.startswith("cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_preds.extend(predicted.detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        train_acc = 100 * correct / total
        train_loss = running_loss / max(len(train_loader), 1)
        train_f1 = f1_score(train_labels, train_preds, average="weighted") if train_labels else 0.0

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_preds.extend(predicted.detach().cpu().numpy())
                val_labels.extend(labels.detach().cpu().numpy())

        val_acc = 100 * val_correct / max(val_total, 1)
        val_loss = val_running_loss / max(len(val_loader), 1)
        val_f1 = f1_score(val_labels, val_preds, average="weighted") if val_labels else 0.0

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"{Fore.GREEN}Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}"
        )

        # Save best model
        if (val_f1 > best_val_f1) or (abs(val_f1 - best_val_f1) < 1e-4 and val_acc > best_val_acc):
            best_val_acc = val_acc
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"{Fore.YELLOW}New best model saved with Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")

        scheduler.step()

    print(
        f"{Fore.CYAN}Training complete. Best Validation Accuracy: {best_val_acc:.2f}% | "
        f"Best Validation F1: {best_val_f1:.4f}"
    )
    return model, history, {"best_val_acc": best_val_acc, "best_val_f1": best_val_f1}
