import os
import torch
from src.dataset import CrowdBehaviorDataset
from src.model import CNNLSTM
from src.train import train_model
from src.evaluate import evaluate_model
from colorama import Fore, init

# Initialize colorama
init(autoreset=True)

def main():
    print(f"{Fore.CYAN}Crowd Behavior Analysis System (Normal, Violent, Panic)")
    
    # Configuration
    BASE_DIR = os.getcwd()
    VIOLENT_FLOW_DIR = os.path.join(BASE_DIR, "Violent flow")
    
    normal_dir = os.path.join(VIOLENT_FLOW_DIR, "training dataset", "non-violence")
    violent_dir = os.path.join(VIOLENT_FLOW_DIR, "training dataset", "violence")
    avenue_dir = os.path.join(BASE_DIR, "Avenue Dataset", "training_videos")
    
    root_dirs = {
        'Normal': normal_dir,
        'Violent': violent_dir
    }
    
    if os.path.exists(avenue_dir):
        root_dirs['Panic'] = avenue_dir
    else:
        print(f"{Fore.YELLOW}Warning: 'Panic' class data (Avenue Dataset) not found at {avenue_dir}.")
        print(f"{Fore.YELLOW}Proceeding with 2 classes: Normal, Violent.")
        
    # Hyperparameters - Increased epochs as requested
    BATCH_SIZE = 4
    NUM_EPOCHS = 15  # Increased from 5
    LEARNING_RATE = 1e-4
    SEQUENCE_LENGTH = 16
    RESIZE = (224, 224)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"{Fore.CYAN}Using device: {DEVICE}")
    print(f"{Fore.CYAN}Training for {NUM_EPOCHS} epochs.")
    
    # Dataset
    print(f"{Fore.CYAN}Loading dataset...")
    dataset = CrowdBehaviorDataset(root_dirs, sequence_length=SEQUENCE_LENGTH, resize=RESIZE)
    
    if len(dataset) == 0:
        print(f"{Fore.RED}No videos found in the specified directories. Exiting.")
        return

    num_classes = len(root_dirs)
    class_names = list(root_dirs.keys())
    print(f"{Fore.GREEN}Classes: {class_names}")
    
    # Model
    print(f"{Fore.CYAN}Initializing model...")
    model = CNNLSTM(num_classes=num_classes)
    
    # Train
    print(f"{Fore.CYAN}Starting training...")
    trained_model, history = train_model(model, dataset, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, device=DEVICE)
    
    if trained_model:
        # Evaluate
        evaluate_model(trained_model, dataset, history=history, device=DEVICE, batch_size=BATCH_SIZE, class_names=class_names)

if __name__ == "__main__":
    main()
