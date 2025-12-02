import os
import torch
from torch.utils.data import Dataset
from .utils import extract_frames
from colorama import Fore
import glob

class CrowdBehaviorDataset(Dataset):
    def __init__(self, root_dirs, sequence_length=16, transform=None, resize=(224, 224)):
        """
        root_dirs: dict mapping class names to directory paths.
                   e.g., {'Normal': 'path/to/normal', 'Violent': 'path/to/violent', 'Panic': 'path/to/panic'}
        """
        self.samples = []
        self.labels = []
        self.class_map = {name: i for i, name in enumerate(root_dirs.keys())}
        self.sequence_length = sequence_length
        self.transform = transform
        self.resize = resize

        for class_name, dir_path in root_dirs.items():
            if not os.path.exists(dir_path):
                print(f"{Fore.YELLOW}Warning: Directory not found for {class_name}: {dir_path}")
                continue
            
            # Support avi, mp4, etc.
            video_files = glob.glob(os.path.join(dir_path, "*.avi")) + \
                          glob.glob(os.path.join(dir_path, "*.mp4")) + \
                          glob.glob(os.path.join(dir_path, "**", "*.avi"), recursive=True) # Recursive for subfolders

            print(f"{Fore.CYAN}Found {len(video_files)} videos for class {class_name}")
            
            for video_path in video_files:
                self.samples.append(video_path)
                self.labels.append(self.class_map[class_name])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = self.samples[idx]
        label = self.labels[idx]

        frames = extract_frames(video_path, sequence_length=self.sequence_length, resize=self.resize)
        
        if frames is None:
            # Handle corrupted video by returning the next one (naive approach) or zeros
            # Here we return a zero tensor to avoid crashing, but ideally should filter dataset first
            frames = torch.zeros((self.sequence_length, 3, self.resize[0], self.resize[1]))
        else:
            # Normalize to [0, 1] and transform to Tensor (C, T, H, W) or (T, C, H, W)
            # PyTorch usually expects (C, T, H, W) for 3D Conv or (T, C, H, W) for sequence models
            # We will output (T, C, H, W) for CNN-LSTM
            frames = torch.FloatTensor(frames) / 255.0
            frames = frames.permute(0, 3, 1, 2) # (T, H, W, C) -> (T, C, H, W)

        if self.transform:
            # Apply transforms if any
            pass

        return frames, label

