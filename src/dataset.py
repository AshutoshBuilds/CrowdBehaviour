import glob
import os
import random
import torch
from torch.utils.data import Dataset
from colorama import Fore
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from .utils import extract_frames


class CrowdBehaviorDataset(Dataset):
    def __init__(
        self,
        root_dirs,
        sequence_length=16,
        resize=(224, 224),
        apply_normalization=True,
        apply_augmentations=False,
    ):
        """
        root_dirs: dict mapping class names to directory paths.
                   e.g., {'Normal': 'path/to/normal', 'Violent': 'path/to/violent', 'Panic': 'path/to/panic'}
        """
        self.samples = []
        self.labels = []
        self.class_map = {name: i for i, name in enumerate(root_dirs.keys())}
        self.sequence_length = sequence_length
        self.resize = resize
        self.apply_augmentations = apply_augmentations

        # ImageNet normalization aligns with pretrained backbones.
        base_transforms = [T.ConvertImageDtype(torch.float)]
        if apply_normalization:
            base_transforms.append(
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        self.normalizer = T.Compose(base_transforms)

        for class_name, dir_path in root_dirs.items():
            if not os.path.exists(dir_path):
                print(f"{Fore.YELLOW}Warning: Directory not found for {class_name}: {dir_path}")
                continue

            # Support avi, mp4, etc. (recursive included) with de-duplication.
            video_files = set()
            for pattern in ("*.avi", "*.mp4", os.path.join("**", "*.avi"), os.path.join("**", "*.mp4")):
                video_files.update(glob.glob(os.path.join(dir_path, pattern), recursive="**" in pattern))
            video_files = sorted(video_files)

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
            # Handle corrupted video by returning zeros; ideally filter these offline.
            frames = torch.zeros((self.sequence_length, 3, self.resize[0], self.resize[1]))
        else:
            # (T, H, W, C) -> float tensor (T, C, H, W)
            frames = torch.FloatTensor(frames) / 255.0
            frames = frames.permute(0, 3, 1, 2)

        # Apply consistent augmentations per sequence if enabled.
        if self.apply_augmentations:
            flip = random.random() < 0.5
            # Sample deterministic jitter params once per clip to keep all frames coherent.
            brightness = random.uniform(0.8, 1.2)  # 1 ± 0.2
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            hue = random.uniform(-0.05, 0.05)
            rotation_deg = random.uniform(-5, 5)
        else:
            flip = False
            brightness = contrast = saturation = hue = None
            rotation_deg = 0.0

        processed_frames = []
        for frame in frames:
            if flip:
                frame = torch.flip(frame, dims=[2])  # horizontal flip (W dimension)
            if brightness is not None:
                frame = F.adjust_brightness(frame, brightness)
            if contrast is not None:
                frame = F.adjust_contrast(frame, contrast)
            if saturation is not None:
                frame = F.adjust_saturation(frame, saturation)
            if hue is not None:
                frame = F.adjust_hue(frame, hue)
            if rotation_deg != 0.0:
                frame = F.rotate(frame, rotation_deg, interpolation=InterpolationMode.BILINEAR)
            frame = self.normalizer(frame)
            processed_frames.append(frame)

        frames = torch.stack(processed_frames)

        return frames, label

