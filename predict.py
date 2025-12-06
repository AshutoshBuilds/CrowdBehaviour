import argparse
import os
import torch
from colorama import Fore, init

from src.model import CNNLSTM
from src.dataset import CrowdBehaviorDataset
from src.utils import extract_frames


init(autoreset=True)


def load_model(checkpoint_path, num_classes, device):
    model = CNNLSTM(num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_video(model, video_path, device, sequence_length=16, resize=(224, 224), class_names=None):
    frames = extract_frames(video_path, sequence_length=sequence_length, resize=resize)
    if frames is None:
        print(f"{Fore.RED}Could not read video: {video_path}")
        return None
    frames = torch.FloatTensor(frames) / 255.0
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
    # Apply ImageNet normalization to align with training/pretrained backbones
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    frames = frames.unsqueeze(0).to(device)
    frames = (frames - mean) / std

    with torch.no_grad():
        logits = model(frames)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

    top_idx = probs.argmax()
    result = {
        "predicted_class": class_names[top_idx] if class_names else str(top_idx),
        "probabilities": {class_names[i] if class_names else str(i): float(p) for i, p in enumerate(probs)},
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict crowd behavior for a single video.")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--checkpoint", default="model_checkpoint.pth", help="Path to model checkpoint")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["Normal", "Violent", "Panic"],
        help="Class names in order",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference")
    args = parser.parse_args()

    device = args.device
    if not os.path.exists(args.checkpoint):
        print(f"{Fore.RED}Checkpoint not found at {args.checkpoint}")
        return
    if not os.path.exists(args.video):
        print(f"{Fore.RED}Video not found at {args.video}")
        return

    model = load_model(args.checkpoint, num_classes=len(args.classes), device=device)
    result = predict_video(model, args.video, device=device, class_names=args.classes)

    if result:
        print(f"{Fore.GREEN}Prediction: {result['predicted_class']}")
        print("Probabilities:")
        for cls, prob in result["probabilities"].items():
            print(f"  {cls}: {prob:.4f}")


if __name__ == "__main__":
    main()
