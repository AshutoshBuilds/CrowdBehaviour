import cv2
import numpy as np
import os
from colorama import Fore

def extract_frames(video_path, sequence_length=16, resize=(224, 224)):
    """
    Extracts a fixed number of frames from a video.
    If video has fewer frames, it pads with the last frame.
    If it has more, it uniformly samples.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return None

        # Calculate indices for uniform sampling
        if total_frames >= sequence_length:
            indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)
        else:
            # If fewer frames, take all and we will pad later
            indices = np.arange(total_frames)

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame = cv2.resize(frame, resize)
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                if len(frames) == sequence_length and total_frames >= sequence_length:
                    break
        
        # Padding if not enough frames
        if len(frames) < sequence_length:
            while len(frames) < sequence_length:
                frames.append(frames[-1])
                
    except Exception as e:
        print(f"{Fore.RED}Error processing {video_path}: {e}")
        return None
    finally:
        cap.release()

    return np.array(frames)

