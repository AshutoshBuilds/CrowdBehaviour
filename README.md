# Crowd Behavior Analysis Project

## Overview
An automated system using Deep Learning (CNN-LSTM) to classify crowd behavior into **Normal**, **Violent**, and **Panic** categories.

## Features
- **Multi-Class Classification**: Normal / Violent / Panic (Avenue proxy for panic).
- **Hybrid Architecture**: ResNet50 (spatial) + LSTM (temporal).
- **Better Evaluation**: Confusion Matrix, ROC, PR curves, Loss/Accuracy curves, Error Analysis, per-class metrics.
- **Inference**: `predict.py` for single-video scoring.
- **Results Saved**: Visuals and metrics in `docs/val/` and `docs/test/`.

## Quick Start

1) **Install Dependencies**
```bash
python -m venv .venv
.venv\Scripts\Activate
pip install -r requirements.txt
```

2) **Data Layout**
- `Violent flow/training dataset/non-violence` → Normal
- `Violent flow/training dataset/violence` → Violent
- `Avenue Dataset/training_videos` → Panic proxy (optional; pipeline falls back to 2-class if missing)

3) **Train & Evaluate**
```bash
python main.py
```
- Uses train/val/test splits (seeded), weighted sampler optional, mixed precision on CUDA.
- Saves metrics/plots under `docs/val/` and `docs/test/`.

4) **Inference on a Single Video**
```bash
python predict.py --video /path/to/video.avi --checkpoint model_checkpoint.pth --classes Normal Violent Panic
```

## Notes
- `model_checkpoint.pth` is ~105 MB (GitHub’s 100 MB limit). Use Git LFS or export a smaller ONNX/quantized model if you need to version the checkpoint.
- For a higher-fidelity “Panic” class, replace the Avenue proxy with real panic/evacuation clips or labeled abnormal clips from Avenue.

## Documentation
See `docs/methodology.md` for full details on data, model, training/evaluation, outputs, and recommendations.

