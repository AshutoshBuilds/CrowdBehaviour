# Crowd Behavior Analysis Project

## Overview
An automated system using Deep Learning (CNN-LSTM) to classify crowd behavior into **Normal**, **Violent**, and **Panic** categories.

## Features
- **Multi-Class Classification**: Detects Normal, Violent, and Panic behaviors.
- **Hybrid Architecture**: ResNet50 (Spatial) + LSTM (Temporal).
- **Comprehensive Evaluation**: Confusion Matrix, ROC Curves, Loss/Accuracy Plots, and Error Analysis.
- **Visual Results**: All metrics and plots are automatically saved to `docs/`.

## Quick Start

1.  **Install Dependencies**:
    ```bash
    python -m venv .venv
    .venv\Scripts\Activate
    pip install -r requirements.txt
    ```

2.  **Run Training**:
    ```bash
    python main.py
    ```

## Documentation
See `docs/methodology.md` for detailed explanation of the model, dataset, and results.

