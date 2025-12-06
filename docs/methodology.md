# Crowd Behavior Analysis System

This project implements an automated system for multi-class classification of crowd behavior into three categories: **Normal**, **Violent**, and **Panic**. It utilizes deep learning techniques (CNN-LSTM) to analyze video sequences and detect abnormal events.

## 1. Methodology

### 1.1 Dataset
The system uses a combination of two standard datasets to represent the three classes:

*   **Violent Flow Dataset**:
    *   **Normal**: Sourced from the 'non-violence' subset. Represents standard crowd movement without aggression.
    *   **Violent**: Sourced from the 'violence' subset. Represents fights, riots, and aggressive behavior.
*   **Avenue Dataset**:
    *   **Panic**: Sourced from the 'training_videos' subset of the Avenue dataset. While originally 'Normal' in the Avenue context, for this specific project demonstration, we utilize these clips to simulate a distinct crowd behavior class (labeled as 'Panic') to test the system's multi-class capability. In a production environment, this would be replaced with specific panic/evacuation footage.

### 1.2 Model Architecture: CNN-LSTM
We employ a hybrid architecture that leverages the strengths of Convolutional Neural Networks (CNNs) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for temporal modeling.

1.  **Spatial Feature Extractor (ResNet50)**:
    *   A **ResNet50** model, pre-trained on ImageNet, is used as the backbone.
    *   The final classification layer is removed, allowing us to extract a rich, 2048-dimensional feature vector from each video frame.
    *   This captures visual details like objects, people, and scene context.

2.  **Temporal Sequence Modeling (LSTM)**:
    *   The sequence of feature vectors (from 16 consecutive frames) is fed into an **LSTM** layer.
    *   The LSTM maintains an internal state that tracks changes over time, allowing the model to understand motion patterns (e.g., running, fighting vs. walking).
    *   We use the output from the final time step of the sequence to classify the entire clip.

3.  **Classification Head**:
    *   A fully connected (Linear) layer maps the LSTM's final hidden state to the 3 class scores (Normal, Violent, Panic).

### 1.3 Training Pipeline
*   **Preprocessing**: Videos are resized to 224x224 pixels. We extract fixed-length sequences of 16 frames.
*   **Normalization**: ImageNet mean/std applied per frame to align with ResNet50 pretraining.
*   **Splits**: Train / Val / Test with fixed seed (default 70/15/15).
*   **Class balancing**: Optional weighted sampler on the training split.
*   **Optimizer / Loss**: Adam (LR=1e-4) with CrossEntropyLoss.
*   **Scheduler**: StepLR (gamma=0.5 every 5 epochs).
*   **Mixed Precision**: Enabled on CUDA for speed (can be disabled).
*   **Epochs**: Defaults to 15 (configurable in `main.py`).

## 2. Usage

### 2.1 Installation
1.  Create a virtual environment:
    ```bash
    python -m venv .venv
    .venv\Scripts\Activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you have a CUDA-compatible GPU for faster training, though CPU is supported.*

### 2.2 Data Setup
1.  Ensure the **Violent Flow** dataset is in the `Violent flow` directory.
2.  Ensure the **Avenue Dataset** is in the `Avenue Dataset` directory (extracted).
    *   If missing, run `python download_data.py` (or manually place `Avenue_Dataset.zip` in the root if the script fails).

### 2.3 Training & Evaluation
Run the main pipeline:
```bash
python main.py
```
This will:
1.  Load and process the datasets.
2.  Train the CNN-LSTM model with train/val splits (weighted sampler optional).
3.  Evaluate on validation and test splits.
4.  Generate and save visualization results under `docs/val/` and `docs/test/`.

## 3. Results & Visualization
Key outputs are saved in `docs/val/` and `docs/test/`:

*   **`confusion_matrix.png`**: Misclassification heatmap.
*   **`roc_curves.png`**: ROC curves per class with AUC.
*   **`pr_curves.png`**: Precision-Recall curves per class with AUC.
*   **`loss_curve.png`**, **`accuracy_curve.png`**: Training vs Validation over epochs (from history).
*   **`error_analysis.png`**: Misclassified counts per class.
*   **`metrics.txt`**: Weighted + per-class metrics.
*   **`sample_predictions.txt`**: Random sample predictions.

## 4. Inference
Use `predict.py` for single-video inference:
```bash
python predict.py --video /path/to/video.avi --checkpoint model_checkpoint.pth --classes Normal Violent Panic
```

## 5. Notes and Recommendations
*   **Panic data quality**: Replace the Avenue proxy with real panic/evacuation clips or labeled abnormal clips from Avenue for higher fidelity.
*   **Large artifacts**: `model_checkpoint.pth` (~105 MB) exceeds GitHub’s standard limit—use Git LFS or export a smaller ONNX/quantized model for sharing.
*   **Reproducibility**: Seeds are set (Python/NumPy/Torch) and cuDNN deterministic mode is enabled.
*   **Stability of Panic metrics**: Panic support is currently tiny (val/test ≤ 6 clips); bootstrap CIs are therefore extremely wide despite perfect point metrics. Prefer collecting more Panic clips and/or using stratified k-fold evaluation plus class-weighted loss or targeted augmentation to stabilize estimates.

## 4. Performance Analysis
*   **Accuracy**: Overall correctness of the model.
*   **Precision**: Reliability of positive predictions for each class.
*   **Recall**: Ability of the model to find all instances of a class.
*   **F1-Score**: Harmonic mean of Precision and Recall, useful for imbalanced datasets.
*   **Error Analysis**: The error plot helps identify if the model confuses 'Normal' with 'Panic' or 'Violent', guiding further improvements (e.g., more data or tuning).
