# Crowd Behavior Analysis System

This project implements an automated system for multi-label classification of crowd behavior into three categories: **Normal**, **Violent**, and **Panic**. It utilizes deep learning techniques (CNN-LSTM) to analyze video sequences and detect abnormal events.

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
*   **Augmentation**: Basic normalization is applied.
*   **Optimizer**: Adam optimizer (Learning Rate: 1e-4) is used for stable convergence.
*   **Loss Function**: CrossEntropyLoss is used, suitable for multi-class classification.
*   **Epochs**: The model is trained for 15 epochs to ensure sufficient learning without overfitting.

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
2.  Train the CNN-LSTM model.
3.  Evaluate on a validation set.
4.  Generate and save visualization results in the `docs/` folder.

## 3. Results & Visualization
All results are saved in the `docs/` directory:

*   **`confusion_matrix.png`**: A heatmap showing True vs. Predicted labels to visualize misclassifications.
*   **`roc_curves.png`**: Receiver Operating Characteristic curves for each class, showing the trade-off between sensitivity and specificity (AUC scores included).
*   **`loss_curve.png`**: Plot of Training vs. Validation Loss over epochs.
*   **`accuracy_curve.png`**: Plot of Training vs. Validation Accuracy over epochs.
*   **`error_analysis.png`**: Bar chart showing which classes are most frequently misclassified.
*   **`metrics.txt`**: Text file containing final Accuracy, Weighted Precision, Recall, and F1-Score.
*   **`sample_predictions.txt`**: A log of random sample predictions (Actual vs. Predicted).

## 4. Performance Analysis
*   **Accuracy**: Overall correctness of the model.
*   **Precision**: Reliability of positive predictions for each class.
*   **Recall**: Ability of the model to find all instances of a class.
*   **F1-Score**: Harmonic mean of Precision and Recall, useful for imbalanced datasets.
*   **Error Analysis**: The error plot helps identify if the model confuses 'Normal' with 'Panic' or 'Violent', guiding further improvements (e.g., more data or tuning).
