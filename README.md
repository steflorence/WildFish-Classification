
# WildFish Image Classification using ConvNeXt Tiny

This project is part of the KD34403 Applied AI Prototype Assignment. It aims to build an AI model that can classify different fish species based on uploaded images using transfer learning with a pretrained ConvNeXt Tiny model. The system is designed for aquaculture applications to assist in fish species identification and monitoring.

---

## Project Objective

To develop a fish classification prototype that:
- Accurately classifies uploaded fish images
- Supports real-time prediction through a simple UI
- Saves user input and results for future analysis
- Rejects non-fish or unclear inputs using confidence thresholding
- Demonstrates a full AI pipeline from data to deployment

---

## Dataset

The dataset used is stored in:


- Each folder represents a fish species (label).
- Images are real-world fish photos, preprocessed using `torchvision.transforms` from the pretrained weights.

---

## Model Architecture

- **Base Model**: ConvNeXt Tiny (from `torchvision.models`)
- **Transfer Learning**: We use pretrained ImageNet weights and fine-tune the final classification layer.
- **Output Layer**: Modified to match the number of fish classes dynamically.

---

## Training Configuration

- **Split**: 70% training, 20% validation, 10% test
- **Batch Size**: 32
- **Epochs**: 4 (can be adjusted)
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix

All class mappings are saved to `class_names.json` for compatibility with the deployed UI.

---

## Evaluation Results

| Metric          | Value         |
|-----------------|---------------|
| Test Accuracy   | 97.79%        |
| Loss (Test Set) | 0.2367        |

A detailed classification report and confusion matrix are printed after training.

---

## Gradio App Features

The system includes an interactive UI built with [Gradio](https://gradio.app), allowing users to:

Upload fish images  
View top-3 predictions with confidence  
See accuracy score (confidence %)  
Reject unclear or non-fish images  
Save high-confidence predictions (≥90%) to a local dataset  
View & download prediction history from `prediction_history.csv`

---

## Project Files

- `train_fish_classifier.py` – Model training script
- `convnext_fish_classifier.pth` – Trained model weights
- `class_names.json` – Class name mapping
- `fish_classifier_app.py` – Gradio UI for testing/inference
- `prediction_history.csv` – Auto-generated log of predictions
- `Saved_HighConfidence/` – Saved images with high confidence
- `README.md` – Project description

---

## How to Run

```bash
# Train the model (optional if model is already trained)
python train_fish_classifier.py

# Run the Gradio UI
python fish_classifier_app.py
