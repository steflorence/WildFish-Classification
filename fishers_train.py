import os
import json
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from sklearn.metrics import classification_report, confusion_matrix

# === Reproducibility ===
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# === Paths ===
data_dir = "/Users/steflorence/Documents/Workspace/Project/WildFish Data"
model_path = "convnext_fish_classifier.pth"
class_map_path = "class_names.json"

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# === Load Dataset ===
if not os.path.isdir(data_dir):
    raise FileNotFoundError(f" Dataset folder not found at: {data_dir}")

weights = ConvNeXt_Tiny_Weights.DEFAULT
transform = weights.transforms()

full_dataset = ImageFolder(data_dir, transform=transform)
class_names = full_dataset.classes
print(f"[INFO] Found classes: {class_names}")

# === Save Class Mapping ===
with open(class_map_path, "w") as f:
    json.dump(class_names, f)
print(f"[INFO] Saved class mapping to {class_map_path}")

# === Train/Val/Test Split ===
total_len = len(full_dataset)
train_len = int(0.7 * total_len)
val_len = int(0.2 * total_len)
test_len = total_len - train_len - val_len  # ensure total adds up
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_len, val_len, test_len])
print(f"[INFO] Dataset split — Train: {train_len}, Val: {val_len}, Test: {test_len}")

# === Data Loaders ===
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# === Load Pretrained Model ===
model = convnext_tiny(weights=weights)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(class_names))
model = model.to(device)

# === Loss & Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# === Evaluation Function ===
def evaluate(model, loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    return loss_sum / len(loader), acc

# === Training Loop ===
num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0
    start = time.time()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    val_loss, val_acc = evaluate(model, val_loader)
    duration = time.time() - start

    print(f"[{epoch+1}/{num_epochs}] "
          f"Train Loss: {running_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.2f}% | "
          f"Time: {duration:.1f}s")

# === Final Test Evaluation ===
def test_metrics(model, loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=2))

    cm = confusion_matrix(all_labels, all_preds)
    print("\n Confusion Matrix (rows=actual, columns=predicted):")
    print(cm)

    print("\nClass Index Mapping:")
    for i, name in enumerate(class_names):
        print(f"{i}: {name}")

# === Run Final Evaluation
test_loss, test_acc = evaluate(model, test_loader)
print(f"\n🧪 FINAL TEST RESULTS — Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}% on {len(test_dataset)} images")
test_metrics(model, test_loader, class_names)

# === Save Model ===
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
