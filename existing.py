
import os
import openpyxl
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# =========================================================
# DATASET PATH
# =========================================================

dataset_path = "BreaKHis_v1/histology_slides/breast"

# =========================================================
# TRANSFORMS
# =========================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(20),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])

# =========================================================
# LOAD DATASET
# =========================================================

dataset = ImageFolder(
    root=dataset_path,
    transform=transform
)

# =========================================================
# SPLIT DATASET
# =========================================================

train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(

    dataset,

    [train_size, val_size, test_size]

)

# =========================================================
# DATALOADERS
# =========================================================

batch_size = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# =========================================================
# NUMBER OF CLASSES
# =========================================================

num_classes = len(dataset.classes)

print("Classes:", dataset.classes)

# =========================================================
# SWISH
# =========================================================

class Swish(nn.Module):

    def forward(self, x):

        return x * torch.sigmoid(x)

# =========================================================
# SIMPLE RAEDNET
# =========================================================

class RAEDNet(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 64, 3, padding=1),

            nn.BatchNorm2d(64),

            Swish(),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),

            nn.BatchNorm2d(128),

            Swish(),

            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),

            nn.BatchNorm2d(256),

            Swish(),

            nn.AdaptiveAvgPool2d(1)

        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256, 128),

            Swish(),

            nn.Dropout(0.5),

            nn.Linear(128, num_classes)

        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

# =========================================================
# LOAD EXISTING MODELS
# =========================================================

models_dict = {}

# ---------------------------------------------------------
# VGG19
# ---------------------------------------------------------

vgg19 = models.vgg19(weights="DEFAULT")

vgg19.classifier[6] = nn.Linear(
    vgg19.classifier[6].in_features,
    num_classes
)

models_dict["VGG-19"] = vgg19.to(device)

# ---------------------------------------------------------
# EfficientNetV2
# ---------------------------------------------------------

efficientnet = models.efficientnet_v2_s(weights="DEFAULT")

efficientnet.classifier[1] = nn.Linear(
    efficientnet.classifier[1].in_features,
    num_classes
)

models_dict["EfficientNetV2"] = efficientnet.to(device)

# ---------------------------------------------------------
# MobileNetV2
# ---------------------------------------------------------

mobilenet = models.mobilenet_v2(weights="DEFAULT")

mobilenet.classifier[1] = nn.Linear(
    mobilenet.classifier[1].in_features,
    num_classes
)

models_dict["MobileNetV2"] = mobilenet.to(device)

# ---------------------------------------------------------
# Swin Transformer
# ---------------------------------------------------------

swin = models.swin_t(weights="DEFAULT")

swin.head = nn.Linear(
    swin.head.in_features,
    num_classes
)

models_dict["Swin Transformer"] = swin.to(device)

# ---------------------------------------------------------
# Proposed RAEDNet
# ---------------------------------------------------------

raednet = RAEDNet(num_classes=num_classes)

models_dict["RAEDNet"] = raednet.to(device)

# =========================================================
# TRAINING FUNCTION
# =========================================================

def train_model(model, epochs=3):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0001
    )

    best_model = copy.deepcopy(model.state_dict())

    best_acc = 0

    for epoch in range(epochs):

        model.train()

        running_corrects = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(
                preds == labels
            )

            total += labels.size(0)

        epoch_acc = running_corrects.double() / total

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Accuracy: {epoch_acc:.4f}"
        )

        if epoch_acc > best_acc:

            best_acc = epoch_acc

            best_model = copy.deepcopy(
                model.state_dict()
            )

    model.load_state_dict(best_model)

    return model

# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_model(model):

    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.numpy())

            y_pred.extend(preds.cpu().numpy())

            y_prob.extend(probs[:,1].cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    auc_score = roc_auc_score(
        y_true,
        y_prob
    )

    return (

        accuracy * 100,
        precision * 100,
        recall * 100,
        f1 * 100,
        auc_score * 100

    )

# =========================================================
# TRAIN & EVALUATE ALL MODELS
# =========================================================

results = []

for model_name, model in models_dict.items():

    print("\n================================================")
    print(f"TRAINING: {model_name}")
    print("================================================")

    trained_model = train_model(
        model,
        epochs=3
    )

    print(f"\nEvaluating {model_name}...")

    accuracy, precision, recall, f1, auc_value = evaluate_model(
        trained_model
    )

    results.append({

        "Method": model_name,

        "Accuracy (%)": round(accuracy, 2),

        "Precision (%)": round(precision, 2),

        "Recall (%)": round(recall, 2),

        "F1-Score (%)": round(f1, 2),

        "AUC (%)": round(auc_value, 2)

    })

    # =====================================================
    # SAVE MODEL
    # =====================================================

    model_filename = model_name.replace(" ", "_")

    torch.save(

        trained_model.state_dict(),

        f"{model_filename}.pth"

    )

    print(f"{model_name} Saved!")

# =========================================================
# RESULTS DATAFRAME
# =========================================================

df = pd.DataFrame(results)

# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n================================================")
print("FINAL RESULTS")
print("================================================")

print(df)

# =========================================================
# SAVE CSV
# =========================================================

csv_filename = "all_models_comparison.csv"

df.to_csv(
    csv_filename,
    index=False
)

print(f"\nCSV Saved: {csv_filename}")

print("\n================================================")
print("ALL MODELS IMPLEMENTED SUCCESSFULLY")
print("================================================")

print("\nSaved Models:")

for model_name in models_dict.keys():

    print(model_name.replace(' ', '_') + ".pth")