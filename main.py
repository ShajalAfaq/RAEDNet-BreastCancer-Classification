
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    roc_curve,
    auc,
    classification_report,
    confusion_matrix
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

if not os.path.exists(dataset_path):

    raise FileNotFoundError(
        f"Dataset path not found:\n{dataset_path}"
    )

# =========================================================
# TEMP TRANSFORM FOR MEAN/STD
# =========================================================

temp_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor()

])

# =========================================================
# TEMP DATASET
# =========================================================

temp_dataset = ImageFolder(
    root=dataset_path,
    transform=temp_transform
)

temp_loader = DataLoader(
    temp_dataset,
    batch_size=64,
    shuffle=False
)

# =========================================================
# COMPUTE MEAN & STD
# =========================================================

mean = 0.
std = 0.
total_images = 0

print("\nComputing Dataset Mean & Std...")

for images, _ in temp_loader:

    batch_samples = images.size(0)

    images = images.view(
        batch_samples,
        images.size(1),
        -1
    )

    mean += images.mean(2).sum(0)

    std += images.std(2).sum(0)

    total_images += batch_samples

mean /= total_images
std /= total_images

mean = mean.tolist()
std = std.tolist()

print("Mean:", mean)
print("Std :", std)

# =========================================================
# DATA AUGMENTATION
# =========================================================

train_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomRotation((0, 90)),

    transforms.RandomAffine(
        degrees=0,
        translate=(56/224, 56/224)
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomVerticalFlip(p=0.5),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=mean,
        std=std
    )

])

test_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=mean,
        std=std
    )

])

# =========================================================
# LOAD DATASET
# =========================================================

full_dataset = ImageFolder(root=dataset_path)

train_size = int(0.7 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size]
)

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = test_transform
test_dataset.dataset.transform = test_transform

# =========================================================
# SWISH
# =========================================================

class Swish(nn.Module):

    def forward(self, x):

        return x * torch.sigmoid(x)

# =========================================================
# DEPTHWISE SEPARABLE CONV
# =========================================================

class DSConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1
    ):

        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):

        x = self.depthwise(x)

        x = self.pointwise(x)

        return x

# =========================================================
# SE BLOCK
# =========================================================

class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Linear(channels, channels // reduction),

            nn.ReLU(),

            nn.Linear(channels // reduction, channels),

            nn.Sigmoid()

        )

    def forward(self, x):

        b, c, _, _ = x.size()

        y = self.pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y

# =========================================================
# SPATIAL ATTENTION
# =========================================================

class SpatialAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3
        )

    def forward(self, x):

        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        y = torch.cat(
            [avg_out, max_out],
            dim=1
        )

        y = torch.sigmoid(
            self.conv(y)
        )

        return x * y

# =========================================================
# NON LOCAL SELF ATTENTION
# =========================================================

class NonLocalBlock(nn.Module):

    def __init__(self, in_channels):

        super().__init__()

        self.query = nn.Conv2d(
            in_channels,
            in_channels // 2,
            1
        )

        self.key = nn.Conv2d(
            in_channels,
            in_channels // 2,
            1
        )

        self.value = nn.Conv2d(
            in_channels,
            in_channels,
            1
        )

    def forward(self, x):

        b, c, h, w = x.size()

        query = self.query(x).view(
            b,
            -1,
            h*w
        )

        key = self.key(x).view(
            b,
            -1,
            h*w
        )

        value = self.value(x).view(
            b,
            -1,
            h*w
        )

        attention = torch.softmax(

            torch.bmm(
                query.permute(0,2,1),
                key
            ),

            dim=-1
        )

        out = torch.bmm(
            value,
            attention.permute(0,2,1)
        )

        out = out.view(b, c, h, w)

        return x + out

# =========================================================
# CBAM
# =========================================================

class CBAM(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.channel_attention = SEBlock(channels)

        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        x = self.channel_attention(x)

        x = self.spatial_attention(x)

        return x

# =========================================================
# CHANNEL GATED RESIDUAL
# =========================================================

class CGR(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(
            channels,
            channels // 16,
            1
        )

        self.conv2 = nn.Conv2d(
            channels // 16,
            channels,
            1
        )

    def forward(self, x):

        y = self.pool(x)

        y = F.relu(self.conv1(y))

        y = torch.sigmoid(self.conv2(y))

        return x * y + x

# =========================================================
# REDB BLOCK
# =========================================================

class REDB(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        attention_type=None,
        dilation=False
    ):

        super().__init__()

        self.bottleneck = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

        self.conv3 = DSConv(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.conv5 = DSConv(
            out_channels,
            out_channels,
            kernel_size=5,
            padding=2
        )

        self.conv7 = DSConv(
            out_channels,
            out_channels,
            kernel_size=7,
            padding=3
        )

        self.project = nn.Conv2d(
            out_channels * 3,
            out_channels,
            kernel_size=1
        )

        self.se = SEBlock(out_channels)

        self.bn = nn.BatchNorm2d(out_channels)

        self.act = Swish()

        self.residual = nn.Identity()

        if attention_type == "sa":

            self.residual = SpatialAttention()

        elif attention_type == "nlsa":

            self.residual = NonLocalBlock(out_channels)

        elif attention_type == "cbam":

            self.residual = CBAM(out_channels)

        elif attention_type == "cgr":

            self.residual = CGR(out_channels)

        self.dilation = dilation

        if dilation:

            self.dilated = DSConv(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=2,
                dilation=2
            )

    def forward(self, x):

        identity = x

        x = self.bottleneck(x)

        x1 = self.conv3(x)

        x2 = self.conv5(x)

        x3 = self.conv7(x)

        x = torch.cat(
            [x1, x2, x3],
            dim=1
        )

        x = self.project(x)

        if self.dilation:

            x = self.dilated(x)

        x = self.se(x)

        x = self.residual(x)

        x = self.bn(x)

        x = self.act(x)

        if identity.shape == x.shape:

            x = x + identity

        return x

# =========================================================
# TRANSITION BLOCK
# =========================================================

class TransitionBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout_rate=0.3
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1
            ),

            nn.BatchNorm2d(out_channels),

            Swish(),

            nn.Dropout(dropout_rate),

            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            )

        )

    def forward(self, x):

        return self.block(x)

# =========================================================
# STEM BLOCK
# =========================================================

class StemBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),

            Swish(),

            DSConv(
                64,
                64,
                kernel_size=3,
                padding=1
            )

        )

    def forward(self, x):

        return self.block(x)

# =========================================================
# RAEDNET
# =========================================================

class RAEDNet(nn.Module):

    def __init__(
        self,
        num_classes=2,
        dropout_rate=0.5
    ):

        super().__init__()

        self.stem = StemBlock()

        self.redb1 = REDB(
            64,
            64,
            attention_type="sa"
        )

        self.trans1 = TransitionBlock(
            64,
            128,
            dropout_rate
        )

        self.redb2 = REDB(
            128,
            128,
            attention_type="nlsa"
        )

        self.trans2 = TransitionBlock(
            128,
            256,
            dropout_rate
        )

        self.redb3 = REDB(
            256,
            256,
            attention_type="cbam"
        )

        self.trans3 = TransitionBlock(
            256,
            512,
            dropout_rate
        )

        self.redb4 = REDB(
            512,
            512,
            dilation=True
        )

        self.trans4 = TransitionBlock(
            512,
            512,
            dropout_rate
        )

        self.redb5 = REDB(
            512,
            512,
            attention_type="cgr"
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.classifier = nn.Sequential(

            nn.Linear(1024, 512),

            Swish(),

            nn.Dropout(dropout_rate),

            nn.Linear(512, num_classes)

        )

    def forward(self, x):

        x = self.stem(x)

        x = self.redb1(x)
        x = self.trans1(x)

        x = self.redb2(x)
        x = self.trans2(x)

        x = self.redb3(x)
        x = self.trans3(x)

        x = self.redb4(x)
        x = self.trans4(x)

        x = self.redb5(x)

        gap = self.gap(x)
        gmp = self.gmp(x)

        gap = gap.view(gap.size(0), -1)
        gmp = gmp.view(gmp.size(0), -1)

        x = torch.cat([gap, gmp], dim=1)

        x = self.classifier(x)

        return x

# =========================================================
# CEAO HYPERPARAMETER OPTIMIZATION
# =========================================================

print("\nStarting CEAO Optimization...")

search_agents = 3
iterations = 3

best_accuracy = 0.0
best_params = None

for agent in range(search_agents):

    chaotic = random.random()

    chaotic = 4 * chaotic * (1 - chaotic)

    lr = 10 ** np.random.uniform(-5, -3)

    dropout_rate = np.random.uniform(0.2, 0.5)

    batch_size = random.choice([8, 16])

    weight_decay = 10 ** np.random.uniform(-6, -3)

    print("\n=================================")
    print(f"AGENT {agent+1}")
    print("=================================")

    print("Learning Rate :", lr)
    print("Dropout Rate  :", dropout_rate)
    print("Batch Size    :", batch_size)
    print("Weight Decay  :", weight_decay)

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

    model = RAEDNet(
        num_classes=len(full_dataset.classes),
        dropout_rate=dropout_rate
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    epochs = 3

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    # =====================================================
    # TRAINING LOOP
    # =====================================================

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

        train_loss = running_loss / len(train_loader)

        train_acc =  correct / total

        train_losses.append(train_loss)

        train_accuracies.append(train_acc)

        # =================================================
        # VALIDATION
        # =================================================

        model.eval()

        running_loss = 0.0

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        val_loss = running_loss / len(val_loader)

        val_acc =  correct / total

        val_losses.append(val_loss)

        val_accuracies.append(val_acc)

        print(
            f"\nEpoch [{epoch+1}/{epochs}] "
            f"Train Acc: {train_acc:.2f}% "
            f"Val Acc: {val_acc:.2f}%"
        )

    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    if val_acc > best_accuracy:

        best_accuracy = val_acc

        best_params = {

            "lr": lr,
            "dropout": dropout_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay

        }

        torch.save(
            model.state_dict(),
            "best_raednet_model.pth"
        )

# =========================================================
# BEST PARAMETERS
# =========================================================

print("\n=================================")
print("BEST HYPERPARAMETERS")
print("=================================")

print(best_params)

print(f"Best Validation Accuracy: {best_accuracy:.2f}%")

# =========================================================
# LOAD BEST MODEL
# =========================================================

model.load_state_dict(
    torch.load("best_raednet_model.pth")
)

# =========================================================
# TEST EVALUATION
# =========================================================

model.eval()

all_labels = []
all_preds = []
all_probs = []

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_preds.extend(
            predicted.cpu().numpy()
        )

        all_probs.extend(
            probs[:,1].cpu().numpy()
        )

# =========================================================
# TEST ACCURACY
# =========================================================

test_accuracy = 100 * correct / total

print("\n=================================")
print("TEST RESULTS")
print("=================================")

print(f"Test Accuracy: {test_accuracy:.2f}%")

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:\n")

print(

    classification_report(
        all_labels,
        all_preds,
        target_names=full_dataset.classes
    )

)


plt.figure(figsize=(8,6))

plt.plot(
    train_accuracies,
    label="Train Accuracy"
)

plt.plot(
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.title("Accuracy vs Epochs")

plt.legend()

# plt.grid(True)

plt.savefig("Results//accuracy_vs_epochs.png")

plt.show()

# =========================================================
# ROC CURVE
# =========================================================

fpr, tpr, thresholds = roc_curve(
    all_labels,
    all_probs
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

# plt.grid(True)

plt.savefig("Results//roc_curve.png")

plt.show()

# =========================================================
# SAVE FINAL MODEL
# =========================================================

torch.save(
    model.state_dict(),
    "final_raednet_model.pth"
)

print("\nFinal Model Saved!")

# =========================================================
# SAVE TRAINING HISTORY
# =========================================================

history = {

    "train_loss": train_losses,
    "val_loss": val_losses,
    "train_accuracy": train_accuracies,
    "val_accuracy": val_accuracies

}

torch.save(
    history,
    "training_history.pth"
)

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n=================================")
print("RAEDNet PIPELINE COMPLETED")
print("=================================")

print("\nSaved Files:")