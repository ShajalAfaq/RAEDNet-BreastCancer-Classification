# =========================================================
# FULL RAEDNET ABLATION STUDY
# BREAKHIS + BACH DATASETS
# UPDATED FINAL FULL CODE
# =========================================================

import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# =========================================================
# DATASET PATHS
# =========================================================

breakhis_path = "BreaKHis_v1/histology_slides/breast"

bach_path = "ICIAR2018_BACH_Challenge/Photos"

# =========================================================
# CHECK DATASET PATHS
# =========================================================

if not os.path.exists(breakhis_path):

    raise FileNotFoundError(
        f"BreakHis dataset not found:\n{breakhis_path}"
    )

if not os.path.exists(bach_path):

    raise FileNotFoundError(
        f"BACH dataset not found:\n{bach_path}"
    )

# =========================================================
# TRANSFORMS
# =========================================================

train_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.RandomRotation((0,90)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.ToTensor()

])

test_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor()

])

# =========================================================
# LOAD DATASETS
# =========================================================

breakhis_dataset = ImageFolder(

    root=breakhis_path,
    transform=train_transform

)

bach_dataset = ImageFolder(

    root=bach_path,
    transform=train_transform

)

print("\nBreakHis Classes:")
print(breakhis_dataset.classes)

print("\nBACH Classes:")
print(bach_dataset.classes)

# =========================================================
# SPLIT FUNCTION
# =========================================================

def split_dataset(dataset):

    train_size = int(0.7 * len(dataset))

    val_size = int(0.1 * len(dataset))

    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(

        dataset,

        [train_size, val_size, test_size]

    )

    val_dataset.dataset.transform = test_transform

    test_dataset.dataset.transform = test_transform

    return train_dataset, val_dataset, test_dataset

# =========================================================
# SPLIT DATASETS
# =========================================================

breakhis_train, breakhis_val, breakhis_test = split_dataset(
    breakhis_dataset
)

bach_train, bach_val, bach_test = split_dataset(
    bach_dataset
)

# =========================================================
# DATALOADER FUNCTION
# =========================================================

def create_loaders(train_ds, val_ds, batch_size=16):

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader

# =========================================================
# SWISH
# =========================================================

class Swish(nn.Module):

    def forward(self, x):

        return x * torch.sigmoid(x)

# =========================================================
# DEPTHWISE SEPARABLE CONVOLUTION
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

            nn.Linear(channels, channels//reduction),

            nn.ReLU(),

            nn.Linear(channels//reduction, channels),

            nn.Sigmoid()

        )

    def forward(self, x):

        b,c,_,_ = x.size()

        y = self.pool(x).view(b,c)

        y = self.fc(y).view(b,c,1,1)

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

        max_out,_ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        y = torch.cat([avg_out,max_out], dim=1)

        y = torch.sigmoid(self.conv(y))

        return x * y

# =========================================================
# NON LOCAL SELF ATTENTION
# =========================================================

class NonLocalBlock(nn.Module):

    def __init__(self, in_channels):

        super().__init__()

        self.query = nn.Conv2d(
            in_channels,
            in_channels//2,
            1
        )

        self.key = nn.Conv2d(
            in_channels,
            in_channels//2,
            1
        )

        self.value = nn.Conv2d(
            in_channels,
            in_channels,
            1
        )

    def forward(self, x):

        b,c,h,w = x.size()

        query = self.query(x).view(b,-1,h*w)

        key = self.key(x).view(b,-1,h*w)

        value = self.value(x).view(b,-1,h*w)

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

        out = out.view(b,c,h,w)

        return x + out

# =========================================================
# CBAM
# =========================================================

class CBAM(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.channel = SEBlock(channels)

        self.spatial = SpatialAttention()

    def forward(self, x):

        x = self.channel(x)

        x = self.spatial(x)

        return x

# =========================================================
# CGR
# =========================================================

class CGR(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(
            channels,
            channels//16,
            1
        )

        self.conv2 = nn.Conv2d(
            channels//16,
            channels,
            1
        )

    def forward(self, x):

        y = self.pool(x)

        y = F.relu(self.conv1(y))

        y = torch.sigmoid(self.conv2(y))

        return x * y + x

# =========================================================
# REDB
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
            1
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
            out_channels*3,
            out_channels,
            1
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

        x = torch.cat([x1,x2,x3], dim=1)

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
                1
            ),

            nn.BatchNorm2d(out_channels),

            Swish(),

            nn.Dropout(dropout_rate),

            nn.AvgPool2d(2)

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

            DSConv(64,64)

        )

    def forward(self, x):

        return self.block(x)

# =========================================================
# ABLATION RAEDNET
# =========================================================

class AblationRAEDNet(nn.Module):

    def __init__(

        self,

        num_classes,

        use_sa=False,
        use_nlsa=False,
        use_cbam=False,
        use_dc=False,
        use_cgr=False,

        dropout_rate=0.3

    ):

        super().__init__()

        self.stem = StemBlock()

        self.redb1 = REDB(
            64,
            64,
            attention_type="sa" if use_sa else None
        )

        self.trans1 = TransitionBlock(64,128)

        self.redb2 = REDB(
            128,
            128,
            attention_type="nlsa" if use_nlsa else None
        )

        self.trans2 = TransitionBlock(128,256)

        self.redb3 = REDB(
            256,
            256,
            attention_type="cbam" if use_cbam else None
        )

        self.trans3 = TransitionBlock(256,512)

        self.redb4 = REDB(
            512,
            512,
            dilation=use_dc
        )

        self.trans4 = TransitionBlock(512,512)

        self.redb5 = REDB(
            512,
            512,
            attention_type="cgr" if use_cgr else None
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.classifier = nn.Sequential(

            nn.Linear(1024,512),

            Swish(),

            nn.Dropout(dropout_rate),

            nn.Linear(512,num_classes)

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

        gap = self.gap(x).view(x.size(0), -1)

        gmp = self.gmp(x).view(x.size(0), -1)

        x = torch.cat([gap,gmp], dim=1)

        x = self.classifier(x)

        return x

# =========================================================
# TRAIN & VALIDATION
# =========================================================

def train_and_evaluate(

    dataset_name,

    train_dataset,

    val_dataset,

    num_classes,

    use_sa=False,
    use_nlsa=False,
    use_cbam=False,
    use_dc=False,
    use_cgr=False,

    epochs=3

):

    train_loader, val_loader = create_loaders(
        train_dataset,
        val_dataset
    )

    model = AblationRAEDNet(

        num_classes=num_classes,

        use_sa=use_sa,
        use_nlsa=use_nlsa,
        use_cbam=use_cbam,
        use_dc=use_dc,
        use_cgr=use_cgr

    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    best_acc = 0.0

    for epoch in range(epochs):

        model.train()

        for images, labels in train_loader:

            images = images.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

        # VALIDATION

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)

                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs,1)

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        accuracy = 100 * correct / total

        if accuracy > best_acc:

            best_acc = accuracy

    return best_acc

# =========================================================
# ABLATION CONFIGS
# =========================================================

configs = [

    ("Variant 1", False, False, False, False, False),

    ("Variant 2", True, False, False, False, False),

    ("Variant 3", True, True, False, False, False),

    ("Variant 4", True, True, True, False, False),

    ("Variant 5", True, True, True, True, False),

    ("Variant 6", True, True, True, True, True)

]

# =========================================================
# RUN ABLATION
# =========================================================

results = []

print("\n================================================")
print("RUNNING ABLATION STUDY")
print("================================================")

for name, sa, nlsa, cbam, dc, cgr in configs:

    print(f"\n{name}")

    # -----------------------------------------------------
    # BREAKHIS
    # -----------------------------------------------------

    breakhis_acc = train_and_evaluate(

        "BreakHis",

        breakhis_train,

        breakhis_val,

        num_classes=len(breakhis_dataset.classes),

        use_sa=sa,
        use_nlsa=nlsa,
        use_cbam=cbam,
        use_dc=dc,
        use_cgr=cgr

    )

    # -----------------------------------------------------
    # BACH
    # -----------------------------------------------------

    bach_acc = train_and_evaluate(

        "BACH",

        bach_train,

        bach_val,

        num_classes=len(bach_dataset.classes),

        use_sa=sa,
        use_nlsa=nlsa,
        use_cbam=cbam,
        use_dc=dc,
        use_cgr=cgr

    )

    print(f"BreakHis Accuracy : {breakhis_acc:.2f}%")

    print(f"BACH Accuracy      : {bach_acc:.2f}%")

    attention_name = "None"

    if sa and not nlsa and not cbam and not dc and not cgr:

        attention_name = "SA"

    elif sa and nlsa and not cbam:

        attention_name = "SA + NLSA"

    elif sa and nlsa and cbam and not dc:

        attention_name = "SA + NLSA + CBAM"

    elif sa and nlsa and cbam and dc and not cgr:

        attention_name = "SA + NLSA + CBAM + DC"

    elif sa and nlsa and cbam and dc and cgr:

        attention_name = "Full Attention"

    results.append({

        "Model Variant": name,

        "REDB Blocks": "✓",

        "Attention Modules": attention_name,

        "CEAO": "✗",

        "BreakHis Accuracy (%)": round(breakhis_acc,2),

        "BACH Accuracy (%)": round(bach_acc,2)

    })

# =========================================================
# SAVE RESULTS
# =========================================================

results_df = pd.DataFrame(results)

os.makedirs("Results", exist_ok=True)

csv_path = "Results/Ablation_Study_70_10_20.csv"

# results_df.to_csv(csv_path, index=False)

# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n================================================")
print("FINAL ABLATION RESULTS")
print("================================================")

print(results_df)

print("\nCSV Saved At:")
print(csv_path)