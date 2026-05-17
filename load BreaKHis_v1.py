import os
import torch

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

# =========================================================
# DATASET PATH
# =========================================================

dataset_path = "BreaKHis_v1/histology_slides/breast"

# =========================================================
# CHECK PATH
# =========================================================

print("\nChecking Dataset Path...")

if not os.path.exists(dataset_path):

    raise FileNotFoundError(
        f"\nDataset path not found:\n{dataset_path}"
    )

print("Dataset Path Exists.")

# =========================================================
# IMAGE TRANSFORMS
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================================================
# LOAD DATASET
# =========================================================

dataset = ImageFolder(
    root=dataset_path,
    transform=transform
)

# =========================================================
# DATASET INFORMATION
# =========================================================

print("\n===================================================")
print("BREAKHIS DATASET INFORMATION")
print("===================================================")

print("\nTotal Images Found:", len(dataset))

print("\nClasses:")
print(dataset.classes)

print("\nClass Mapping:")
print(dataset.class_to_idx)

# =========================================================
# TRAIN / VALIDATION / TEST SPLIT
# =========================================================

train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

print("\nTrain Size      :", len(train_dataset))
print("Validation Size :", len(val_dataset))
print("Test Size       :", len(test_dataset))

# =========================================================
# DATALOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# =========================================================
# TEST DATALOADER
# =========================================================

print("\n===================================================")
print("TESTING DATALOADER")
print("===================================================")

for images, labels in train_loader:

    print("\nImages Shape :", images.shape)
    print("Labels Shape :", labels.shape)

    print("\nNumeric Labels:")
    print(labels)

    print("\nString Labels:")

    label_names = [dataset.classes[label] for label in labels]

    print(label_names)

    break

print("\nDataset Loaded Successfully.")