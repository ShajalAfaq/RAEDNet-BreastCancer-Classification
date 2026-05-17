import os
import zipfile
import matplotlib.pyplot as plt
from PIL import Image

# =========================================================
# DATASET PATH
# =========================================================

dataset_path = "ICIAR2018_BACH_Challenge/Photos"

# =========================================================
# EXTRACT ZIP FILES
# =========================================================

print("Extracting ZIP files if needed...\n")

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.endswith(".zip"):

            zip_path = os.path.join(root, file)

            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(root)

                print("Extracted:", file)

            except Exception as e:
                print("Error:", file)
                print(e)

print("\nExtraction Completed.")

# =========================================================
# CLASS NAMES
# =========================================================

classes = [
    "Benign",
    "InSitu",
    "Invasive",
    "Normal"
]

# =========================================================
# FIND ONE SAMPLE IMAGE PER CLASS
# =========================================================

sample_images = {}

for cls in classes:

    class_path = os.path.join(dataset_path, cls)

    for file in os.listdir(class_path):

        if file.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):

            sample_images[cls] = os.path.join(class_path, file)

            break

# =========================================================
# PLOT : 1 ROW × 4 COLUMNS
# =========================================================

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

for ax, cls in zip(axes, classes):

    img_path = sample_images[cls]

    img = Image.open(img_path)

    ax.imshow(img)

    ax.set_title(cls, fontsize=13)

    ax.axis("off")

# plt.suptitle(
#     "BACH 2018 Dataset Sample Images",
#     fontsize=16
# )

plt.tight_layout()

plt.show()

# =========================================================
# PRINT IMAGE PATHS
# =========================================================

print("\nSelected Images:\n")

for cls, path in sample_images.items():

    print(f"{cls} --> {path}")