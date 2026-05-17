import os
import matplotlib.pyplot as plt
from PIL import Image

# =========================================================
# BREAKHIS DATASET PATH
# =========================================================

dataset_path = "BreaKHis_v1/histology_slides/breast"

# =========================================================
# FIND ONE 400X IMAGE FROM EACH CLASS
# =========================================================

sample_images = {}

classes = ["benign", "malignant"]

for cls in classes:

    class_path = os.path.join(dataset_path, cls)

    found = False

    for root, dirs, files in os.walk(class_path):

        # Select only 400X folders
        if "400X" in root:

            for file in files:

                if file.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):

                    image_path = os.path.join(root, file)

                    sample_images[cls] = image_path

                    found = True
                    break

        if found:
            break

# =========================================================
# PLOT IMAGES
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, (cls, img_path) in zip(axes, sample_images.items()):

    img = Image.open(img_path)

    ax.imshow(img)

    ax.set_title(f"{cls.capitalize()} - 400X")

    ax.axis("off")

plt.suptitle("BreakHis Dataset Sample Images (400X Magnification)",
             fontsize=14)

plt.tight_layout()

plt.show()

# =========================================================
# PRINT IMAGE PATHS
# =========================================================

print("\nSelected Sample Images:\n")

for cls, path in sample_images.items():

    print(f"{cls} --> {path}")