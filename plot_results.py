

import pickle
import matplotlib.pyplot as plt
import os

import numpy as np

# =========================================================
# Load files
# =========================================================

binary_path = os.path.join("preevaluated", "binary_roc.pkl")
multiclass_path = os.path.join("preevaluated", "multiclass_roc.pkl")

with open(binary_path, "rb") as f:
    binary_roc = pickle.load(f)

with open(multiclass_path, "rb") as f:
    multiclass_roc = pickle.load(f)

# =========================================================
# Create Figure
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# =========================================================
# LEFT PLOT
# Binary ROC
# =========================================================

axes[0].plot(
    binary_roc["fpr"],
    binary_roc["tpr"],
    linewidth=2,
    label=f'Proposed Model (AUC = {binary_roc["auc"]:.4f})'
)

# Diagonal reference line
axes[0].plot([0, 1], [0, 1], linestyle='--')

# Title and labels
axes[0].set_title(
    'ROC Curve for Binary Classification (BACH)',
    fontsize=13,
    fontweight='bold'
)

axes[0].set_xlabel(
    'False Positive Rate',
    fontsize=11,
    fontweight='bold'
)

axes[0].set_ylabel(
    'True Positive Rate',
    fontsize=11,
    fontweight='bold'
)

# Bold ticks
for label in axes[0].get_xticklabels():
    label.set_fontweight('bold')

for label in axes[0].get_yticklabels():
    label.set_fontweight('bold')

axes[0].grid(True)
axes[0].legend(loc='lower right', prop={'weight': 'bold'})


# =========================================================
# RIGHT PLOT
# Multiclass ROC
# =========================================================

fpr_multi = multiclass_roc["fpr"]

for class_name in ["class_0", "class_1", "class_2", "class_3"]:

    axes[1].plot(
        fpr_multi,
        multiclass_roc[class_name]["tpr"],
        linewidth=2,
        label=f'{class_name.replace("_", " ").title()} '
              f'(AUC = {multiclass_roc[class_name]["auc"]:.4f})'
    )

# Diagonal reference line
axes[1].plot([0, 1], [0, 1], linestyle='--')

# Title and labels
axes[1].set_title(
    'ROC Curves for Multiclass Classification (BACH)',
    fontsize=13,
    fontweight='bold'
)

axes[1].set_xlabel(
    'False Positive Rate',
    fontsize=11,
    fontweight='bold'
)

axes[1].set_ylabel(
    'True Positive Rate',
    fontsize=11,
    fontweight='bold'
)

# Bold ticks
for label in axes[1].get_xticklabels():
    label.set_fontweight('bold')

for label in axes[1].get_yticklabels():
    label.set_fontweight('bold')

axes[1].grid(True)

axes[1].legend(
    loc='lower right',
    fontsize=9,
    prop={'weight': 'bold'}
)

# =========================================================
# Save Figure
# =========================================================

plt.tight_layout()

plt.savefig(
    "ROC_Curves_BACH.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("ROC curves plotted successfully .")


# =========================================================
# Create directory
# =========================================================

save_dir = "preevaluated"
os.makedirs(save_dir, exist_ok=True)

# =========================================================
# Epochs
# =========================================================

epochs = np.arange(1, 51)


# =========================================================
# LOAD  FILES
# =========================================================

with open(os.path.join(save_dir, "binary_accuracy.pkl"), "rb") as f:
    binary_loaded = pickle.load(f)

with open(os.path.join(save_dir, "multiclass_accuracy.pkl"), "rb") as f:
    multi_loaded = pickle.load(f)

# =========================================================
# Plotting
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# =========================================================
# LEFT PLOT - Binary Classification
# =========================================================

axes[0].plot(
    epochs,
    binary_loaded["train_accuracy"],
    linewidth=2,
    label='Training Accuracy'
)

axes[0].plot(
    epochs,
    binary_loaded["val_accuracy"],
    linewidth=2,
    label='Validation Accuracy'
)

axes[0].set_title(
    'Training and Validation Accuracy (Binary Classification)- BACH',
    fontsize=12,
    fontweight='bold'
)

axes[0].set_xlabel(
    'Epochs',
    fontsize=11,
    fontweight='bold'
)

axes[0].set_ylabel(
    'Accuracy (%)',
    fontsize=11,
    fontweight='bold'
)

for label in axes[0].get_xticklabels():
    label.set_fontweight('bold')

for label in axes[0].get_yticklabels():
    label.set_fontweight('bold')

axes[0].grid(True)

axes[0].legend(
    loc='lower right',
    prop={'weight':'bold'}
)

# =========================================================
# RIGHT PLOT - Multiclass Classification
# =========================================================

axes[1].plot(
    multi_loaded["epochs"],
    multi_loaded["train_accuracy"],
    linewidth=2,
    label='Training Accuracy'
)

axes[1].plot(
    multi_loaded["epochs"],
    multi_loaded["val_accuracy"],
    linewidth=2,
    label='Validation Accuracy'
)

axes[1].set_title(
    'Training and Validation Accuracy (Multiclass Classification)-BACH',
    fontsize=12,
    fontweight='bold'
)

axes[1].set_xlabel(
    'Epochs',
    fontsize=11,
    fontweight='bold'
)

axes[1].set_ylabel(
    'Accuracy (%)',
    fontsize=11,
    fontweight='bold'
)

for label in axes[1].get_xticklabels():
    label.set_fontweight('bold')

for label in axes[1].get_yticklabels():
    label.set_fontweight('bold')

axes[1].grid(True)

axes[1].legend(
    loc='upper left',
    prop={'weight':'bold'}
)

# =========================================================
# Save Figure
# =========================================================

plt.tight_layout()

plt.savefig(
    "Training_Validation_Accuracy_BACH.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("Accuracy curves plotted successfully from .")


# =========================================================
# LOAD
# =========================================================

with open(os.path.join(save_dir, "binary_proposed_roc.pkl"), "rb") as f:
    binary_loaded = pickle.load(f)

with open(os.path.join(save_dir, "multiclass_proposed_roc.pkl"), "rb") as f:
    multi_loaded = pickle.load(f)

# =========================================================
# Create Figure
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# =========================================================
# LEFT PLOT
# Binary Classification ROC
# =========================================================

fpr_binary = binary_loaded["fpr"]

for key in ["MF-40x", "MF-100x", "MF-200x", "MF-400x"]:

    axes[0].plot(
        fpr_binary,
        binary_loaded[key]["tpr"],
        linewidth=2,
        label=f'{key} (AUC = {binary_loaded[key]["auc"]:.4f})'
    )

# Diagonal line
axes[0].plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    color='gray'
)

axes[0].set_title(
    'ROC Curve for Binary Classification (Proposed Model)',
    fontsize=12,
    fontweight='bold'
)

axes[0].set_xlabel(
    'False Positive Rate',
    fontsize=11,
    fontweight='bold'
)

axes[0].set_ylabel(
    'True Positive Rate',
    fontsize=11,
    fontweight='bold'
)

# Bold ticks
for label in axes[0].get_xticklabels():
    label.set_fontweight('bold')

for label in axes[0].get_yticklabels():
    label.set_fontweight('bold')

# axes[0].grid(True)

axes[0].legend(
    loc='lower right',
    fontsize=8,
    prop={'weight':'bold'}
)

# =========================================================
# RIGHT PLOT
# Multiclass Classification ROC
# =========================================================

fpr_multi = multi_loaded["fpr"]

for key in ["MF-40x", "MF-100x", "MF-200x", "MF-400x"]:

    axes[1].plot(
        fpr_multi,
        multi_loaded[key]["tpr"],
        linewidth=2,
        label=f'{key} (AUC = {multi_loaded[key]["auc"]:.4f})'
    )

# Diagonal line
axes[1].plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    color='gray'
)

axes[1].set_title(
    'ROC Curve for Multiclass Classification (Proposed Model)',
    fontsize=12,
    fontweight='bold'
)

axes[1].set_xlabel(
    'False Positive Rate',
    fontsize=11,
    fontweight='bold'
)

axes[1].set_ylabel(
    'True Positive Rate',
    fontsize=11,
    fontweight='bold'
)

# Bold ticks
for label in axes[1].get_xticklabels():
    label.set_fontweight('bold')

for label in axes[1].get_yticklabels():
    label.set_fontweight('bold')

# axes[1].grid(True)

axes[1].legend(
    loc='lower right',
    fontsize=8,
    prop={'weight':'bold'}
)

# =========================================================
# Save Figure
# =========================================================

plt.tight_layout()

plt.savefig(
    "ROC_Proposed_ModelBreakhis.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("Figure saved as ROC_Proposed_Model.png")

# =========================================================
# Epochs
# =========================================================

epochs = np.arange(1, 51)

with open(os.path.join(save_dir, "binary_training_accuracy.pkl"), "rb") as f:
    binary_loaded = pickle.load(f)

with open(os.path.join(save_dir, "multiclass_training_accuracy.pkl"), "rb") as f:
    multi_loaded = pickle.load(f)

# =========================================================
# Plot Figure
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# =========================================================
# LEFT PLOT - Binary Classification
# =========================================================

axes[0].plot(
    epochs,
    binary_loaded["train_accuracy"],
    linewidth=2,
    label='Training Accuracy'
)

axes[0].plot(
    epochs,
    binary_loaded["val_accuracy"],
    linewidth=2,
    label='Validation Accuracy'
)

axes[0].set_title(
    'Training and Validation Accuracy (Binary Classification)',
    fontsize=12,
    fontweight='bold'
)

axes[0].set_xlabel(
    'Epochs',
    fontsize=11,
    fontweight='bold'
)

axes[0].set_ylabel(
    'Accuracy',
    fontsize=11,
    fontweight='bold'
)

for label in axes[0].get_xticklabels():
    label.set_fontweight('bold')

for label in axes[0].get_yticklabels():
    label.set_fontweight('bold')

# axes[0].grid(True)

axes[0].legend(
    loc='lower right',
    prop={'weight':'bold'}
)

# =========================================================
# RIGHT PLOT - Multiclass Classification
# =========================================================

axes[1].plot(
    multi_loaded["epochs"],
    multi_loaded["train_accuracy"],
    linewidth=2,
    label='Training Accuracy'
)

axes[1].plot(
    multi_loaded["epochs"],
    multi_loaded["val_accuracy"],
    linewidth=2,
    label='Validation Accuracy'
)

axes[1].set_title(
    'Training and Validation Accuracy (Multiclass Classification)',
    fontsize=12,
    fontweight='bold'
)

axes[1].set_xlabel(
    'Epochs',
    fontsize=11,
    fontweight='bold'
)

axes[1].set_ylabel(
    'Accuracy',
    fontsize=11,
    fontweight='bold'
)

for label in axes[1].get_xticklabels():
    label.set_fontweight('bold')

for label in axes[1].get_yticklabels():
    label.set_fontweight('bold')

# axes[1].grid(True)

axes[1].legend(
    loc='lower right',
    prop={'weight':'bold'}
)

# =========================================================
# Save Figure
# =========================================================

plt.tight_layout()

plt.savefig(
    "Training_Validation_Accuracy_Breakhis.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()