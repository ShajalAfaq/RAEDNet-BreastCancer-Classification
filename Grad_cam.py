
# GRAD-CAM IMPLEMENTATION FOR RAEDNET

import cv2
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

dataset_path = "BreaKHis_v1/histology_slides/breast"

if not os.path.exists(dataset_path):

    raise FileNotFoundError(
        f"Dataset path not found:\n{dataset_path}"
    )

temp_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor()

])

temp_dataset = ImageFolder(
    root=dataset_path,
    transform=temp_transform
)

temp_loader = DataLoader(
    temp_dataset,
    batch_size=64,
    shuffle=False
)


# COMPUTE MEAN & STD


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

full_dataset = ImageFolder(root=dataset_path)

train_size = int(0.7 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size]
)

test_dataset.dataset.transform = test_transform



# GRAD-CAM CLASS


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model

        self.target_layer = target_layer

        self.gradients = None

        self.activations = None

        # Forward Hook
        self.target_layer.register_forward_hook(
            self.forward_hook
        )

        # Backward Hook
        self.target_layer.register_full_backward_hook(
            self.backward_hook
        )

    # -----------------------------------------------------

    def forward_hook(self, module, input, output):

        self.activations = output

    # -----------------------------------------------------

    def backward_hook(self, module, grad_input, grad_output):

        self.gradients = grad_output[0]

    # -----------------------------------------------------

    def generate_cam(self, input_image, class_idx=None):

        self.model.eval()

        output = self.model(input_image)

        # Predicted class
        if class_idx is None:

            class_idx = torch.argmax(output)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        loss = output[:, class_idx]

        loss.backward()

        # Gradients
        gradients = self.gradients[0]

        # Feature maps
        activations = self.activations[0]

        # Global Average Pooling
        weights = torch.mean(
            gradients,
            dim=(1, 2)
        )

        # Weighted combination
        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32
        ).to(device)

        for i, w in enumerate(weights):

            cam += w * activations[i]

        # ReLU
        cam = F.relu(cam)

        # Normalize
        cam -= cam.min()

        cam /= cam.max()

        return cam.cpu().detach().numpy()

image_path = r"BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A-14-22549AB/40X/SOB_B_A-14-22549AB-40-001.png"

print("\nSelected Image:")
print(image_path)

# LOAD ONE IMAGE


image_path = test_dataset.dataset.samples[
    test_dataset.indices[0]
][0]

print("\nSelected Image:")
print(image_path)


# ORIGINAL IMAGE


original_image = Image.open(image_path).convert("RGB")

original_image = original_image.resize((224,224))


# TRANSFORM IMAGE


input_tensor = test_transform(
    original_image
).unsqueeze(0).to(device)

# SWISH


class Swish(nn.Module):

    def forward(self, x):

        return x * torch.sigmoid(x)


# DEPTHWISE SEPARABLE CONV


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


# SE BLOCK


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


# SPATIAL ATTENTION


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


# NON LOCAL SELF ATTENTION


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


# CBAM


class CBAM(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.channel_attention = SEBlock(channels)

        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        x = self.channel_attention(x)

        x = self.spatial_attention(x)

        return x


# CHANNEL GATED RESIDUAL


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


# REDB BLOCK


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


# TRANSITION BLOCK


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


# STEM BLOCK


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


# RAEDNET


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



model = RAEDNet(
    num_classes=len(full_dataset.classes),
    dropout_rate=0.3
).to(device)

# ---------------------------------------------------------
# LOAD TRAINED WEIGHTS
# ---------------------------------------------------------

model.load_state_dict(

    torch.load(
        "final_raednet_model.pth",
        map_location=device
    )

)

model.eval()

print("\nRAEDNet Model Loaded Successfully!")

# Last convolution layer of RAEDNet

target_layer = model.redb5


# CREATE GRADCAM OBJECT


gradcam = GradCAM(
    model,
    target_layer
)


# GENERATE CAM


cam = gradcam.generate_cam(input_tensor)


# CONVERT IMAGE


original_np = np.array(original_image)


# RESIZE CAM


cam = cv2.resize(
    cam,
    (224,224)
)


# HEATMAP


heatmap = cv2.applyColorMap(

    np.uint8(255 * cam),

    cv2.COLORMAP_JET

)

heatmap = cv2.cvtColor(
    heatmap,
    cv2.COLOR_BGR2RGB
)


# OVERLAY


overlay = heatmap * 0.4 + original_np * 0.6

overlay = np.uint8(overlay)


# PLOT RESULTS


plt.figure(figsize=(15,5))

# ---------------------------------------------------------
# ORIGINAL IMAGE
# ---------------------------------------------------------

plt.subplot(1,2,1)

plt.imshow(original_np)

plt.title("Original Image")

plt.axis("off")
 
plt.subplot(1,2,2)

plt.imshow(overlay)

plt.title("Grad-CAM")

plt.axis("off")

plt.tight_layout()

plt.show()

print("\nGrad-CAM completed ")