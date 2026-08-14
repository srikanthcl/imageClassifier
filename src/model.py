"""
model.py — Model factory: builds the neural network architecture.

WHAT THIS FILE DOES (plain English):
  Instead of designing a neural network from scratch (which needs huge
  amounts of data and compute), we start from a model that's already been
  trained on 1.2 million general images (ImageNet) and just replace its
  final "decision layer" with a new one sized for OUR classes. This is
  called TRANSFER LEARNING — we transfer the general visual knowledge
  (edges, shapes, textures) the model already learned, and only teach it
  the new, specific task of telling OUR classes apart.
"""

# nn: PyTorch's neural network building blocks (layers, activation
# functions, loss functions).
import torch.nn as nn

# models: pre-built, pretrained architectures (ResNet, EfficientNet, etc.)
# that ship with torchvision, already trained on ImageNet.
from torchvision import models


def build_model(arch: str, num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    Builds and returns a model ready for training on our own classes.

    Args:
        arch: which architecture to use. Options:
              "resnet50"        — a strong general-purpose choice, good
                                   balance of accuracy and speed.
              "efficientnet_b0" — often slightly more accurate for the
                                   same compute budget, worth trying if
                                   resnet50 isn't accurate enough.
              "custom_cnn"      — a small network built from scratch, no
                                   pretrained weights. Only worth using
                                   for very simple tasks or when you
                                   can't use ImageNet-pretrained weights
                                   for licensing/domain reasons.
        num_classes: how many categories the model needs to choose between
                     (e.g. 2 for "defective / ok", 5 for 5 product types).
        freeze_backbone: if True, we LOCK all the pretrained layers so
                          they don't change during training — only the
                          new final layer learns. This is fast and works
                          well when you have a small dataset (a few
                          hundred to a few thousand images).
                          Set to False once you have more data (several
                          thousand+ images per class) to let the whole
                          network fine-tune — usually squeezes out a bit
                          more accuracy at the cost of slower training.

    Returns:
        A PyTorch nn.Module (the model), ready to be moved to a device
        (CPU/GPU) and trained.
    """

    if arch == "resnet50":
        # Loads ResNet50 with weights already trained on ImageNet.
        # "IMAGENET1K_V2" is a specific, improved set of pretrained
        # weights torchvision provides (better than the original V1
        # weights from the ResNet paper).
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        if freeze_backbone:
            # requires_grad = False means "don't compute gradients for
            # this parameter" — i.e. don't update it during training.
            # We do this for every existing parameter in the pretrained
            # model, effectively freezing all the learned feature
            # detectors (edges, textures, shapes) exactly as they are.
            for param in model.parameters():
                param.requires_grad = False

        # ResNet50's final layer (model.fc) is a "fully connected" layer
        # that originally outputs 1000 scores (one per ImageNet class).
        # We replace it with a NEW layer sized for OUR number of classes.
        # This new layer is NOT frozen — its requires_grad defaults to
        # True, so it's the one part of the network that actually learns.
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch == "efficientnet_b0":
        # Same transfer-learning pattern as ResNet50, but with the
        # EfficientNet-B0 architecture instead — generally a bit more
        # parameter-efficient (fewer weights) for similar accuracy.
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # EfficientNet's classifier is a small Sequential block; index
        # [1] is the actual Linear layer we need to resize for our
        # number of classes (index [0] is a Dropout layer we keep as-is).
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if arch == "custom_cnn":
        # A small Convolutional Neural Network built from scratch — no
        # pretrained weights, no transfer learning. Useful as a fallback
        # or baseline comparison, or for very simple/small tasks where a
        # huge pretrained model would be overkill.
        return nn.Sequential(
            # Conv2d(in_channels=3, out_channels=32, kernel_size=3):
            # slides a 3x3 filter across the image, producing 32 different
            # "feature maps" that each highlight different simple patterns
            # (edges, color blobs, etc). padding=1 keeps the output the
            # same width/height as the input.
            nn.Conv2d(3, 32, 3, padding=1),
            # BatchNorm2d: normalizes the outputs of the conv layer so
            # training is faster and more stable.
            nn.BatchNorm2d(32),
            # ReLU: activation function — turns negative values to 0,
            # keeps positive values unchanged. Adds non-linearity so the
            # network can learn complex patterns, not just straight lines.
            nn.ReLU(),
            # MaxPool2d(2): shrinks the image by half in each dimension,
            # keeping only the strongest signal in each 2x2 block. Reduces
            # computation and makes the network less sensitive to exact
            # pixel position.
            nn.MaxPool2d(2),

            # Second conv block: takes the 32 feature maps from above and
            # produces 64 more complex feature maps (combinations of
            # edges into shapes, etc).
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),

            # Third conv block: 64 -> 128 feature maps, even more abstract
            # patterns (parts of objects).
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),

            # AdaptiveAvgPool2d(1): collapses each of the 128 feature maps
            # down to a single average number, regardless of the original
            # image size. This makes the network work with any input
            # resolution.
            nn.AdaptiveAvgPool2d(1),
            # Flatten: reshapes the 128x1x1 output into a simple list of
            # 128 numbers, ready for the final decision layer.
            nn.Flatten(),
            # Final layer: turns the 128 numbers into num_classes scores
            # (one per class) — the highest score is the model's prediction.
            nn.Linear(128, num_classes),
        )

    # If someone passes an architecture name we don't recognise, fail
    # loudly and clearly rather than silently doing the wrong thing.
    raise ValueError(f"Unknown arch: {arch}. Choose resnet50, efficientnet_b0, or custom_cnn.")
