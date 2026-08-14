"""
data.py — Dataset loading and image augmentation pipeline.

WHAT THIS FILE DOES (plain English):
  It takes a folder of labelled images and turns it into something PyTorch
  can feed into a neural network in small batches, with random tweaks
  (crops, flips, rotations) applied to the TRAINING images only, so the
  model learns to recognise the general pattern instead of memorising
  each exact picture.

EXPECTED FOLDER STRUCTURE:
    data_dir/
        class_a/
            img1.jpg
            img2.jpg
        class_b/
            img1.jpg
            ...
  Each sub-folder name becomes a class label automatically — you don't
  need to write any label file yourself. This is called "ImageFolder"
  format and it's a PyTorch convention.
"""

# torch: the core deep learning library — handles tensors (multi-dimensional
# arrays) and the machinery for training neural networks.
import torch

# DataLoader: wraps a dataset and hands it out to the training loop in
# small batches (e.g. 32 images at a time) instead of all at once, which
# would use too much memory.
# random_split: splits one dataset into two (train + validation) randomly
# but reproducibly if we fix a seed.
from torch.utils.data import DataLoader, random_split

# datasets: pre-built dataset loaders from torchvision, including
# ImageFolder which auto-reads our class_a/class_b folder structure.
# transforms: a pipeline of image edits (resize, crop, flip, normalize)
# applied to every image before it's fed to the model.
from torchvision import datasets, transforms


# These two numbers are the average and spread of pixel values across the
# 1.2 million images in ImageNet — the dataset our pretrained model was
# originally trained on. We normalize OUR images to match that same scale,
# because the pretrained weights expect input in this specific range.
# (Without this, the pretrained model would perform poorly on our data.)
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # average brightness per R, G, B channel
IMAGENET_STD = [0.229, 0.224, 0.225]   # spread/variance per R, G, B channel


def build_transforms(image_size: int = 224, train: bool = True):
    """
    Builds the sequence of image-editing steps applied before an image
    enters the model.

    Args:
        image_size: the width/height (in pixels) every image gets resized
                    to. 224 is the standard size ResNet/EfficientNet expect.
        train: if True, apply RANDOM augmentation (for training data, so
               the model sees slightly different versions of each image
               every epoch and doesn't just memorise it).
               If False, apply only deterministic resizing (for
               validation/test data — we want consistent, repeatable
               results when measuring accuracy, not random variation).

    Returns:
        A transforms.Compose object — basically a list of steps that get
        applied to an image, in order, from top to bottom.
    """
    if train:
        # transforms.Compose chains multiple transformations together —
        # each image passes through every step below, in sequence.
        return transforms.Compose([
            # Randomly crops a region of the image covering 80%-100% of
            # its area, then resizes that crop to image_size x image_size.
            # This teaches the model to recognise objects even if they're
            # not perfectly centered or fully visible.
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),

            # 50% chance of flipping the image left-right. Useful because
            # most real-world objects look equally valid mirrored (a cat
            # facing left vs right is still a cat).
            transforms.RandomHorizontalFlip(),

            # Randomly rotates the image by up to 15 degrees in either
            # direction. Helps the model handle slightly tilted photos.
            transforms.RandomRotation(15),

            # Randomly adjusts brightness/contrast/saturation by up to 20%.
            # Makes the model robust to different lighting conditions
            # (e.g. photos taken indoors vs outdoors, different cameras).
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

            # Converts the image from a PIL Image (0-255 pixel values) into
            # a PyTorch tensor with values scaled to 0.0-1.0 — the format
            # the model actually expects as numeric input.
            transforms.ToTensor(),

            # Rescales the 0.0-1.0 pixel values to match the distribution
            # the pretrained ImageNet model expects (see comment above).
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    # Validation/test path: NO randomness. We want the same image to
    # always produce the same prediction when we're measuring how
    # accurate the model really is.
    return transforms.Compose([
        # Resize directly to the target size — no random cropping.
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
):
    """
    Loads the image folder, splits it into training and validation sets,
    and wraps each in a DataLoader ready for the training loop.

    Args:
        data_dir: path to the folder containing one sub-folder per class.
        image_size: pixel size every image gets resized to.
        batch_size: how many images are grouped together per training step.
        val_split: fraction of the data held back for validation
                   (0.2 = 20% held out, 80% used for training).
        seed: fixes the randomness of the train/val split so re-running
              this function gives you the EXACT same split every time —
              important for reproducible results.

    Returns:
        train_loader, val_loader: PyTorch DataLoaders, ready to iterate
                                   over in batches.
        class_to_idx: a dictionary mapping class folder names to numeric
                       labels, e.g. {"cats": 0, "dogs": 1}. We need this
                       later at inference time to turn a predicted number
                       back into a human-readable class name.
    """
    # ImageFolder scans data_dir, treats each sub-folder as a class, and
    # builds a dataset of (image, label) pairs automatically.
    # We apply the TRAINING transforms here first — we'll override the
    # validation subset's transform below.
    full_dataset = datasets.ImageFolder(data_dir, transform=build_transforms(image_size, train=True))

    # This dictionary lets us convert between folder names and numeric
    # label indices later, e.g. {"defective": 0, "ok": 1}.
    class_to_idx = full_dataset.class_to_idx

    # Compute how many images go into validation vs training based on
    # the val_split fraction (e.g. 20% of 1000 images = 200 for validation).
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    # A torch.Generator with a fixed seed makes the random split
    # reproducible — running this script twice gives the same split,
    # which matters for comparing experiments fairly.
    generator = torch.Generator().manual_seed(seed)

    # Randomly assigns each image to either the train or val subset.
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    # IMPORTANT: the validation subset currently still has the RANDOM
    # augmentation transform attached (inherited from full_dataset).
    # We overwrite it here with the deterministic version, because we
    # never want randomness when measuring validation accuracy.
    val_ds.dataset.transform = build_transforms(image_size, train=False)

    # Wrap each subset in a DataLoader:
    #   - shuffle=True for training: images are seen in a different order
    #     each epoch, which helps the model generalise better.
    #   - shuffle=False for validation: order doesn't matter, and keeping
    #     it fixed makes debugging easier.
    #   - num_workers=2: uses 2 background processes to load images in
    #     parallel while the GPU/CPU is busy training, so we're not
    #     waiting on disk I/O.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, class_to_idx
