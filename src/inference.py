"""
inference.py — Run the trained model on new images from the command line.

WHAT THIS FILE DOES (plain English):
  Loads the saved model checkpoint (from training) and uses it to predict
  the class of one image, or every image in a folder. Prints the predicted
  class name and how confident the model is, for each image.

HOW TO RUN:
  Single image:
    python inference.py --checkpoint checkpoints/best_model.pt --image path/to/photo.jpg
  Whole folder:
    python inference.py --checkpoint checkpoints/best_model.pt --folder path/to/photos/
"""

import argparse
# glob: finds all files matching a pattern (e.g. every file in a folder).
import glob
import os

import torch
# PIL (Pillow): the library used to open and manipulate image files.
from PIL import Image

try:
  from src.data import build_transforms
  from src.model import build_model
except ModuleNotFoundError:
  from data import build_transforms
  from model import build_model


def load_model(checkpoint_path, device):
    """
    Loads a saved checkpoint file and rebuilds the exact model that
    produced it, with its trained weights restored.
    """
    # torch.load reads the checkpoint dictionary we saved in train.py —
    # it contains the weights AND the metadata needed to rebuild the
    # model correctly (architecture, class names, image size).
    ckpt = torch.load(checkpoint_path, map_location=device)

    # class_to_idx was saved as e.g. {"cats": 0, "dogs": 1}. We need the
    # REVERSE mapping at inference time: given the model outputs "0", we
    # need to know that means "cats".
    idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}

    # Rebuild the same architecture used during training (so the weights
    # actually fit), sized for the same number of classes.
    model = build_model(ckpt["arch"], num_classes=len(idx_to_class))

    # Load the trained weights into this freshly-built model structure.
    model.load_state_dict(ckpt["model_state_dict"])

    # Move to the target device and switch to evaluation mode (turns off
    # training-only behaviours like Dropout, for consistent predictions).
    model.to(device).eval()

    return model, idx_to_class, ckpt["image_size"]


def predict_image(model, image_path, transform, idx_to_class, device):
    """Runs one image through the model and returns (predicted_class, confidence)."""

    # Open the image file and force it to 3-channel RGB (some images are
    # grayscale or have a 4th transparency channel, which would break
    # the model's expected input shape).
    image = Image.open(image_path).convert("RGB")

    # Apply the same resize/normalize steps used during training's
    # validation pass (no random augmentation — we want a stable,
    # repeatable prediction).
    # unsqueeze(0) adds a "batch" dimension of size 1, because the model
    # always expects a batch of images, even if that batch contains just one.
    tensor = transform(image).unsqueeze(0).to(device)

    # no_grad() because we're not training — saves memory and time.
    with torch.no_grad():
        outputs = model(tensor)
        # softmax converts the model's raw scores into probabilities that
        # sum to 1.0 across all classes, e.g. [0.92, 0.05, 0.03].
        probs = torch.softmax(outputs, dim=1)[0]
        # The class with the highest probability is our prediction.
        pred_idx = probs.argmax().item()

    return idx_to_class[pred_idx], probs[pred_idx].item()


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained classifier.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file from training")
    parser.add_argument("--image", type=str, help="Path to a single image to classify")
    parser.add_argument("--folder", type=str, help="Path to a folder — every image inside will be classified")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx_to_class, image_size = load_model(args.checkpoint, device)
    # Build the deterministic (non-augmented) transform, matching the
    # image_size the model was actually trained with.
    transform = build_transforms(image_size, train=False)

    # If --image was given, classify just that one file. Otherwise, grab
    # every file inside --folder.
    paths = [args.image] if args.image else glob.glob(os.path.join(args.folder, "*"))

    for path in paths:
        label, confidence = predict_image(model, path, transform, idx_to_class, device)
        # :.2% formats a decimal like 0.923 as "92.30%"
        print(f"{os.path.basename(path)}: {label} ({confidence:.2%} confidence)")


if __name__ == "__main__":
    main()
