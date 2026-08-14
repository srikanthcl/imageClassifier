"""
train.py — Trains the classifier and tracks every result for reproducibility.

WHAT THIS FILE DOES (plain English):
  Runs the model through the training images repeatedly (each full pass is
  called an "epoch"), each time slightly adjusting the model's internal
  numbers (weights) so its predictions get closer to the correct answer.
  After every epoch, it checks performance on the validation set (images
  the model never trains on) and saves the best-performing version.

HOW TO RUN:
  python train.py --data_dir path/to/images --arch resnet50 --epochs 15
"""

# argparse: lets us run this script with command-line options like
# --data_dir and --epochs instead of hardcoding values.
import argparse
# csv: used to write the per-epoch metrics log as a spreadsheet-readable file.
import csv
# json: used to save the run configuration in a structured, readable format.
import json
# os: used for creating folders (checkpoints/, reports/) if they don't exist.
import os
# time: used to measure how long each training epoch takes.
import time

import torch
import torch.nn as nn
# f1_score: a metric that balances precision and recall — more informative
# than plain accuracy when classes are imbalanced (e.g. 90 images of one
# class, 10 of another).
from sklearn.metrics import f1_score

try:
    from src.data import get_dataloaders
    from src.model import build_model
except ModuleNotFoundError:
    from data import get_dataloaders
    from model import build_model


def evaluate(model, loader, device, criterion):
    """
    Runs the model on a dataset WITHOUT updating its weights — used to
    check performance on validation data after each training epoch.
    """
    # .eval() switches off training-only behaviours like Dropout, so
    # predictions are consistent and deterministic.
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    # torch.no_grad() tells PyTorch not to track operations for
    # backpropagation — we're not training here, just measuring, so this
    # saves memory and runs faster.
    with torch.no_grad():
        for images, labels in loader:
            # Move this batch of images/labels to the same device (CPU or
            # GPU) the model lives on — required for the math to work.
            images, labels = images.to(device), labels.to(device)

            # Forward pass: run the images through the model to get
            # predicted scores for each class.
            outputs = model(images)

            # Compare predicted scores to the true labels to get a loss
            # value (lower = better predictions).
            loss = criterion(outputs, labels)
            # Accumulate the total loss, weighted by batch size, so we can
            # compute a correct average at the end (batches can vary in size).
            total_loss += loss.item() * images.size(0)

            # argmax picks the class with the highest predicted score —
            # that's the model's actual prediction for each image.
            preds = outputs.argmax(dim=1)
            # Count how many predictions matched the true label.
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Keep every prediction/label so we can compute F1 score
            # across the WHOLE validation set at the end (F1 needs the
            # full set, not a running average per batch).
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total
    accuracy = correct / total
    # average="macro" computes F1 per class then averages them equally —
    # this stops a large class from hiding poor performance on a small
    # class, which a simple accuracy number can do.
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, accuracy, f1


def train(args):
    """Main training loop, driven by the command-line arguments in `args`."""

    # Use a GPU if one's available (much faster), otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load our train/validation DataLoaders and the class name mapping.
    train_loader, val_loader, class_to_idx = get_dataloaders(
        args.data_dir, args.image_size, args.batch_size, args.val_split, args.seed
    )
    num_classes = len(class_to_idx)

    # Build the model and move it to the chosen device.
    model = build_model(args.arch, num_classes, freeze_backbone=args.freeze_backbone).to(device)

    # CrossEntropyLoss: the standard loss function for multi-class
    # classification — measures how far the predicted probability
    # distribution is from the true label.
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer: the algorithm that actually updates the model's
    # weights based on the loss. filter(...) ensures we only pass in
    # parameters that are trainable (requires_grad=True) — i.e. we skip
    # the frozen backbone if freeze_backbone=True.
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # ReduceLROnPlateau: automatically shrinks the learning rate if
    # validation accuracy stops improving for `patience` epochs — helps
    # the model fine-tune more precisely once it's close to its best
    # performance instead of overshooting.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    # Make sure our output folders exist before we try to save into them.
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Set up the CSV log file with a header row — every epoch's metrics
    # get appended to this file as training progresses.
    log_path = "reports/training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_f1", "lr", "seconds"])

    # Track the best validation accuracy seen so far, and how many epochs
    # in a row we've gone WITHOUT improving (for early stopping).
    best_acc, patience_counter = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        # .train() switches ON training-only behaviours (Dropout etc).
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Reset gradients from the previous batch — PyTorch
            # accumulates gradients by default, so we must clear them
            # before each new batch's backward pass.
            optimizer.zero_grad()

            # Forward pass: get predictions for this batch.
            outputs = model(images)
            # Compute how wrong those predictions were.
            loss = criterion(outputs, labels)
            # Backward pass: compute how much each trainable weight
            # contributed to the error (the gradient).
            loss.backward()
            # Update the trainable weights using those gradients.
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # Average training loss across the whole epoch.
        train_loss = running_loss / len(train_loader.dataset)

        # Check performance on the validation set (images never trained on).
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, criterion)

        # Let the scheduler know this epoch's validation accuracy, so it
        # can decide whether to reduce the learning rate.
        scheduler.step(val_acc)

        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]

        # Print a human-readable progress line to the terminal.
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} ({elapsed:.1f}s)")

        # Append this epoch's numbers as a new row in the CSV log.
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_acc, val_f1, current_lr, elapsed])

        # If this epoch beat our previous best validation accuracy, save
        # a checkpoint — the model file we'll actually use later.
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0  # reset the "no improvement" counter
            torch.save({
                "model_state_dict": model.state_dict(),  # the learned weights
                "arch": args.arch,                        # so inference knows which architecture to rebuild
                "class_to_idx": class_to_idx,              # so inference can decode predictions back to class names
                "image_size": args.image_size,             # so inference resizes images the same way
                "val_accuracy": val_acc,
            }, "checkpoints/best_model.pt")
        else:
            # No improvement this epoch — count it towards early stopping.
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.early_stop_patience} epochs).")
                break

    # Save the full run configuration (all command-line args used) plus
    # the final best accuracy — this makes every training run fully
    # reproducible and auditable later.
    with open("reports/run_config.json", "w") as f:
        json.dump(vars(args) | {"best_val_accuracy": best_acc, "class_to_idx": class_to_idx}, f, indent=2)

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    print("Checkpoint saved to checkpoints/best_model.pt")
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    # This block only runs when the script is executed directly (not when
    # imported by another file) — it defines all the command-line options.
    parser = argparse.ArgumentParser(description="Train an image classifier via transfer learning.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to folder with one sub-folder per class")
    parser.add_argument("--arch", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0", "custom_cnn"])
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Freeze pretrained layers, train only the final layer")
    parser.add_argument("--image_size", type=int, default=224, help="Pixel size images are resized to")
    parser.add_argument("--batch_size", type=int, default=32, help="Images per training step")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data held out for validation")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum number of full passes over the training data")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate — how big each weight update step is")
    parser.add_argument("--early_stop_patience", type=int, default=4, help="Stop if val accuracy doesn't improve for this many epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible train/val split")
    args = parser.parse_args()
    train(args)
