# Production-Ready Image Classification Pipeline

A complete, reproducible pipeline for turning a labelled image folder into a deployable
CNN classifier — built with transfer learning (ResNet / EfficientNet), configurable
augmentation, tracked training metrics, and a ready-to-serve inference API.

## What this demonstrates

This project mirrors a real-world client brief: given a folder of labelled images,
deliver a trained model, an inference endpoint, and a methodology report — with zero
guesswork on the client's side about how to reproduce or extend the results.

## Features

- **Data pipeline** (`src/data.py`) — loads any `ImageFolder`-structured dataset,
  applies configurable augmentation (random flip, rotation, color jitter, normalization)
  so the network generalizes instead of memorizing.
- **Model factory** (`src/model.py`) — swap between ResNet50, EfficientNet-B0, or a
  custom small CNN via a single config flag. Transfer learning with frozen/unfrozen
  backbone options.
- **Training loop** (`src/train.py`) — full training script with:
  - train/val split, early stopping, LR scheduling
  - per-epoch accuracy/loss/F1 tracked and logged to `reports/training_log.csv`
  - checkpointing best model by validation accuracy
- **Inference** (`src/inference.py`) — loads a trained checkpoint and serves predictions
  either via CLI (single image / batch folder) or as a FastAPI endpoint (`src/api.py`).
- **Methodology report template** (`reports/report_template.md`) — the exact
  deliverable format a client expects: approach, architecture choice rationale,
  final metrics, and recommendations for improvement.

## Quickstart

```bash
pip install -r requirements.txt

# Train
python src/train.py --data_dir sample_data --arch resnet50 --epochs 15

# Run inference on a single image
python src/inference.py --checkpoint checkpoints/best_model.pt --image path/to/img.jpg

# Serve as an API
uvicorn src.api:app --reload
```

## Project structure

```
├── src/
│   ├── data.py          # dataset loading + augmentation
│   ├── model.py          # model factory (ResNet/EfficientNet/custom)
│   ├── train.py           # training loop with metric tracking
│   ├── inference.py       # CLI inference
│   └── api.py             # FastAPI serving endpoint
├── reports/
│   ├── training_log.csv   # per-epoch metrics (generated)
│   └── report_template.md # methodology report format
├── notebooks/
│   └── exploration.ipynb  # EDA + augmentation visualization
└── requirements.txt
```

## Why this approach

- **Transfer learning first**: starting from ImageNet-pretrained weights (ResNet50 /
  EfficientNet-B0) converges faster and needs far less labelled data than training
  from scratch — the right default for most client datasets under ~10k images.
- **Reproducibility**: every run logs its config, seed, and metrics so results can be
  regenerated exactly — a common gap in one-off client deliverables.
- **Deployment-ready from day one**: the same model checkpoint powers both the CLI
  script and the API, so "delivering a model" doesn't mean handing over a notebook
  that only runs on the author's machine.

## Stack

Python · PyTorch · torchvision · FastAPI · scikit-learn (metrics) · Pillow
