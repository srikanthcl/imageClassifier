# Image Classification — Methodology Report (Template)

*This is the report format delivered to clients alongside the trained model and
inference endpoint. Fill in the bracketed sections per project.*

## 1. Objective
[What the model classifies, and why it matters for the client's use case.]

## 2. Dataset
- Total images: [N]
- Classes: [list]
- Class balance: [balanced / imbalanced — note any classes under-represented]
- Train/validation split: 80/20, stratified by class, seed=42 for reproducibility

## 3. Approach
- **Architecture**: [ResNet50 / EfficientNet-B0 / custom CNN] via transfer learning
  from ImageNet weights
- **Why this architecture**: [e.g. "ResNet50 chosen for balance of accuracy and
  inference speed given dataset size of ~N images; EfficientNet-B0 considered
  but offered no meaningful accuracy gain at 2x training time."]
- **Augmentation**: random crop, horizontal flip, rotation (±15°), color jitter —
  applied to reduce overfitting given the dataset size
- **Training**: Adam optimizer, ReduceLROnPlateau scheduler, early stopping on
  validation accuracy (patience=4)

## 4. Results
| Metric | Value |
|---|---|
| Validation Accuracy | [X.XX%] |
| Validation F1 (macro) | [X.XX] |
| Training time | [X min on Y hardware] |

See `training_log.csv` for full per-epoch metrics.

## 5. Deliverables
- `checkpoints/best_model.pt` — trained model weights + metadata
- `src/inference.py` — CLI inference script
- `src/api.py` — FastAPI endpoint (`POST /predict`)
- This report

## 6. Recommendations for future improvement
- [e.g. "Accuracy is currently limited by class imbalance in [class X] —
  collecting 200+ more labelled examples would likely improve macro F1 by
  several points."]
- [e.g. "Fine-tuning the full backbone (not just the classifier head) could
  yield further gains once more data/compute is available."]
- [Any data quality issues noticed during EDA.]
