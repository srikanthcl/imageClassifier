# Production Report — Image Classification Pipeline

## 1. What this delivers

A trained image classifier plus everything needed to run it in a real
application: a reviewable dashboard, a command-line tool, and a
production-ready web API — not just a notebook that only runs on one
machine.

## 2. Methodology summary

- **Approach**: Transfer learning from ImageNet-pretrained weights
  (ResNet50 or EfficientNet-B0), with the final classification layer
  retrained on your labelled images.
- **Why transfer learning**: for datasets under roughly 10,000 images,
  training a network from scratch rarely outperforms fine-tuning a
  pretrained one, and needs far more data and compute to get there.
- **Data augmentation**: random crop, flip, rotation, and color jitter
  applied only during training — reduces overfitting on small datasets
  and improves robustness to real-world photo variation (lighting,
  angle, camera differences).
- **Validation**: 80/20 train/validation split, stratified and seeded
  for reproducibility — every reported metric is measured on images the
  model never trained on.
- **Metrics tracked**: accuracy and macro F1-score (F1 avoids
  accuracy's blind spot when classes are imbalanced).

## 3. Demo dataset — what to download for `sample_data/`

For demos and portfolio purposes (before a real client dataset is
available), use a small, free, clearly-licensed public dataset rather
than scraping images yourself. Recommended options, easiest first:

| Dataset | Why it works well for a demo | Size | Link |
|---|---|---|---|
| **Kaggle Cats vs Dogs (subset)** | Instantly recognisable to anyone watching — no domain expertise needed to judge if predictions look right | Use ~200-300 images per class, not the full 25k | kaggle.com/c/dogs-vs-cats |
| **Intel Image Classification** | 6 clean classes (buildings, forest, glacier, mountain, sea, street) — shows the pipeline handles more than 2 classes | ~14k train images total, use a subset | kaggle.com/datasets/puneet6060/intel-image-classification |
| **Oxford-IIIT Pet Dataset** | 37 pet breeds — good for showing fine-grained classification (harder task) if a client's use case needs subtle distinctions | ~7,400 images | robots.ox.ac.uk/~vgg/data/pets |
| **PyTorch built-in (no download needed)** | Zero setup — good if you just want to sanity-check the code runs | CIFAR-10 (10 classes, tiny images) | Loadable directly via `torchvision.datasets.CIFAR10` |

**Recommended for your portfolio demo specifically**: Cats vs Dogs, ~250
images per class. It's the most universally recognisable to a
freelance-platform client skimming your portfolio — they can tell at a
glance whether the model is right, without needing domain knowledge.

### How to set it up

1. Download the dataset zip from Kaggle (free account required) or use
   `kagglehub`/`opendatasets` to pull it via code.
2. Reorganize into the `ImageFolder` structure this pipeline expects:
   ```
   sample_data/
       cats/
           cat.0.jpg
           cat.1.jpg
           ...
       dogs/
           dog.0.jpg
           dog.1.jpg
           ...
   ```
3. Keep the demo set small (200-300 images per class) — trains in a few
   minutes on CPU, which matters when demoing live on a call.

### Quick download via code (no manual Kaggle browsing)

```bash
pip install kagglehub
python -c "
import kagglehub
path = kagglehub.dataset_download('salader/dogs-vs-cats')
print('Downloaded to:', path)
"
```
Then copy/move a subset of images from the downloaded folder into
`sample_data/cats/` and `sample_data/dogs/` as shown above.

## 4. How to run each part

### A. Train the model
```bash
pip install -r requirements.txt
python src/train.py --data_dir path/to/your/images --arch resnet50 --epochs 15
```
Produces: `checkpoints/best_model.pt`, `reports/training_log.csv`,
`reports/run_config.json`.

### B. Review results (dashboard)
```bash
python -m streamlit run dashboard.py
```
Opens a browser page showing training accuracy/loss charts and a
drag-and-drop image uploader for live predictions. No coding required.

### C. Command-line predictions (for scripting/batch use)
```bash
python src/inference.py --checkpoint checkpoints/best_model.pt --folder path/to/new_images/
```

### D. Production API (for integrating into an app/website)
```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Any application can then POST an image to `http://your-server:8000/predict`
and receive a JSON response with the predicted class and confidence.

## 5. Deploying to production — options

| Option | Best for | Notes |
|---|---|---|
| **Single server (uvicorn/gunicorn)** | Low-moderate traffic, simplest setup | Run `api.py` behind a process manager (systemd, supervisor) so it restarts on crash/reboot |
| **Docker container** | Consistent deployment across environments | Package `requirements.txt` + `src/` + checkpoint into an image; deploy to any container host |
| **Cloud managed service** (AWS SageMaker, GCP Vertex AI, Azure ML) | Need auto-scaling, managed infra | More setup overhead but handles traffic spikes and monitoring out of the box |
| **Serverless (AWS Lambda + API Gateway)** | Sporadic/low-frequency requests | Cold-start latency for large models is a tradeoff to test |

For most small-to-mid client deployments, a single Docker container behind
a reverse proxy (nginx) on a modest cloud VM is the simplest path from
"works on my machine" to "works in production" — no orchestration platform
needed unless traffic genuinely requires it.

## 6. Minimal Dockerfile (starting point)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY checkpoints/ ./checkpoints/
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 7. Monitoring & maintenance recommendations

- **Track prediction confidence over time**: if average confidence drops
  after deployment, it often signals the real-world data has started to
  drift from the training data (new product variants, different camera,
  etc.) — a sign it's time to retrain.
- **Log a sample of low-confidence predictions** for periodic human
  review — cheap way to catch systematic errors early.
- **Retrain cadence**: for most use cases, revisiting the model every
  1–3 months (or after any noticeable process/environment change) is a
  reasonable default; adjust based on how fast the client's real-world
  images are actually changing.

## 8. Results (fill in per project)

| Metric | Value |
|---|---|
| Validation Accuracy | [X.XX%] |
| Validation F1 (macro) | [X.XX] |
| Inference latency (single image, CPU) | [~Xms] |
| Model size | [~XX MB] |

See `reports/report_template.md` for the full methodology write-up format
used per client delivery.
