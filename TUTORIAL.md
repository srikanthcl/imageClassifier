# TUTORIAL — How This Folder Is Organized

*This file is for you (not the customer) — a quick map of what every file
does and why it exists, so you can navigate, extend, or explain any part
of this project on a call without re-reading everything from scratch.*

## Folder map

```
image-classifier-portfolio/
│
├── README.md                    ← Client-facing overview: what it is, quickstart
├── TUTORIAL.md                  ← This file — for you, not the client
├── PRODUCTION_REPORT.md         ← Client-facing: methodology + how to deploy
├── requirements.txt             ← All Python packages needed (pip install -r)
├── dashboard.py                 ← Run this for the interactive demo (Streamlit)
│
├── src/                         ← All the actual working code
│   ├── data.py                  ← Loads images, applies augmentation
│   ├── model.py                 ← Builds the neural network (ResNet/EfficientNet/custom)
│   ├── train.py                 ← Trains the model, logs metrics, saves checkpoints
│   ├── inference.py              ← Command-line prediction on new images
│   └── api.py                    ← Production web API (FastAPI) serving predictions
│
├── notebooks/
│   └── exploration.ipynb         ← Step-by-step walkthrough notebook (good for demos)
│
├── reports/
│   ├── report_template.md        ← Fill-in-the-blank client report format
│   ├── training_log.csv          ← Auto-generated when you run train.py
│   └── run_config.json           ← Auto-generated: exact settings used for a training run
│
├── sample_data/                  ← Put a small example dataset here (class_a/, class_b/…)
└── checkpoints/                  ← Auto-created; best_model.pt saved here after training
```

## How the pieces connect (the flow)

```
sample_data/  →  data.py  →  train.py  →  checkpoints/best_model.pt
                                               │
                            ┌──────────────────┼──────────────────┐
                            ▼                  ▼                  ▼
                     inference.py        dashboard.py           api.py
                     (CLI, one-off)     (interactive demo)   (production serving)
```

Every one of `inference.py`, `dashboard.py`, and `api.py` reads the SAME
checkpoint file and reuses the SAME `data.py` / `model.py` code — there's
no duplicated logic between "the version you demo" and "the version that
ships to production." This matters because it means what a client sees
in the dashboard is guaranteed to behave identically to what the API
returns in production — no surprises after handover.

## What to do when bidding / demoing this project

1. **Client call demo**: run `streamlit run dashboard.py` — let them
   upload their own image live. This is the "wow" moment; don't just
   show code.
2. **If they ask "how does the model actually work"**: open
   `notebooks/exploration.ipynb` and walk through the augmentation
   visualization (Section 2) — it's the most intuitive way to explain
   transfer learning to a non-technical client.
3. **If they ask about production/deployment**: point them to
   `PRODUCTION_REPORT.md` — it's written for them, not for you.
4. **To adapt this to a new client's actual dataset**: only `data_dir`
   (folder path) and possibly `--arch` need to change. Nothing else in
   `src/` needs editing — that's the point of building it generically.

## Rebranding checklist (per client / per bid)

- [ ] Update `README.md` title/description if the use case is specific
      (e.g. "Defect Detection Pipeline" instead of generic wording)
- [ ] Swap `sample_data/` for a small anonymized sample of the client's
      actual images (or leave as a placeholder structure if NDA-restricted)
- [ ] Fill in `reports/report_template.md` with real numbers once trained
- [ ] Remove this TUTORIAL.md before sending the folder to a client —
      it's for your reference only
