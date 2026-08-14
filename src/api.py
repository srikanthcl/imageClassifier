"""
api.py — Serves the trained model as a web API, so any application
(a website, a mobile app, another service) can send it an image over the
internet and get a prediction back — this is the "production deployment"
piece a client needs to actually USE the model day-to-day, rather than
running a script manually every time.

HOW TO RUN:
  uvicorn api:app --reload
  Then open http://127.0.0.1:8000/docs in a browser for an interactive
  test page (FastAPI builds this automatically).

HOW TO CALL IT (example using curl):
  curl -X POST -F "file=@photo.jpg" http://127.0.0.1:8000/predict
"""

# io: lets us treat raw bytes received over the network as if they were
# a file, so PIL can open them.
import io
from pathlib import Path

import torch
# FastAPI: the web framework that turns Python functions into HTTP
# endpoints. File/UploadFile: FastAPI's tools for accepting an uploaded
# file (our image) in a request.
from fastapi import FastAPI, File, UploadFile
from PIL import Image

try:
    # Supports package execution (e.g. `uvicorn src.api:app`).
    from .data import build_transforms
    from .model import build_model
except ImportError:
    # Fallback for script-style execution from inside `src/`.
    from data import build_transforms
    from model import build_model

# Creates the actual web application object. "title" shows up in the
# auto-generated API documentation page.
app = FastAPI(title="Image Classifier API")

# Where the trained model checkpoint lives — this must exist (i.e. you
# must have run train.py first) before starting this API.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A simple dictionary to hold the loaded model and related objects in
# memory, so we only load them ONCE at startup — not on every single
# request, which would be far too slow.
_state = {}


@app.on_event("startup")
def load_model():
    """
    Runs automatically once, when the API server first starts up —
    loads the model into memory and keeps it there for every future
    request to reuse.
    """
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}
    model = build_model(ckpt["arch"], num_classes=len(idx_to_class))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    _state["model"] = model
    _state["idx_to_class"] = idx_to_class
    _state["transform"] = build_transforms(ckpt["image_size"], train=False)


@app.get("/health")
def health():
    """
    A simple 'is this API alive and working' check — useful for
    monitoring tools / load balancers in production to confirm the
    service is up before sending it real traffic.
    """
    return {"status": "ok", "classes": list(_state.get("idx_to_class", {}).values())}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    The main endpoint: accepts an uploaded image file and returns the
    predicted class plus confidence scores for every class.

    file: UploadFile = File(...) tells FastAPI to expect a file upload
    named "file" in the incoming request — the "..." means it's required.
    """
    # Read the raw bytes of the uploaded file.
    image_bytes = await file.read()

    # io.BytesIO wraps those raw bytes so PIL can open them exactly as
    # if they were a file on disk.
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Apply the same preprocessing used during training/validation, then
    # add the batch dimension the model expects.
    tensor = _state["transform"](image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = _state["model"](tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()

    # Return a JSON response — FastAPI automatically converts this
    # Python dictionary into JSON for the client.
    return {
        "predicted_class": _state["idx_to_class"][pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
        # Include every class's probability, not just the winner — useful
        # for clients who want to show "92% cat, 5% dog, 3% other" style
        # results, or apply their own confidence threshold.
        "all_probabilities": {
            _state["idx_to_class"][i]: round(p.item(), 4) for i, p in enumerate(probs)
        },
    }
