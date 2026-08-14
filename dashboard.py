"""
dashboard.py — A point-and-click dashboard for the image classifier.

WHAT THIS FILE DOES (plain English):
  This is the file a CUSTOMER runs to actually see and use the project —
  no coding knowledge needed. It opens a web page (in their own browser,
  running locally) where they can:
    1. See how well the model performed during training (accuracy/loss
       charts, so they can judge quality at a glance).
    2. Upload their own image and get an instant prediction with a
       confidence score.

HOW TO RUN:
  streamlit run dashboard.py
  (This opens automatically in your default web browser, usually at
  http://localhost:8501)

WHY STREAMLIT:
  Streamlit turns a plain Python script into an interactive web app with
  almost no extra code — no separate HTML/JavaScript/frontend needed.
  Every Streamlit command below (st.title, st.file_uploader, etc.) adds
  one visual element to the page, top to bottom.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
from src.data import build_transforms
from src.model import build_model

# ---------------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------------
# st.set_page_config controls the browser tab title/icon and overall
# page layout. Must be the very first Streamlit command in the script.
st.set_page_config(page_title="Image Classifier Dashboard", page_icon="🖼️", layout="wide")

st.title("🖼️ Image Classification Dashboard")
st.markdown(
    "This dashboard lets you review how the model performed during "
    "training, and test it live by uploading your own image."
)

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
LOG_PATH = PROJECT_ROOT / "reports" / "training_log.csv"


# ---------------------------------------------------------------------
# CACHING: load the model only ONCE, not every time the page reruns
# (Streamlit reruns the whole script on every user interaction, so
# without caching we'd reload the model on every click, which is slow).
# ---------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """Loads the checkpoint once and keeps it cached across interactions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}
    model = build_model(ckpt["arch"], num_classes=len(idx_to_class))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    transform = build_transforms(ckpt["image_size"], train=False)
    return model, idx_to_class, transform, device, ckpt.get("val_accuracy")


# ---------------------------------------------------------------------
# SECTION 1: TRAINING PERFORMANCE OVERVIEW
# ---------------------------------------------------------------------
st.header("1. Training Performance")

if LOG_PATH.exists():
    # Read the CSV log written by train.py — one row per training epoch.
    log_df = pd.read_csv(LOG_PATH)

    # st.columns splits the page into side-by-side sections — here we
    # show 3 headline numbers next to each other instead of stacked.
    col1, col2, col3 = st.columns(3)
    # .metric() shows a big number with a label — good for headline stats.
    col1.metric("Best Validation Accuracy", f"{log_df['val_accuracy'].max():.2%}")
    col2.metric("Final Training Loss", f"{log_df['train_loss'].iloc[-1]:.4f}")
    col3.metric("Epochs Trained", len(log_df))

    # Two charts side by side: accuracy over time, and loss over time.
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Validation Accuracy per Epoch")
        # st.line_chart expects the x-axis as the DataFrame index — we
        # set epoch as the index so it plots correctly.
        st.line_chart(log_df.set_index("epoch")[["val_accuracy"]])
    with chart_col2:
        st.subheader("Loss per Epoch")
        st.line_chart(log_df.set_index("epoch")[["train_loss", "val_loss"]])

    with st.expander("View raw training log"):
        st.dataframe(log_df)
else:
    st.info("No training log found yet. Run `python src/train.py --data_dir <your_data>` first.")


# ---------------------------------------------------------------------
# SECTION 2: LIVE PREDICTION — upload an image, get a prediction
# ---------------------------------------------------------------------
st.header("2. Try the Model")

if not CHECKPOINT_PATH.exists():
    st.warning("No trained model found yet. Run training first to enable live predictions.")
else:
    model, idx_to_class, transform, device, val_acc = load_trained_model()

    if val_acc:
        st.caption(f"Model trained to {val_acc:.2%} validation accuracy.")

    # st.file_uploader creates a drag-and-drop / click-to-browse upload
    # box in the browser — this is the "just run it and get clarity"
    # interaction a non-technical customer needs.
    uploaded_file = st.file_uploader("Upload an image to classify", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        # Show the uploaded image and the result side by side.
        img_col, result_col = st.columns(2)
        with img_col:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with result_col:
            # Preprocess and run the model exactly as inference.py does.
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_idx = probs.argmax().item()

            predicted_class = idx_to_class[pred_idx]
            confidence = probs[pred_idx].item()

            st.subheader("Prediction")
            st.success(f"**{predicted_class}** ({confidence:.1%} confidence)")

            # Build a small table of every class's probability, sorted
            # highest first, and show it as a bar chart — gives the
            # customer a full picture, not just the top guess.
            prob_df = pd.DataFrame({
                "class": list(idx_to_class.values()),
                "probability": [probs[i].item() for i in range(len(idx_to_class))],
            }).sort_values("probability", ascending=False)

            st.bar_chart(prob_df.set_index("class"))


# ---------------------------------------------------------------------
# SECTION 3: HOW TO DEPLOY THIS
# ---------------------------------------------------------------------
with st.expander("ℹ️ How this connects to production"):
    st.markdown(
        """
        This dashboard is for **review and demoing** — it's not the
        production interface itself. In production, the same trained
        model (`checkpoints/best_model.pt`) is served by `src/api.py`
        (a FastAPI endpoint) which any real application can call over
        HTTP. See `PRODUCTION_REPORT.md` for full deployment details.
        """
    )
