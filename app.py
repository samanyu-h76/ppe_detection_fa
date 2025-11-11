# app.py
import os, io, pathlib
from typing import Tuple

import requests
import numpy as np
from PIL import Image
import cv2
import streamlit as st
import torch
import importlib
from pathlib import Path

# ---------- CONFIG ----------
st.set_page_config(page_title="PPE Compliance Detection", layout="centered")

# your release asset (keep this exactly as your GitHub Release URL)
WEIGHTS_URL = "https://github.com/samanyu-h76/ppe_detection_fa/releases/download/v1.0/best.pt"
WEIGHTS_DIR = pathlib.Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)
WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"

# ---------- UTIL: DOWNLOAD WEIGHTS ----------
def download_weights() -> str:
    """Download weights from GitHub Releases if not present (cached on disk)."""
    if WEIGHTS_PATH.exists() and WEIGHTS_PATH.stat().st_size > 0:
        return str(WEIGHTS_PATH)

    st.info("Downloading model weights (first run)…")
    with requests.get(WEIGHTS_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        bytes_dl = 0
        with open(WEIGHTS_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                if total:
                    bytes_dl += len(chunk)
                    st.progress(min(bytes_dl / total, 1.0))
    return str(WEIGHTS_PATH)

# ---------- IMAGE HELPERS ----------
def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def draw_boxes(image_bgr: np.ndarray, boxes: np.ndarray, names: dict) -> np.ndarray:
    img = image_bgr.copy()
    for *xyxy, conf, cls in boxes:
        x1, y1, x2, y2 = map(int, xyxy)
        cls_id = int(cls)
        label = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"
        # rectangle + label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - baseline), (x1 + tw, y1), (0, 165, 255), -1)
        cv2.putText(img, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return img

# ---------- MODEL LOADING (LOCAL, NO INTERNET) ----------
@st.cache_resource(show_spinner=True)
def load_model_local(weights_path: str):
    """
    Load YOLOv5 from the installed package in site-packages,
    then load your custom weights.
    """
    # find installed yolov5 package path
    y5 = importlib.import_module("yolov5")
    repo_dir = Path(y5.__file__).resolve().parent  # .../site-packages/yolov5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load(
        str(repo_dir),          # local repo path
        "custom",
        path=weights_path,
        source="local"          # <-- key bit: no remote hub calls
    )
    model.to(device).eval()
    return model

# ---------- UI ----------
st.title("🦺 PPE Compliance Detection (YOLOv5)")
st.caption("Weights download once from GitHub Releases. YOLOv5 code is loaded locally (no runtime hub calls).")

conf_thres = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
iou_thres  = st.sidebar.slider("IOU threshold (NMS)", 0.1, 0.9, 0.45, 0.05)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# prepare model once
with st.spinner("Preparing model…"):
    local_weights = download_weights()
    model = load_model_local(local_weights)
    model.conf = float(conf_thres)
    model.iou  = float(iou_thres)

if not uploaded:
    st.info("Upload an image to run detection.")
    st.stop()

# read image
img_bytes = uploaded.read()
pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
st.image(pil_img, caption="Uploaded image", use_container_width=True)

# inference
with st.spinner("Running detection…"):
    results = model(pil_img, size=640)
    boxes = results.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,cls]
    names = results.names

# draw + show
bgr = pil_to_cv2(pil_img)
annotated = draw_boxes(bgr, boxes, names)
st.image(cv2_to_pil(annotated), caption="Detections", use_container_width=True)

# table
df = results.pandas().xyxy[0]
if len(df) == 0:
    st.warning("No objects detected.")
else:
    counts = df["name"].value_counts().to_dict()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Detections")
        st.dataframe(df[["name", "confidence", "xmin", "ymin", "xmax", "ymax"]], use_container_width=True)
    with c2:
        st.subheader("Counts")
        for k, v in counts.items():
            st.write(f"- **{k}**: {v}")

st.caption("Tip: first load might take a bit while weights
