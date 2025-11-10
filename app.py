# app.py
import os
import io
import time
import pathlib
from typing import Tuple

import requests
import numpy as np
from PIL import Image
import cv2
import streamlit as st
import torch

# ---------- CONFIG ----------
st.set_page_config(page_title="PPE Compliance Detection", layout="centered")

OWNER = "samanyu-h76"
REPO = "ppe_detection_fa"
TAG = "v1.0"
FILENAME = "best.pt"

WEIGHTS_URL = f"https://github.com/samanyu-h76/ppe_detection_fa/releases/download/v1.0/best.pt"
WEIGHTS_DIR = pathlib.Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)
WEIGHTS_PATH = WEIGHTS_DIR / FILENAME

# ---------- UTIL: DOWNLOAD WEIGHTS ----------
def download_weights() -> str:
    """Download weights from GitHub Releases if not present."""
    if WEIGHTS_PATH.exists() and WEIGHTS_PATH.stat().st_size > 0:
        return str(WEIGHTS_PATH)

    st.info("Downloading model weights (first run)…")
    with requests.get(WEIGHTS_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        chunk = 8192
        with open(WEIGHTS_PATH, "wb") as f:
            for data in r.iter_content(chunk_size=chunk):
                if data:
                    f.write(data)
                    downloaded += len(data)
                    if total:
                        st.progress(min(downloaded / total, 1.0))
    return str(WEIGHTS_PATH)

# ---------- UTIL: IMAGE HELPERS ----------
def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def draw_boxes(
    image_bgr: np.ndarray,
    boxes: np.ndarray,
    classes: list,
    names: dict,
) -> np.ndarray:
    img = image_bgr.copy()
    for *xyxy, conf, cls in boxes:
        x1, y1, x2, y2 = map(int, xyxy)
        cls_id = int(cls)
        label = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"
        # rectangle + label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - baseline), (x1 + tw, y1), (0, 165, 255), -1)
        cv2.putText(img, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img

# ---------- MODEL LOADING ----------
@st.cache_resource(show_spinner=False)
def load_model(local_weights_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # requires yolov5 dependency in requirements.txt (git+https://github.com/ultralytics/yolov5.git)
    model = torch.hub.load("ultralytics/yolov5", "custom", path=local_weights_path, source="local")
    model.to(device)
    return model

# ---------- APP UI ----------
st.title("🦺 PPE Compliance Detection (YOLOv5)")
st.caption("Downloads weights from GitHub Releases on first run, then caches the model.")

conf_thres = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
iou_thres = st.sidebar.slider("IOU threshold (NMS)", 0.1, 0.9, 0.45, 0.05)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Prepare model (download + load) once
with st.spinner("Preparing model…"):
    local_path = download_weights()
    model = load_model(local_path)
    model.conf = float(conf_thres)  # type: ignore
    model.iou = float(iou_thres)    # type: ignore

if uploaded is None:
    st.info("Upload an image to run detection.")
    st.stop()

# Read image
image_bytes = uploaded.read()
pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
st.image(pil_img, caption="Uploaded image", use_container_width=True)

# Run inference
with st.spinner("Running detection…"):
    results = model(pil_img, size=640)
    # boxes: [x1,y1,x2,y2,conf,cls]
    boxes = results.xyxy[0].cpu().numpy()
    names = results.names

# Draw + show
img_bgr = pil_to_cv2(pil_img)
annotated = draw_boxes(img_bgr, boxes, classes=None, names=names)
st.image(cv2_to_pil(annotated), caption="Detections", use_container_width=True)

# Small table
df = results.pandas().xyxy[0]
if len(df) == 0:
    st.warning("No objects detected.")
else:
    counts = df["name"].value_counts().to_dict()
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Detections")
        st.dataframe(df[["name", "confidence", "xmin", "ymin", "xmax", "ymax"]], use_container_width=True)
    with cols[1]:
        st.subheader("Counts")
        for k, v in counts.items():
            st.write(f"- **{k}**: {v}")

st.caption("Tip: first load might take a bit while weights download. Subsequent runs are cached.")
