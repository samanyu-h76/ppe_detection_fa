# app.py
import io, pathlib, importlib
import requests
import numpy as np
from PIL import Image
import cv2
import streamlit as st
import torch
from pathlib import Path

# ----- CONFIG -----
st.set_page_config(page_title="PPE Compliance Detection", layout="centered")

WEIGHTS_URL = "https://github.com/samanyu-h76/ppe_detection_fa/releases/download/v1.0/best.pt"
WEIGHTS_DIR = pathlib.Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)
WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"

# ----- DOWNLOAD WEIGHTS -----
def download_weights() -> str:
    if WEIGHTS_PATH.exists() and WEIGHTS_PATH.stat().st_size > 0:
        return str(WEIGHTS_PATH)
    st.info("Downloading model weights (first run)...")
    with requests.get(WEIGHTS_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        done = 0
        with open(WEIGHTS_PATH, "wb") as f:
            for chunk in r.iter_content(8192):
                if not chunk:
                    continue
                f.write(chunk)
                if total:
                    done += len(chunk)
                    st.progress(min(done / total, 1.0))
    return str(WEIGHTS_PATH)

# ----- IMAGE HELPERS -----
def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def draw_boxes(bgr: np.ndarray, boxes: np.ndarray, names: dict) -> np.ndarray:
    out = bgr.copy()
    for *xyxy, conf, cls in boxes:
        x1, y1, x2, y2 = map(int, xyxy)
        cls_id = int(cls)
        label = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 2)
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - bl), (x1 + tw, y1), (0, 165, 255), -1)
        cv2.putText(out, label, (x1, y1 - bl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out

# ----- MODEL (LOCAL YOLOv5, NO HUB) -----
@st.cache_resource(show_spinner=True)
def load_model_local(weights_path: str):
    y5 = importlib.import_module("yolov5")
    repo_dir = Path(y5.__file__).resolve().parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load(str(repo_dir), "custom", path=weights_path, source="local")
    model.to(device).eval()
    return model

# ----- UI -----
st.title("PPE Compliance Detection (YOLOv5)")
st.caption("Weights download once from GitHub Releases. YOLOv5 code loads locally.")

conf_thres = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
iou_thres  = st.sidebar.slider("IOU threshold (NMS)", 0.1, 0.9, 0.45, 0.05)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

with st.spinner("Preparing model..."):
    weights = download_weights()
    model = load_model_local(weights)
    model.conf = float(conf_thres)
    model.iou  = float(iou_thres)

if not uploaded:
    st.info("Upload an image to run detection.")
    st.stop()

img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
st.image(img, caption="Uploaded image", use_container_width=True)

with st.spinner("Running detection..."):
    results = model(img, size=640)
    boxes = results.xyxy[0].cpu().numpy()
    names = results.names

bgr = pil_to_bgr(img)
annot = draw_boxes(bgr, boxes, names)
st.image(bgr_to_pil(annot), caption="Detections", use_container_width=True)

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
            st.write(f"- {k}: {v}")

st.caption("First run downloads weights; later runs use cache.")
