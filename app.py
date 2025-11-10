import streamlit as st
import torch
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="PPE Detection System", page_icon="🦺", layout="wide")
st.title("🦺 PPE Compliance Detection System")
st.write("Upload an image to detect PPE items like helmets, masks, and vests.")

MODEL_PATH = "best.pt"            # keep at repo root for Streamlit Cloud
CONF_THRESHOLD = 0.25

@st.cache_resource(show_spinner=False)
def load_model():
    # Use GitHub hub repo so it works on Streamlit Cloud
    model = torch.hub.load(
        'ultralytics/yolov5',     # pulls repo on first run
        'custom',
        path=MODEL_PATH,
        force_reload=False
    )
    model.conf = CONF_THRESHOLD
    model.eval()
    return model

model = load_model()

def get_compliance(detections):
    has_hat = 'Hardhat' in detections
    has_mask = 'Mask' in detections
    has_vest = 'Safety Vest' in detections
    no_hat = 'NO-Hardhat' in detections
    no_mask = 'NO-Mask' in detections
    no_vest = 'NO-Safety Vest' in detections

    if has_hat and has_mask and has_vest and not (no_hat or no_mask or no_vest):
        return "🟢 Fully Compliant"
    elif no_hat and no_mask and no_vest:
        return "🔴 Non-Compliant"
    else:
        return "🟡 Partially Compliant"

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model(image)  # PIL works
    df = results.pandas().xyxy[0]
    detections = df['name'].tolist()

    st.subheader("Detection Results")
    # results.render() returns BGR numpy; convert to RGB before showing
    annotated_bgr = results.render()[0]
    annotated_rgb = annotated_bgr[:, :, ::-1]
    st.image(annotated_rgb, caption="Detected PPE Items", use_column_width=True)

    if len(detections):
        st.write("**Detected Objects:**", ", ".join(sorted(set(detections))))
    else:
        st.write("No objects detected.")

    st.markdown(f"### Compliance Status: {get_compliance(detections)}")

    os.makedirs("detections", exist_ok=True)
    Image.fromarray(annotated_rgb).save("detections/last_result.jpg")
