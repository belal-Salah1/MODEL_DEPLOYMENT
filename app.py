from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras
import preprocess



MODEL_PATH = Path(__file__).parent / "models" / "wildfire_model.keras"
IMG_SIZE = (224, 224)
SMOKE_INDEX = 2
SMOKE_THRESHOLD = 0.5

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    preprocess.preprocess_image(image)
    return np.expand_dims(arr, axis=0)


def main():
    st.set_page_config(page_title="Smoke Detector", layout="centered")
    st.title("Smoke Detector")
    st.write("Upload an image and the model will tell you if it contains smoke.")

    uploaded = st.file_uploader(
        "Drop an image here, or click to browse",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded is None:
        st.info("Waiting for an image...")
        return

    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Running model..."):
        model = load_model()
        x = preprocess(image)
        probs = model.predict(x, verbose=0)[0]

    smoke_prob = float(probs[SMOKE_INDEX])

    st.metric(label="Smoke probability", value=f"{smoke_prob:.1%}")
    st.progress(min(max(smoke_prob, 0.0), 1.0))

    if smoke_prob >= SMOKE_THRESHOLD:
        st.error("Smoke detected")
    else:
        st.success("No smoke")


if __name__ == "__main__":
    main()
