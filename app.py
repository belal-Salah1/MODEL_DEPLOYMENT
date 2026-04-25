from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras

MODEL_PATH = Path(__file__).parent / "models" / "wildfire_model.keras"
IMG_SIZE = (224, 224)

# TODO: replace with the real class names in the order the model was trained on.
CLASS_NAMES = ["class_0", "class_1", "class_2"]


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def main():
    st.set_page_config(page_title="Wildfire Classifier", layout="centered")
    st.title("Wildfire Image Classifier")
    st.write("Upload an image and the model will predict the class.")

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

    top_idx = int(np.argmax(probs))
    st.success(
        f"Prediction: **{CLASS_NAMES[top_idx]}**  "
        f"(confidence {probs[top_idx]:.1%})"
    )

    st.subheader("All class probabilities")
    chart_df = pd.DataFrame({"probability": probs}, index=CLASS_NAMES)
    st.bar_chart(chart_df)


if __name__ == "__main__":
    main()
