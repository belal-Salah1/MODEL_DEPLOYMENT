# Wildfire Classifier — Streamlit Deployment

A Streamlit web app that serves a Keras image classifier for wildfire images.
Drop in a photo, and the model returns the predicted class with confidence.

## Model

- **Architecture:** MobileNetV3Small (transfer learning) → GlobalAveragePool → BatchNorm → Dense → Dropout → Dense
- **Input:** 224×224 RGB images
- **Output:** 3 classes with sigmoid activation (treated as independent probabilities)
- **Saved with:** Keras 3.11.2

The trained model lives at `models/wildfire_model.keras`.

## Project structure

```
modeldeploy/
├── app.py              # Streamlit UI + inference
├── requirements.txt    # Python dependencies
├── models/
│   └── wildfire_model.keras
├── .gitignore
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app opens at <http://localhost:8501>. Upload a JPG or PNG and the model will:

1. Resize the image to 224×224 and convert to RGB
2. Run inference with the cached model
3. Show the top class with confidence and a bar chart of all 3 probabilities

## Configuration

Before running for the first time, edit the `CLASS_NAMES` constant near the top
of `app.py` to match the labels used during training (in the same order):

```python
CLASS_NAMES = ["class_0", "class_1", "class_2"]
```

The class names are not stored in the `.keras` file, so this must be set
manually from the training notebook.

## Notes

- **Preprocessing:** MobileNetV3Small includes a `Rescaling` layer internally,
  so the app feeds raw `[0, 255]` float pixels — no manual normalization.
- **Model caching:** `@st.cache_resource` keeps the model loaded across
  uploads, so only the first prediction pays the load cost.
- **Sigmoid vs softmax:** Outputs are independent. The bar chart shows all
  three so you can see when multiple classes score high.
