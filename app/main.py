from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "bird_resnet18_model.pth"
LABELS_PATH = APP_DIR / "bird_labels.csv"
CLASS_COUNT = 525

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@st.cache_resource
def load_label_mapping() -> list[str]:
    labels_df = pd.read_csv(LABELS_PATH).sort_values("model_index")
    labels = labels_df["label"].tolist()

    if len(labels) != CLASS_COUNT:
        raise ValueError(f"Expected {CLASS_COUNT} labels, found {len(labels)}.")

    return labels


def build_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, CLASS_COUNT)
    return model


@st.cache_resource
def load_model() -> nn.Module:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}.")

    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    state_dict = checkpoint

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint

    model = build_model()
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_bird(image: Image.Image) -> tuple[str, float]:
    model = load_model()
    labels = load_label_mapping()

    image_tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    label = labels[predicted_idx.item()]
    return label, confidence.item()


def clear_upload() -> None:
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1


def render_header() -> None:
    title_col, action_col = st.columns([6, 1])

    with title_col:
        st.markdown("### Bird Species Identifier")

    with action_col:
        st.button("New upload", on_click=clear_upload, use_container_width=True)


def render_styles() -> None:
    st.markdown(
        """
        <style>
            header[data-testid="stHeader"],
            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"],
            #MainMenu,
            footer {
                display: none;
            }

            .stApp {
                background: #f5f8fc;
                color: #16324f;
            }

            .block-container {
                max-width: 1100px;
                padding-top: 1.25rem;
                padding-bottom: 2rem;
            }

            [data-testid="stFileUploader"] {
                background: transparent;
                border: 0;
                border-radius: 0;
                padding: 0;
            }

            [data-testid="stFileUploaderDropzone"] {
                background: #ffffff;
                border: 1px dashed #4a84c4;
                border-radius: 0;
                padding: 2rem 1rem;
            }

            [data-testid="stImage"] {
                width: 100%;
            }

            [data-testid="stImage"] img {
                width: 100% !important;
                height: auto !important;
                display: block;
                object-fit: contain;
                border-radius: 0;
            }

            .stButton > button {
                border-radius: 0;
                border: 1px solid #4a84c4;
                background: #4a84c4;
                color: #ffffff;
                box-shadow: none;
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                border-radius: 0;
                border: 1px solid #d7e1ee;
                background: #ffffff;
            }

            .label {
                font-size: 0.85rem;
                color: #54708f;
                margin-bottom: 0.5rem;
            }

            .empty-state {
                color: #54708f;
                margin: 0;
            }

            .prediction {
                font-size: 1.35rem;
                color: #16324f;
                margin: 0.25rem 0 1rem;
            }

            .confidence {
                font-size: 1rem;
                color: #2f5f92;
                margin: 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Bird Species Identifier", layout="wide")

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    render_styles()
    render_header()

    left_col, right_col = st.columns(2, gap="large")
    image: Image.Image | None = None
    uploaded_file = None

    with left_col:
        st.markdown('<p class="label">Upload image</p>', unsafe_allow_html=True)
        with st.container(border=True):
            uploaded_file = st.file_uploader(
                "Drop a JPG or PNG here",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
                key=f"uploader_{st.session_state['uploader_key']}",
            )

            if uploaded_file is not None:
                image_bytes = uploaded_file.getvalue()
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                st.image(image, use_container_width=True)
            else:
                st.markdown(
                    '<p class="empty-state">Drag and drop a bird image here, or click to browse.</p>',
                    unsafe_allow_html=True,
                )

    with right_col:
        st.markdown('<p class="label">Result</p>', unsafe_allow_html=True)
        with st.container(border=True):
            if uploaded_file is None or image is None:
                st.markdown(
                    '<p class="empty-state">Prediction output will appear here after you upload an image.</p>',
                    unsafe_allow_html=True,
                )
            else:
                with st.spinner("Processing image..."):
                    try:
                        predicted_label, confidence = predict_bird(image)
                        st.markdown(f'<p class="prediction">{predicted_label}</p>', unsafe_allow_html=True)
                        st.markdown(
                            f'<p class="confidence">Confidence: {confidence * 100:.2f}%</p>',
                            unsafe_allow_html=True,
                        )
                    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
                        st.error(f"Unable to run inference: {error}")


if __name__ == "__main__":
    main()
