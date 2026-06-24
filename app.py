import streamlit as st
import torch
import torch.nn as nn

from torchvision import models, transforms, datasets
from PIL import Image
st.title("🌱 Crop Disease Detector")
st.markdown("""Detect crop diseases using a deep learning model trained on
20,638 crop leaf images across 15 disease categories.""")
st.sidebar.header("Model Information")

st.sidebar.write("Model : EfficientNet-B0")
st.sidebar.write("Accuracy : 99.39%")
st.sidebar.write("Classes : 15")
uploaded_file = st.file_uploader(
    "Upload a Leaf Image",
    type=["jpg","jpeg","png"]
)
dataset = datasets.ImageFolder("data/raw")
class_names = dataset.classes

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model = models.efficientnet_b0(weights = None)

model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280,15)
)

model.load_state_dict(
    torch.load(
        "models/efficientnet_b0_99_39.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
from utils import (
    predict_image,
    format_name
)
if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        width="stretch"
    )

    if st.button("Predict"):

        image = Image.open(uploaded_file)
        results = predict_image(image,model,device,test_transform,class_names)
        st.success( f"Most Likely Disease : {format_name(results[0][0])} ")
        st.subheader("Top 3 Predictions")

        for i, (disease, confidence) in enumerate(results):
            st.write(f"### {i+1}. {format_name(disease)}")
            st.progress(min(confidence / 100,1.0))
            st.write(f"Confidence: {confidence:.4f}")