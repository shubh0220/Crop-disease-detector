import streamlit as st
import torch
import torch.nn as nn

from torchvision import models, transforms, datasets
from PIL import Image
st.title("🌱 Crop Disease Detector")
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
def predict_image(image):
    image_tensor = test_transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        proba = torch.softmax(outputs,dim=1)
        top_proba , top_indices = torch.topk(proba,k=3)
        results = []
        for i in range(3):
            disease = class_names[top_indices[0][i].item()]
            confidence = (top_proba[0][i].item()*100)
            results.append((disease,confidence))
        return results

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        width="stretch"
    )

    if st.button("Predict"):

        image = Image.open(uploaded_file)
        results = predict_image(image)
        st.success( "Prediction Complete")
        st.subheader("Top 3 Predictions")

        for i, (disease, confidence) in enumerate(results):
            st.write(f"{i+1}. {disease} : {confidence:.4f}%")