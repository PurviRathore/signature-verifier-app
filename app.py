import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import gdown
import os

# -------------------------------
# Download the model from Google Drive
# -------------------------------
MODEL_URL = "https://drive.google.com/uc?id=15uNd8NyJNMeP3c7k4MKNv-aiUK5Wi7CI"
MODEL_PATH = "siamese_similarity_model.pth"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# -------------------------------
# Define model
# -------------------------------
class SiameseRegressor(nn.Module):
    def __init__(self, base_model):
        super(SiameseRegressor, self).__init__()
        self.base = base_model
        self.base.fc = nn.Identity()

    def forward_once(self, x):
        return self.base(x)

    def forward(self, x1, x2):
        emb1 = self.forward_once(x1)
        emb2 = self.forward_once(x2)
        cos_sim = F.cosine_similarity(emb1, emb2)
        return cos_sim.unsqueeze(1)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    base = timm.create_model("xception", pretrained=True)
    base.conv1 = nn.Conv2d(1, base.conv1.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
    with torch.no_grad():
        base.conv1.weight[:, 0] = base.conv1.weight.mean(dim=1)

    model = SiameseRegressor(base)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# -------------------------------
# Image Preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_image(img):
    img = img.convert("L")  # Convert to grayscale
    img = transform(img).unsqueeze(0)
    return img

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Signature Verifier", layout="centered")
st.title("📝 Signature Verifier App")

img1 = st.file_uploader("Upload Signature 1", type=["png", "jpg", "jpeg"])
img2 = st.file_uploader("Upload Signature 2", type=["png", "jpg", "jpeg"])

if img1 and img2:
    image1 = Image.open(img1)
    image2 = Image.open(img2)

    st.image([image1, image2], caption=["Signature 1", "Signature 2"], width=250)

    with st.spinner("Comparing Signatures..."):
        input1 = preprocess_image(image1)
        input2 = preprocess_image(image2)

        with torch.no_grad():
            similarity = model(input1, input2).item()

        # Convert similarity to dissimilarity %
        dissimilarity = (1 - similarity) * 100
        dissimilarity = round(dissimilarity, 2)

        # Color-coded scale
        if dissimilarity < 20:
            color = "🟢"
            label = "Highly Similar - Likely Genuine"
        elif dissimilarity < 40:
            color = "🟢🟡"
            label = "Similar - Possibly Genuine"
        elif dissimilarity < 60:
            color = "🟡🟠"
            label = "Moderately Similar - Needs Review"
        elif dissimilarity < 80:
            color = "🟠🔴"
            label = "Likely Forged"
        else:
            color = "🔴"
            label = "Highly Dissimilar - Likely Forged"

        st.markdown(f"### Dissimilarity Score: **{dissimilarity}%** {color}")
        st.markdown(f"### Result: **{label}**")

