import os
import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import streamlit as st
import timm

# === Download model from Google Drive (only if not present) ===
MODEL_PATH = "siamese_similarity_model.pth"
if not os.path.exists(MODEL_PATH):
    file_id = "15uNd8NyJNMeP3c7k4MKNv-aiUK5Wi7CI"  # 🔁 Replace with your Google Drive file ID
    gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)

# === Siamese Model Definition (same as training) ===
class SiameseRegressor(nn.Module):
    def __init__(self, base_model):
        super(SiameseRegressor, self).__init__()
        self.base = base_model
        self.base.fc = nn.Identity()  # No classifier head in training

    def forward_once(self, x):
        return self.base(x)

    def forward(self, x1, x2):
        emb1 = self.forward_once(x1)
        emb2 = self.forward_once(x2)
        cos_sim = F.cosine_similarity(emb1, emb2)
        return cos_sim.unsqueeze(1)

# === Load Model ===
@st.cache_resource
def load_model():
    base = timm.create_model("xception", pretrained=False)
    base.conv1 = nn.Conv2d(1, base.conv1.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
    with torch.no_grad():
        base.conv1.weight[:, 0] = base.conv1.weight.mean(dim=1)
    model = SiameseRegressor(base)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

# === Image Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess(img):
    img = Image.open(img).convert("L")
    return transform(img).unsqueeze(0)  # Add batch dimension

# === Fraud Level Comment and Color Bar ===
def interpret_score(score):
    if score >= 0.85:
        return "✅ Genuine Signature", "green"
    elif score >= 0.65:
        return "🟡 Likely Genuine", "yellow"
    elif score >= 0.45:
        return "🟠 Possibly Forged", "orange"
    else:
        return "❌ Forged Signature", "red"

# === Streamlit App ===
st.set_page_config(page_title="Signature Similarity Verifier", layout="centered")
st.title("✍️ Signature Verification App")
st.markdown("Upload two signature images to compare their authenticity.")

col1, col2 = st.columns(2)

with col1:
    img1 = st.file_uploader("Upload Signature 1", type=["png", "jpg", "jpeg"], key="img1")
with col2:
    img2 = st.file_uploader("Upload Signature 2", type=["png", "jpg", "jpeg"], key="img2")

if img1 and img2:
    with st.spinner("Comparing signatures..."):
        model = load_model()
        t1 = preprocess(img1)
        t2 = preprocess(img2)

        with torch.no_grad():
            score = model(t1, t2).item()

        label, color = interpret_score(score)

        st.markdown(f"### Similarity Score: `{score:.2f}`")
        st.markdown(f"### Prediction: **{label}**")
        
        st.progress(score)  # Visual scale bar
        
        # Optional custom color scale
        st.markdown(
            f"<div style='background:linear-gradient(to right, green, yellow, orange, red); height:20px; border-radius:10px;'></div>",
            unsafe_allow_html=True
        )
        st.caption("Green = Genuine, Red = Forged")

