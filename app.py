import os
import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import streamlit as st
import timm

# === Download model from Google Drive ===
MODEL_PATH = "siamese_similarity_model.pth"
if not os.path.exists(MODEL_PATH):
    file_id = "YOUR_FILE_ID_HERE"  # 🔁 Replace with actual file ID
    gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)

# === Model class (exact same as training) ===
class SiameseRegressor(nn.Module):
    def __init__(self, base_model):
        super(SiameseRegressor, self).__init__()
        self.base = base_model
        self.base.fc = nn.Identity()
        self.similarity_head = nn.Sequential(
            nn.Linear(self.base.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        return self.base(x)

    def forward(self, x1, x2):
        emb1 = self.forward_once(x1)
        emb2 = self.forward_once(x2)
        diff = torch.abs(emb1 - emb2)
        return self.similarity_head(diff)

# === Load model ===
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

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess(img):
    img = Image.open(img).convert("L")
    return transform(img).unsqueeze(0)

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
st.set_page_config(page_title="Signature Verifier", layout="centered")
st.title("✍️ Signature Verification App")
st.markdown("Upload two signature images to verify if they belong to the same person.")

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
        st.progress(score)
        st.markdown(
            f"<div style='background:linear-gradient(to right, green, yellow, orange, red); height:20px; border-radius:10px;'></div>",
            unsafe_allow_html=True
        )
        st.caption("Green = Genuine, Red = Forged")

