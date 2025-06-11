import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

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

import os
import gdown  # Add this import

@st.cache_resource
def load_model():
    if not os.path.exists("siamese_similarity_model.pth"):
        with st.spinner("📦 Downloading model file..."):
            url = "https://drive.google.com/uc?id=15uNd8NyJNMeP3c7k4MKNv-aiUK5Wi7CI"
            gdown.download(url, "siamese_similarity_model.pth", quiet=False)
    base = timm.create_model("xception", pretrained=True)
    base.conv1 = nn.Conv2d(1, base.conv1.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
    with torch.no_grad():
        base.conv1.weight[:, 0] = base.conv1.weight.mean(dim=1)
    model = SiameseRegressor(base)
    model.load_state_dict(torch.load("siamese_similarity_model.pth", map_location=device))
    model.eval()
    return model.to(device)

st.title("Signature Similarity Verifier")
st.markdown("Upload two signature images to compare their similarity.")

img1_file = st.file_uploader("Upload First Signature", type=['png', 'jpg'])
img2_file = st.file_uploader("Upload Second Signature", type=['png', 'jpg'])

if img1_file and img2_file:
    img1 = Image.open(img1_file).convert("L")
    img2 = Image.open(img2_file).convert("L")
    
    st.image([img1, img2], caption=["Signature 1", "Signature 2"], width=150)

    model = load_model()
    with torch.no_grad():
        t1 = transform(img1).unsqueeze(0).to(device)
        t2 = transform(img2).unsqueeze(0).to(device)
        similarity = model(t1, t2).item()
        dissimilarity = 1 - similarity
        percent = round(dissimilarity * 100, 2)

        if similarity > 0.85:
            comment = "Highly Similar - Likely Genuine ✅"
            color = "green"
        elif similarity > 0.65:
            comment = "Moderate Similarity - Possibly Genuine ⚠️"
            color = "amber"
        elif similarity > 0.4:
            comment = "Low Similarity - Likely Forged ❌"
            color = "orange"
        else:
            comment = "Very Low Similarity - Definite Forgery 🚨"
            color = "red"

        st.markdown(f"### Dissimilarity: **{percent}%**")
        st.markdown(f"### Verdict: **<span style='color:{color}'>{comment}</span>**", unsafe_allow_html=True)
        st.progress(int(percent))
