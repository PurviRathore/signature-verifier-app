import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import gdown
import os

# ---------------------------
# Model Definition (Same as training)
# ---------------------------
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
        cos_sim = F.cosine_similarity(emb1, emb2)
        return cos_sim.unsqueeze(1)

# ---------------------------
# Load Model from Google Drive
# ---------------------------
@st.cache_resource
def load_model():
    model_path = "siamese_similarity_model.pth"
    if not os.path.exists(model_path):
        file_id = "15uNd8NyJNMeP3c7k4MKNv-aiUK5Wi7CI"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

    base = timm.create_model("xception", pretrained=True)
    base.conv1 = nn.Conv2d(1, base.conv1.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
    with torch.no_grad():
        base.conv1.weight[:, 0] = base.conv1.weight.mean(dim=1)

    model = SiameseRegressor(base)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# ---------------------------
# Image Preprocessing
# ---------------------------
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension

# ---------------------------
# Main App
# ---------------------------

st.title("📝 Signature Verifier")
st.write("Upload two signature images to compare their dissimilarity.")

# Display interpretation table
st.markdown("###  Dissimilarity Interpretation Table")
st.markdown("""
<style>
    .range-table td, .range-table th {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
    }
    .range-table tr:nth-child(even){background-color: #f2f2f2;}
    .range-table th {
        padding-top: 12px;
        padding-bottom: 12px;
        background-color: #333;
        color: white;
    }
</style>

<table class="range-table">
<tr>
    <th>Dissimilarity Range (%)</th>
    <th>Interpretation</th>
    <th>Color Indicator</th>
</tr>
<tr>
    <td>0 - 15</td>
    <td>Likely Genuine</td>
    <td style="background-color:#d4edda;">🟩 Green</td>
</tr>
<tr>
    <td>15 - 30</td>
    <td>Moderately Dissimilar</td>
    <td style="background-color:#fff3cd;">🟨 Amber</td>
</tr>
<tr>
    <td>30 - 100</td>
    <td>Highly Dissimilar – Possible Forgery</td>
    <td style="background-color:#f8d7da;">🟥 Red</td>
</tr>
</table>
""", unsafe_allow_html=True)

img1 = st.file_uploader("Upload First Signature", type=["png", "jpg", "jpeg"])
img2 = st.file_uploader("Upload Second Signature", type=["png", "jpg", "jpeg"])

if img1 and img2:
    image1 = Image.open(img1).convert("RGB")
    image2 = Image.open(img2).convert("RGB")

    st.image([image1, image2], caption=["Signature 1", "Signature 2"], width=250)

    input1 = preprocess_image(image1)
    input2 = preprocess_image(image2)

    model = load_model()

    with torch.no_grad():
        sim = model(input1, input2).item()
        dissim = (1 - sim) * 100

    st.subheader("🔍 Dissimilarity Score")
    st.metric(label="Dissimilarity %", value=f"{dissim:.2f}%", delta=None)

    if dissim > 20:
        st.error("❌ Highly Dissimilar – Possible Forgery")
    elif dissim > 10:
        st.warning("⚠️ Moderately Dissimilar – Investigate Further")
    else:
        st.success("✅ Low Dissimilarity – Likely Genuine")

# Show model F1 score (hardcoded)
f1_score_value = 99.15 # Replace with your actual F1
st.info(f"📈 **Model Accuracy**: `{f1_score_value}`%")




