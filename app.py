import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms
import gdown
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model architecture matching your training code
class SiameseRegressor(nn.Module):
    def __init__(self, base_model):
        super(SiameseRegressor, self).__init__()
        self.base = base_model
        self.base.fc = nn.Identity()  # Remove final FC
        self.similarity_head = nn.Sequential(
            nn.Linear(self.base.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()  # Output in range [0, 1]
        )

    def forward_once(self, x):
        return self.base(x)

    def forward(self, x1, x2):
        emb1 = self.forward_once(x1)
        emb2 = self.forward_once(x2)
        diff = torch.abs(emb1 - emb2)
        similarity_score = self.similarity_head(diff)
        return similarity_score

# Load model from Drive using gdown
@st.cache_resource
def load_model():
    file_id = "YOUR_FILE_ID_HERE"  # <- REPLACE with actual file ID from Drive
    output = "siamese_similarity_model.pth"
    if not os.path.exists(output):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

    base = timm.create_model("xception", pretrained=False)
    base.conv1 = nn.Conv2d(1, base.conv1.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
    with torch.no_grad():
        base.conv1.weight[:, 0] = base.conv1.weight.mean(dim=1)

    model = SiameseRegressor(base)
    model.load_state_dict(torch.load(output, map_location=device))
    model.eval().to(device)
    return model

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess(image):
    image = image.convert("L")  # convert to grayscale
    return transform(image).unsqueeze(0).to(device)

def get_dissimilarity_percent(similarity_tensor):
    similarity = similarity_tensor.item()
    return round((1 - similarity) * 100, 2)

def get_color_and_comment(percent):
    if percent <= 20:
        return "🟢", "Highly Similar (Likely Genuine)"
    elif percent <= 50:
        return "🟡", "Moderately Similar (Possibly Genuine)"
    elif percent <= 75:
        return "🟠", "Somewhat Dissimilar (Likely Forged)"
    else:
        return "🔴", "Highly Dissimilar (Fraud Likely)"

# Streamlit UI
st.title("✍️ Signature Dissimilarity Detector")

st.write("Upload two signature images to compare.")

img1 = st.file_uploader("Upload Signature 1", type=["png", "jpg", "jpeg"], key="1")
img2 = st.file_uploader("Upload Signature 2", type=["png", "jpg", "jpeg"], key="2")

if img1 and img2:
    with st.spinner("Analyzing..."):
        model = load_model()

        image1 = preprocess(Image.open(img1))
        image2 = preprocess(Image.open(img2))

        with torch.no_grad():
            similarity = model(image1, image2)

        dissim_percent = get_dissimilarity_percent(similarity)
        color, comment = get_color_and_comment(dissim_percent)

    st.markdown(f"### Dissimilarity: `{dissim_percent}%` {color}")
    st.markdown(f"**Assessment:** {comment}")
    st.image([img1, img2], caption=["Signature 1", "Signature 2"], width=150)



