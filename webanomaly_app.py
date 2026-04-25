from flask import Flask, jsonify, render_template, request
import io
import os

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

from model_visualization import render_model_visualization_html

# -------------------- Config --------------------
MODEL_PATH = "eurosat_autoencoder_anomaly_model_corrected.pth"
THRESHOLD_PATH = "anomaly_threshold1.npy"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORMAL_CLASS_LABEL = 4
NORMAL_CLASS_NAME = "Industrial"
LATENT_SCORE_WEIGHT = 0.35

MEAN = [0.3444, 0.3804, 0.4287]
STD = [0.2312, 0.1873, 0.1610]


# -------------------- Model --------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class BetterAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.dec1(z)
        z = self.dec2(z)
        z = self.dec3(z)
        return self.out(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# -------------------- App Setup --------------------
app = Flask(__name__)

preprocessor = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# Load model once at startup
try:
    model = BetterAutoencoder().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")


# Load threshold once at startup
if not os.path.exists(THRESHOLD_PATH):
    raise FileNotFoundError(
        f"{THRESHOLD_PATH} not found. Run webanomaly.py first to train and save threshold."
    )

try:
    ANOMALY_THRESHOLD = float(np.load(THRESHOLD_PATH).item())
except Exception as e:
    raise RuntimeError(f"Failed to load threshold from {THRESHOLD_PATH}: {e}")

PAPER_TITLE = "Hybrid QC-CNN: A Quantum–Classical Model for Fast Satellite Anomaly Detection"
PAPER_AUTHORS = (
    "Tummala Lakshmi Narasamma¹, Kandula Srikanth², Sanaka Venkata Jahnavi³, "
    "Vejandla Anji Naga Venkata Siva Sai⁴, Mulupuri Sharon⁵"
)
PAPER_AFFILIATION = (
    "¹,2,3,4,5 Department of Artificial Intelligence and Data Science, "
    "Seshadri Rao Gudlavalleru Engineering College, Gudlavalleru, Andhra Pradesh, India."
)
PAPER_CONTACTS = (
    "subhashenireddy224@gmail.com, srikanth.kandula15@gmail.com, "
    "sanakajahnavi@gmail.com, vejandlasai41@gmail.com, mulupurisharon10@gmail.com"
)
PAPER_ABSTRACT = (
    "Hybrid quantum-classical anomaly detection combines quantum feature extraction "
    "with classical CNN spatial processing to handle large satellite imagery more efficiently. "
    "This work uses amplitude encoding and variational quantum circuits to reduce complexity "
    "while preserving expressive anomaly features for EO datasets like EuroSAT."
)
PAPER_KEYWORDS = (
    "Hybrid Quantum-Classical Model, QC-CNN, Satellite Imagery, Earth Observation, "
    "Anomaly Detection, Amplitude Encoding, Quantum Feature Extraction, Hyperspectral Data, "
    "Remote Sensing, Environmental Monitoring"
)


def compute_anomaly_score(image: Image.Image) -> float:
    image = image.convert("RGB")
    image_tensor = preprocessor(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        recon = model(image_tensor)
        z_real = model.encode(image_tensor)
        z_recon = model.encode(recon)

        pixel_mse = torch.mean((recon - image_tensor) ** 2, dim=[1, 2, 3])
        latent_mse = torch.mean((z_recon - z_real) ** 2, dim=[1, 2, 3])
        score = pixel_mse + LATENT_SCORE_WEIGHT * latent_mse

    return float(score.item())


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EuroSAT Anomaly Detector</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #f5f6f8; color: #222; }
    .wrap { max-width: 720px; margin: 40px auto; background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 24px; }
    h1 { margin: 0 0 10px; }
    .meta { background: #f0f4ff; border-left: 4px solid #3766d6; padding: 10px; margin-bottom: 16px; }
    .btn { margin-top: 12px; padding: 10px 14px; border: 0; background: #1a73e8; color: #fff; border-radius: 6px; cursor: pointer; }
    .btn:hover { background: #165fbd; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>EuroSAT One-Class Anomaly Detector</h1>
    <p>Model expects <b>{{ normal_class }}</b> as normal. Everything else is anomaly.</p>
    <div class="meta">
      <div><b>Model:</b> {{ model_path }}</div>
      <div><b>Threshold:</b> {{ threshold }}</div>
    </div>
    <p><a href="/visualization">Open model layer and quantum visualization</a></p>
    <form method="post" action="/predict" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" required>
      <br>
      <button class="btn" type="submit">Analyze Image</button>
    </form>
  </div>
</body>
</html>
"""

RESULT_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Anomaly Result</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #f5f6f8; color: #222; }
    .wrap { max-width: 720px; margin: 40px auto; background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 24px; }
    .badge { display: inline-block; padding: 8px 12px; border-radius: 999px; color: #fff; font-weight: 700; background: {{ color }}; }
    .box { margin-top: 14px; padding: 12px; border-radius: 8px; background: #fafafa; border: 1px solid #ececec; }
    a { color: #1a73e8; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Analysis Result</h1>
    <span class="badge">{{ status }}</span>
    <div class="box">
      <p><b>Anomaly score:</b> {{ score }}</p>
      <p><b>Threshold:</b> {{ threshold }}</p>
      <p><b>Decision margin (score - threshold):</b> {{ margin }}</p>
      <p>{{ message }}</p>
    </div>
    <p><a href="/">Analyze another image</a></p>
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "web3_hero.html",
        paper_title=PAPER_TITLE,
        paper_authors=PAPER_AUTHORS,
        paper_affiliation=PAPER_AFFILIATION,
        paper_contacts=PAPER_CONTACTS,
        paper_abstract=PAPER_ABSTRACT,
        paper_keywords=PAPER_KEYWORDS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        score = compute_anomaly_score(image)
        is_anomaly = score > ANOMALY_THRESHOLD

        if is_anomaly:
            status = "ANOMALY"
            color = "#d93025"
            message = "This image is outside the normal Industrial class profile."
        else:
            status = "NORMAL"
            color = "#188038"
            message = "This image fits the learned Industrial class profile."

        margin = score - ANOMALY_THRESHOLD

        return render_template(
            "web3_result.html",
            status=status,
            color=color,
            score=f"{score:.6f}",
            threshold=f"{ANOMALY_THRESHOLD:.6f}",
            margin=f"{margin:.6f}",
            message=message,
        )
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "device": str(DEVICE),
            "model_path": MODEL_PATH,
            "threshold_path": THRESHOLD_PATH,
            "threshold": ANOMALY_THRESHOLD,
        }
    )


@app.route("/visualization", methods=["GET"])
def visualization():
    try:
        return render_model_visualization_html(model, DEVICE)
    except Exception as e:
        return jsonify({"error": f"Visualization failed: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5501, debug=True)
