
import copy
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# -------------------- Config --------------------
MODEL_PATH = "eurosat_autoencoder_anomaly_model_corrected.pth"
THRESHOLD_PATH = "anomaly_threshold.npy"
DATA_ROOT = "/Users/vejandlaanji/Desktop/EuroSat_data/eurosat_data/EuroSAT"

SEED = 42
BATCH_SIZE = 128
NUM_EPOCHS = 35
PATIENCE = 6
-LEARNING_RATE = 3e-4
-WEIGHT_DECAY = 1e-5
-# On macOS/Python 3.13, DataLoader workers require a strict __main__ guard.
-# This script executes training at top-level, so keep workers at 0 for stability.
-NUM_WORKERS = 0
-
-NORMAL_CLASS_LABEL = 4  # Industrial
-LATENT_SCORE_WEIGHT = 0.35
-
-MEAN = [0.3444, 0.3804, 0.4287]
-STD = [0.2312, 0.1873, 0.1610]
-
-
-def set_seed(seed: int) -> None:
-    random.seed(seed)
-    np.random.seed(seed)
-    torch.manual_seed(seed)
-    if torch.cuda.is_available():
-        torch.cuda.manual_seed_all(seed)
-
-
-set_seed(SEED)
-DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
-
-
-# -------------------- Data --------------------
-train_transform = transforms.Compose([
-    transforms.Resize((64, 64)),
-    transforms.RandomHorizontalFlip(),
-    transforms.RandomVerticalFlip(p=0.2),
-    transforms.RandomRotation(20),
-    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
-    transforms.ToTensor(),
-    transforms.Normalize(mean=MEAN, std=STD),
-])
-
-eval_transform = transforms.Compose([
-    transforms.Resize((64, 64)),
-    transforms.ToTensor(),
-    transforms.Normalize(mean=MEAN, std=STD),
-])
-
-try:
-    base_dataset = datasets.EuroSAT(root=DATA_ROOT, download=True)
-    train_dataset_full = datasets.EuroSAT(root=DATA_ROOT, transform=train_transform, download=False)
-    eval_dataset_full = datasets.EuroSAT(root=DATA_ROOT, transform=eval_transform, download=False)
-except Exception as e:
-    print(f"Error loading EuroSAT dataset: {e}")
-    sys.exit(1)
-
-all_labels = np.array(base_dataset.targets)
-normal_indices = np.where(all_labels == NORMAL_CLASS_LABEL)[0].tolist()
-anomaly_indices = np.where(all_labels != NORMAL_CLASS_LABEL)[0].tolist()
-
-train_normal_indices, holdout_normal_indices = train_test_split(
-    normal_indices, test_size=0.25, random_state=SEED
-)
-val_normal_indices, test_normal_indices = train_test_split(
-    holdout_normal_indices, test_size=0.5, random_state=SEED
-)
-
-train_dataset = Subset(train_dataset_full, train_normal_indices)
-val_dataset = Subset(eval_dataset_full, val_normal_indices)  # only normal for threshold/early stop
-test_dataset = Subset(eval_dataset_full, test_normal_indices + anomaly_indices)
-
-loader_kwargs = {
-    "batch_size": BATCH_SIZE,
-    "num_workers": NUM_WORKERS,
-    "pin_memory": torch.cuda.is_available(),
-}
-
-train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
-val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
-test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
-
-normal_class_name = base_dataset.classes[NORMAL_CLASS_LABEL]
-print(f"Normal Class: {normal_class_name}")
-print(f"Train normals: {len(train_dataset)} | Val normals: {len(val_dataset)} | Test mixed: {len(test_dataset)}")
-
-
-# -------------------- Model --------------------
-class ResidualBlock(nn.Module):
-    def __init__(self, channels: int):
-        super().__init__()
-        self.block = nn.Sequential(
-            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
-            nn.BatchNorm2d(channels),
-            nn.ReLU(inplace=True),
-            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
-            nn.BatchNorm2d(channels),
-        )
-        self.relu = nn.ReLU(inplace=True)
-
-    def forward(self, x: torch.Tensor) -> torch.Tensor:
-        return self.relu(x + self.block(x))
-
-
-class BetterAutoencoder(nn.Module):
-    def __init__(self):
-        super().__init__()
-        self.enc1 = nn.Sequential(
-            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
-            nn.BatchNorm2d(32),
-            nn.ReLU(inplace=True),
-            ResidualBlock(32),
-        )
-        self.enc2 = nn.Sequential(
-            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
-            nn.BatchNorm2d(64),
-            nn.ReLU(inplace=True),
-            ResidualBlock(64),
-        )
-        self.enc3 = nn.Sequential(
-            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
-            nn.BatchNorm2d(128),
-            nn.ReLU(inplace=True),
-            ResidualBlock(128),
-        )
-
-        self.dec1 = nn.Sequential(
-            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
-            nn.BatchNorm2d(64),
-            nn.ReLU(inplace=True),
-            ResidualBlock(64),
-        )
-        self.dec2 = nn.Sequential(
-            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
-            nn.BatchNorm2d(32),
-            nn.ReLU(inplace=True),
-            ResidualBlock(32),
-        )
-        self.dec3 = nn.Sequential(
-            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
-            nn.BatchNorm2d(16),
-            nn.ReLU(inplace=True),
-        )
-        self.out = nn.Conv2d(16, 3, kernel_size=3, padding=1)
-
-    def encode(self, x: torch.Tensor) -> torch.Tensor:
-        x = self.enc1(x)
-        x = self.enc2(x)
-        x = self.enc3(x)
-        return x
-
-    def decode(self, z: torch.Tensor) -> torch.Tensor:
-        z = self.dec1(z)
-        z = self.dec2(z)
-        z = self.dec3(z)
-        return self.out(z)
-
-    def forward(self, x: torch.Tensor) -> torch.Tensor:
-        return self.decode(self.encode(x))
-
-
-model = BetterAutoencoder().to(DEVICE)
-criterion_mse = nn.MSELoss()
-criterion_l1 = nn.L1Loss()
-optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
-scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
-
-
-def reconstruction_train_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
-    mse = criterion_mse(recon, target)
-    l1 = criterion_l1(recon, target)
-    return 0.8 * mse + 0.2 * l1
-
-
-def validate(loader: DataLoader, net: nn.Module) -> float:
-    net.eval()
-    running = 0.0
-    count = 0
-    with torch.no_grad():
-        for data, _ in loader:
-            data = data.to(DEVICE, non_blocking=True)
-            recon = net(data)
-            loss = reconstruction_train_loss(recon, data)
-            running += loss.item() * data.size(0)
-            count += data.size(0)
-    return running / max(count, 1)
-
-
-print(f"\n--- Starting Training on {DEVICE} for up to {NUM_EPOCHS} epochs ---")
-best_val_loss = float("inf")
-best_state = None
-no_improve_epochs = 0
-
-for epoch in range(NUM_EPOCHS):
-    model.train()
-    running_loss = 0.0
-    seen = 0
-
-    for data, _ in train_loader:
-        data = data.to(DEVICE, non_blocking=True)
-        optimizer.zero_grad()
-        recon = model(data)
-        loss = reconstruction_train_loss(recon, data)
-        loss.backward()
-        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
-        optimizer.step()
-
-        running_loss += loss.item() * data.size(0)
-        seen += data.size(0)
-
-    train_loss = running_loss / max(seen, 1)
-    val_loss = validate(val_loader, model)
-    scheduler.step(val_loss)
-
-    current_lr = optimizer.param_groups[0]["lr"]
-    print(
-        f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] "
-        f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}"
-    )
-
-    if val_loss < best_val_loss:
-        best_val_loss = val_loss
-        best_state = copy.deepcopy(model.state_dict())
-        no_improve_epochs = 0
-    else:
-        no_improve_epochs += 1
-        if no_improve_epochs >= PATIENCE:
-            print("Early stopping triggered.")
-            break
-
-if best_state is None:
-    best_state = model.state_dict()
-
-model.load_state_dict(best_state)
-torch.save(model.state_dict(), MODEL_PATH)
-print(f"\n*** Best anomaly model saved to: {MODEL_PATH} ***")
-
-
-# -------------------- Evaluation --------------------
-def anomaly_scores_with_latent(loader: DataLoader, net: BetterAutoencoder) -> tuple[np.ndarray, np.ndarray]:
-    net.eval()
-    scores = []
-    labels = []
-
-    with torch.no_grad():
-        for data, target in loader:
-            data = data.to(DEVICE, non_blocking=True)
-            recon = net(data)
-            z_real = net.encode(data)
-            z_recon = net.encode(recon)
-
-            pixel_mse = torch.mean((recon - data) ** 2, dim=[1, 2, 3])
-            latent_mse = torch.mean((z_recon - z_real) ** 2, dim=[1, 2, 3])
-            batch_scores = pixel_mse + LATENT_SCORE_WEIGHT * latent_mse
-
-            scores.extend(batch_scores.cpu().numpy().tolist())
-            labels.extend(target.cpu().numpy().tolist())
-
-    return np.array(scores), np.array(labels)
-
-
-val_scores, _ = anomaly_scores_with_latent(val_loader, model)
-threshold = float(np.percentile(val_scores, 95))
-np.save(THRESHOLD_PATH, np.array(threshold, dtype=np.float32))
-
-test_scores, test_labels = anomaly_scores_with_latent(test_loader, model)
-binary_labels = np.array([0 if label == NORMAL_CLASS_LABEL else 1 for label in test_labels], dtype=np.int32)
-roc_auc = roc_auc_score(binary_labels, test_scores)
-
-preds = (test_scores > threshold).astype(np.int32)
-accuracy = float((preds == binary_labels).mean())
-
-print("\n--- Evaluation Results ---")
-print(f"AUC-ROC: {roc_auc:.4f}")
-print(f"Threshold (95th percentile normal val score): {threshold:.6f}")
-print(f"Thresholded accuracy on mixed test set: {accuracy:.4f}")
-print(f"Threshold saved to: {THRESHOLD_PATH}")
-
-true_anomaly_indices = np.where(binary_labels == 1)[0]
-true_normal_indices = np.where(binary_labels == 0)[0]
-
-anomaly_sample_scores = [f"{test_scores[i]:<10.6f} | True Label: ANOMALY" for i in true_anomaly_indices[:5]]
-normal_sample_scores = [f"{test_scores[i]:<10.6f} | True Label: NORMAL" for i in true_normal_indices[:5]]
-
-print("\nFirst 5 Anomalies (Expected High Scores):")
-for line in anomaly_sample_scores:
-    print(line)
-
-print("\nFirst 5 Normal Samples (Expected Low Scores):")
-for line in normal_sample_scores:
-    print(line)











9999999999999
from __future__ import annotations

import io
import os

from flask import Flask, jsonify, render_template_string, request
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_visualization import render_model_visualization_html

try:
    import pennylane as qml
    from pennylane.templates.layers import StronglyEntanglingLayers

    HAS_QML = True
except Exception:
    qml = None
    StronglyEntanglingLayers = None
    HAS_QML = False


# -------------------- Config --------------------
MODEL_PATH = "eurosat_hybrid_qccnn_anomaly_model1.pth"
THRESHOLD_PATH = "anomaly_threshold2.npy"

INPUT_CHANNELS = 13
IMAGE_SIZE = 64
N_QUBITS = 8
N_QUANTUM_LAYERS = 3
Q_INPUT_DIM = N_QUBITS
Q_OUTPUT_DIM = 4
QML_DEVICE_NAME = os.getenv("QML_DEVICE", "default.qubit")
QML_DIFF_METHOD = os.getenv("QML_DIFF_METHOD", "backprop")
USE_TTA = os.getenv("USE_TTA", "1") == "1"
DECISION_MARGIN = float(os.getenv("DECISION_MARGIN", "0.02"))

NORMAL_CLASS_LABEL = 4
NORMAL_CLASS_NAME = "Industrial"

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available() and os.getenv("ENABLE_MPS", "0") == "1"
    else "cpu"
)

if torch.backends.mps.is_available() and DEVICE.type != "mps":
    print("INFO: MPS is available but disabled for stability. Set ENABLE_MPS=1 to force MPS execution.")

USE_QML = HAS_QML and DEVICE.type != "mps"


def _load_image_array_from_upload(image_bytes: bytes, filename: str) -> np.ndarray:
    # Prefer tifffile for multi-band TIFF uploads.
    lower = filename.lower()
    if lower.endswith(".tif") or lower.endswith(".tiff"):
        try:
            import tifffile  # type: ignore

            arr = tifffile.imread(io.BytesIO(image_bytes))
            return arr
        except Exception:
            pass

    with Image.open(io.BytesIO(image_bytes)) as img:
        return np.array(img)


def _to_chw_float_tensor(arr: np.ndarray) -> torch.Tensor:
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        # Convert HWC -> CHW when likely channel-last.
        if arr.shape[0] > arr.shape[-1]:
            arr = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported image dimensions: {arr.ndim}")

    x = torch.tensor(arr, dtype=torch.float32)
    if x.shape[0] < INPUT_CHANNELS:
        repeat_factor = int(np.ceil(INPUT_CHANNELS / max(x.shape[0], 1)))
        x = x.repeat(repeat_factor, 1, 1)[:INPUT_CHANNELS]
    elif x.shape[0] > INPUT_CHANNELS:
        x = x[:INPUT_CHANNELS]

    x = F.interpolate(
        x.unsqueeze(0),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    min_val = x.amin()
    max_val = x.amax()
    x = (x - min_val) / (max_val - min_val + 1e-6)
    return x


if USE_QML:
    dev = qml.device(QML_DEVICE_NAME, wires=N_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method=QML_DIFF_METHOD)
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(features=inputs, wires=range(N_QUBITS), rotation="Y")
        StronglyEntanglingLayers(weights=weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(Q_OUTPUT_DIM)]


class FallbackQuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Q_INPUT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, Q_OUTPUT_DIM),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridQCCNNAnomaly(nn.Module):
    def __init__(self, output_classes: int = 2, use_qml: bool | None = None):
        super().__init__()
        self.use_qml = USE_QML if use_qml is None else bool(use_qml and HAS_QML and DEVICE.type != "mps")
        self.classical_prep = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Flatten(),
        )
        self.to_quantum = nn.Sequential(
            nn.Linear(256, Q_INPUT_DIM),
            nn.Tanh(),
        )

        if self.use_qml:
            weight_shape = qml.StronglyEntanglingLayers.shape(
                n_layers=N_QUANTUM_LAYERS, n_wires=N_QUBITS
            )
            self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, {"weights": weight_shape})
        else:
            self.quantum_layer = FallbackQuantumLayer()

        self.classifier = nn.Sequential(
            nn.Linear(Q_OUTPUT_DIM, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classical_prep(x)
        x = torch.nan_to_num(x, nan=1e-8, posinf=1e10, neginf=-1e10)
        q_inputs = self.to_quantum(x) * np.pi
        q_inputs = torch.nan_to_num(q_inputs, nan=0.0, posinf=np.pi, neginf=-np.pi)
        q_out = self.quantum_layer(q_inputs)
        q_out = q_out.to(device=x.device, dtype=x.dtype)
        return self.classifier(q_out)


app = Flask(__name__)


def _load_checkpoint_model() -> tuple[nn.Module, dict]:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        meta = checkpoint
    else:
        # Backward-compat: if raw state_dict was saved.
        state_dict = checkpoint
        meta = {}

    requested_qml = bool(meta.get("uses_pennylane", USE_QML))
    model = HybridQCCNNAnomaly(output_classes=2, use_qml=requested_qml).to(
        device=DEVICE, dtype=torch.float32
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, meta


model, model_meta = _load_checkpoint_model()
MODEL_USES_QML = bool(getattr(model, "use_qml", USE_QML))

if not os.path.exists(THRESHOLD_PATH):
    raise FileNotFoundError(f"Threshold file not found: {THRESHOLD_PATH}")
ANOMALY_THRESHOLD = float(np.load(THRESHOLD_PATH).item())


def compute_anomaly_score(image_bytes: bytes, filename: str) -> float:
    arr = _load_image_array_from_upload(image_bytes, filename)
    x = _to_chw_float_tensor(arr).to(DEVICE)

    with torch.no_grad():
        views = [x]
        if USE_TTA:
            views.extend(
                [
                    torch.flip(x, dims=[2]),
                    torch.flip(x, dims=[1]),
                    torch.rot90(x, k=1, dims=[1, 2]),
                ]
            )

        view_scores: list[float] = []
        for v in views:
            logits = model(v.unsqueeze(0))
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
            probs = torch.softmax(logits, dim=1)[:, 1]
            view_scores.append(float(torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0).item()))
        score = float(np.mean(view_scores))
    return score


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hybrid QC-CNN Anomaly Detector</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #f5f6f8; color: #222; }
    .wrap { max-width: 760px; margin: 40px auto; background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 24px; }
    h1 { margin: 0 0 10px; }
    .meta { background: #f0f4ff; border-left: 4px solid #3766d6; padding: 10px; margin-bottom: 16px; }
    .btn { margin-top: 12px; padding: 10px 14px; border: 0; background: #1a73e8; color: #fff; border-radius: 6px; cursor: pointer; }
    .btn:hover { background: #165fbd; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Hybrid QC-CNN One-Class Anomaly Detector</h1>
    <p>Model expects <b>{{ normal_class }}</b> as normal. Everything else is anomaly.</p>
    <div class="meta">
      <div><b>Model:</b> {{ model_path }}</div>
      <div><b>Threshold:</b> {{ threshold }}</div>
      <div><b>Device:</b> {{ device }}</div>
      <div><b>Quantum active:</b> {{ quantum_active }}</div>
      <div><b>TTA active:</b> {{ tta_active }}</div>
      <div><b>Decision margin:</b> {{ decision_margin }}</div>
    </div>
    <p><a href="/visualization">Open model layer and quantum visualization</a></p>
    <form method="post" action="/predict" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*,.tif,.tiff" required>
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
    .wrap { max-width: 760px; margin: 40px auto; background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 24px; }
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
      <p><b>Anomaly score (P(anomaly)):</b> {{ score }}</p>
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
    return render_template_string(
        INDEX_HTML,
        normal_class=NORMAL_CLASS_NAME,
        threshold=f"{ANOMALY_THRESHOLD:.6f}",
        model_path=MODEL_PATH,
        device=str(DEVICE),
        quantum_active=str(MODEL_USES_QML),
        tta_active=str(USE_TTA),
        decision_margin=f"{DECISION_MARGIN:.4f}",
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
        score = compute_anomaly_score(image_bytes, file.filename)
        margin = score - ANOMALY_THRESHOLD
        abs_margin = abs(margin)
        uncertain = abs_margin < DECISION_MARGIN
        is_anomaly = score > ANOMALY_THRESHOLD

        if uncertain:
            status = "UNCERTAIN"
            color = "#f9ab00"
            message = (
                "Score is very close to threshold. Consider reviewing manually or "
                "using additional context before final action."
            )
        elif is_anomaly:
            status = "ANOMALY"
            color = "#d93025"
            message = "This image is outside the normal Industrial class profile."
        else:
            status = "NORMAL"
            color = "#188038"
            message = "This image fits the learned Industrial class profile."

        return render_template_string(
            RESULT_HTML,
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
            "use_qml": MODEL_USES_QML,
            "use_tta": USE_TTA,
            "decision_margin": DECISION_MARGIN,
            "model_meta": model_meta,
        }
    )


@app.route("/visualization", methods=["GET"])
def visualization():
    try:
        return render_model_visualization_html(model, DEVICE)
    except Exception as e:
        return jsonify({"error": f"Visualization failed: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5502, debug=True)

