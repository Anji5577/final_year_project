from __future__ import annotations

import io
import os
import time
import base64
from datetime import datetime

from flask import Flask, jsonify, render_template, request, session
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

try:
    from captum.attr import LayerAttribution, LayerGradCam

    HAS_CAPTUM = True
except Exception:
    LayerAttribution = None
    LayerGradCam = None
    HAS_CAPTUM = False


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
BAND_NAMES = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
]

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
app.secret_key = os.getenv("FLASK_SECRET_KEY", "hybrid-qccnn-dev-secret")


def _load_checkpoint_model() -> tuple[nn.Module, dict, dict]:
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
    return model, meta, state_dict


model, model_meta, model_state_dict = _load_checkpoint_model()
MODEL_USES_QML = bool(getattr(model, "use_qml", USE_QML))
CALIBRATION_T = float(model_meta.get("temperature", os.getenv("CALIBRATION_T", "1.34")))


def _build_model_variant(use_qml_variant: bool) -> nn.Module:
    variant = HybridQCCNNAnomaly(output_classes=2, use_qml=use_qml_variant).to(
        device=DEVICE, dtype=torch.float32
    )
    variant.load_state_dict(model_state_dict, strict=False)
    variant.eval()
    return variant


AB_MODEL_CLASSICAL = _build_model_variant(use_qml_variant=False)
AB_MODEL_QUANTUM = _build_model_variant(use_qml_variant=True) if HAS_QML and DEVICE.type != "mps" else None

if not os.path.exists(THRESHOLD_PATH):
    raise FileNotFoundError(f"Threshold file not found: {THRESHOLD_PATH}")
ANOMALY_THRESHOLD = float(np.load(THRESHOLD_PATH).item())

PAPER_TITLE = "Hybrid QC-CNN"
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


def _format_metric(value: float | None, percent: bool = False, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if percent:
        return f"{value * 100:.2f}%"
    return f"{value:.3f}{suffix}"


def _find_metric(meta: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in meta:
            try:
                return float(meta[key])
            except (TypeError, ValueError):
                continue
    return None


def _measure_inference_latency_ms(runs: int = 30, warmup: int = 5) -> float | None:
    try:
        sample = torch.randn(1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE, dtype=torch.float32)
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(sample)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(runs):
                _ = model(sample)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        return (elapsed / runs) * 1000.0
    except Exception:
        return None


def _build_benchmarks() -> list[dict[str, str]]:
    accuracy = _find_metric(model_meta, ("best_val_accuracy", "val_accuracy", "accuracy"))
    f1_score = _find_metric(model_meta, ("best_val_f1", "val_f1", "f1_score", "f1"))
    latency_ms = _measure_inference_latency_ms()

    return [
        {
            "metric": "Validation Accuracy",
            "value": _format_metric(accuracy, percent=True),
            "notes": "From checkpoint metadata.",
        },
        {
            "metric": "Validation F1 Score",
            "value": _format_metric(f1_score),
            "notes": "Not available in this checkpoint if shown as N/A.",
        },
        {
            "metric": "Average Inference Latency",
            "value": _format_metric(latency_ms, suffix=" ms/image"),
            "notes": f"Measured on {str(DEVICE)} with synthetic {IMAGE_SIZE}x{IMAGE_SIZE} input.",
        },
    ]


def _get_gradcam_target_layer(model_ref: nn.Module) -> nn.Module:
    if hasattr(model_ref, "classical_prep") and isinstance(model_ref.classical_prep, nn.Sequential):
        for layer in reversed(list(model_ref.classical_prep.children())):
            if isinstance(layer, nn.Conv2d):
                return layer
    for layer in model_ref.modules():
        if isinstance(layer, nn.Conv2d):
            return layer
    raise RuntimeError("No Conv2d layer found for GradCAM.")


def _normalize_0_1(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_min = torch.amin(x)
    x_max = torch.amax(x)
    return (x - x_min) / (x_max - x_min + 1e-8)


def _tensor_chw_to_rgb_uint8(x: torch.Tensor) -> np.ndarray:
    x = _normalize_0_1(x.detach().cpu())
    if x.shape[0] == 1:
        rgb = x.repeat(3, 1, 1)
    elif x.shape[0] >= 3:
        rgb = x[:3]
    else:
        repeat_factor = int(np.ceil(3 / max(int(x.shape[0]), 1)))
        rgb = x.repeat(repeat_factor, 1, 1)[:3]
    arr = (rgb.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def _heatmap_to_rgb(heatmap_2d: np.ndarray) -> np.ndarray:
    h = np.clip(heatmap_2d, 0.0, 1.0).astype(np.float32)
    r = np.clip(1.5 * h, 0.0, 1.0)
    g = np.clip(1.5 * (1.0 - np.abs(2.0 * h - 1.0)), 0.0, 1.0)
    b = np.clip(1.5 * (1.0 - h), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _encode_png_base64(img_uint8: np.ndarray) -> str:
    with io.BytesIO() as buf:
        Image.fromarray(img_uint8).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _compute_gradcam_native(x: torch.Tensor, target_class: int = 1, model_ref: nn.Module = model) -> np.ndarray:
    target_layer = _get_gradcam_target_layer(model_ref)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def _forward_hook(_, __, output):
        activations.append(output)

    def _backward_hook(_, grad_input, grad_output):
        del grad_input
        if grad_output:
            gradients.append(grad_output[0])

    f_handle = target_layer.register_forward_hook(_forward_hook)
    b_handle = target_layer.register_full_backward_hook(_backward_hook)
    try:
        model_ref.zero_grad(set_to_none=True)
        logits = model_ref(x.unsqueeze(0))
        score = logits[:, target_class].sum()
        score.backward()

        if not activations or not gradients:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients.")

        acts = activations[-1]
        grads = gradients[-1]
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam = _normalize_0_1(cam[0, 0]).detach().cpu().numpy()
        return cam
    finally:
        f_handle.remove()
        b_handle.remove()
        model_ref.zero_grad(set_to_none=True)


def _compute_gradcam_captum(x: torch.Tensor, target_class: int = 1, model_ref: nn.Module = model) -> np.ndarray:
    if not HAS_CAPTUM:
        raise RuntimeError("Captum is not available.")
    target_layer = _get_gradcam_target_layer(model_ref)
    layer_gc = LayerGradCam(model_ref, target_layer)
    attribution = layer_gc.attribute(x.unsqueeze(0), target=target_class)
    upsampled = LayerAttribution.interpolate(attribution, (IMAGE_SIZE, IMAGE_SIZE))
    cam = F.relu(upsampled[0, 0])
    cam = _normalize_0_1(cam).detach().cpu().numpy()
    model_ref.zero_grad(set_to_none=True)
    return cam


def compute_explainability_from_tensor(
    x: torch.Tensor, target_class: int = 1, model_ref: nn.Module = model
) -> dict[str, object]:
    base_rgb = _tensor_chw_to_rgb_uint8(x)
    was_training = model_ref.training
    model_ref.eval()
    try:
        cam: np.ndarray
        method = "gradcam_native"
        try:
            cam = _compute_gradcam_captum(x, target_class=target_class, model_ref=model_ref)
            method = "captum_layer_gradcam"
        except Exception:
            cam = _compute_gradcam_native(x, target_class=target_class, model_ref=model_ref)

        heatmap_rgb = _heatmap_to_rgb(cam)
        alpha = 0.45
        overlay = np.clip((1.0 - alpha) * base_rgb + alpha * heatmap_rgb, 0, 255).astype(np.uint8)

        return {
            "method": method,
            "heatmap_png_base64": _encode_png_base64(heatmap_rgb),
            "overlay_png_base64": _encode_png_base64(overlay),
            "input_png_base64": _encode_png_base64(base_rgb),
            "target_class": target_class,
        }
    finally:
        if was_training:
            model_ref.train()


def compute_explainability(image_bytes: bytes, filename: str, target_class: int = 1) -> dict[str, object]:
    arr = _load_image_array_from_upload(image_bytes, filename)
    x = _to_chw_float_tensor(arr).to(device=DEVICE, dtype=torch.float32)
    return compute_explainability_from_tensor(x=x, target_class=target_class, model_ref=model)


def _build_tta_views(x: torch.Tensor) -> list[torch.Tensor]:
    views = [x]
    if USE_TTA:
        views.extend(
            [
                torch.flip(x, dims=[2]),
                torch.flip(x, dims=[1]),
                torch.rot90(x, k=1, dims=[1, 2]),
            ]
        )
    return views


def _predict_with_uncertainty(model_ref: nn.Module, x: torch.Tensor, temperature: float) -> dict[str, float]:
    t = max(float(temperature), 1e-6)
    view_scores: list[float] = []
    with torch.no_grad():
        for v in _build_tta_views(x):
            logits = model_ref(v.unsqueeze(0))
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6) / t
            probs = torch.softmax(logits, dim=1)[:, 1]
            view_scores.append(float(torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0).item()))
    mean_score = float(np.mean(view_scores))
    sigma = float(np.std(view_scores)) if len(view_scores) > 1 else 0.0
    return {
        "score_mean": mean_score,
        "sigma": sigma,
        "ci_low": max(0.0, mean_score - sigma),
        "ci_high": min(1.0, mean_score + sigma),
        "views": len(view_scores),
    }


def _compute_channel_contribution(x: torch.Tensor, target_class: int = 1) -> list[dict[str, float | str]]:
    inp = x.clone().detach().unsqueeze(0).requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(inp) / max(CALIBRATION_T, 1e-6)
    loss = logits[:, target_class].sum()
    loss.backward()
    grad = torch.nan_to_num(inp.grad[0], nan=0.0, posinf=0.0, neginf=0.0)
    channel_values = grad.norm(dim=(1, 2)).detach().cpu().numpy()
    total = float(np.sum(channel_values) + 1e-8)
    rows: list[dict[str, float | str]] = []
    for idx in range(INPUT_CHANNELS):
        label = BAND_NAMES[idx] if idx < len(BAND_NAMES) else f"C{idx+1:02d}"
        raw = float(channel_values[idx]) if idx < len(channel_values) else 0.0
        rows.append({"channel": label, "value": raw, "percent": (raw / total) * 100.0})
    rows.sort(key=lambda r: float(r["value"]), reverse=True)
    model.zero_grad(set_to_none=True)
    return rows


def _compute_attention_rollout_map(x: torch.Tensor) -> np.ndarray:
    cur = x.unsqueeze(0)
    pools: list[dict[str, object]] = []
    for layer in model.classical_prep:
        if isinstance(layer, nn.MaxPool2d):
            out, idx = F.max_pool2d(
                cur,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                ceil_mode=layer.ceil_mode,
                return_indices=True,
            )
            pools.append(
                {
                    "indices": idx,
                    "output_size": cur.shape,
                    "kernel_size": layer.kernel_size,
                    "stride": layer.stride,
                    "padding": layer.padding,
                }
            )
            cur = out
        elif isinstance(layer, nn.Flatten):
            break
        else:
            cur = layer(cur)

    importance = torch.abs(cur)
    for entry in reversed(pools):
        idx_tensor = entry["indices"]
        if importance.shape[1] != idx_tensor.shape[1]:
            importance = importance.mean(dim=1, keepdim=True).repeat(1, idx_tensor.shape[1], 1, 1)
        importance = F.max_unpool2d(
            importance,
            idx_tensor,
            kernel_size=entry["kernel_size"],
            stride=entry["stride"],
            padding=entry["padding"],
            output_size=entry["output_size"],
        )
    rollout = _normalize_0_1(importance.mean(dim=1)[0]).detach().cpu().numpy()
    return rollout


def _compute_quantum_classical_ab(x: torch.Tensor) -> dict[str, float | str | None]:
    quantum_model = AB_MODEL_QUANTUM if AB_MODEL_QUANTUM is not None else model
    quantum_out = _predict_with_uncertainty(quantum_model, x, CALIBRATION_T)
    classical_out = _predict_with_uncertainty(AB_MODEL_CLASSICAL, x, CALIBRATION_T)
    q_score = float(quantum_out["score_mean"])
    c_score = float(classical_out["score_mean"])
    delta = q_score - c_score
    return {
        "quantum_score": q_score,
        "classical_score": c_score,
        "quantum_delta": delta,
        "quantum_model": "quantum" if AB_MODEL_QUANTUM is not None else "fallback",
        "classical_model": "fallback",
    }


def _parse_target_class(value: str | None) -> int:
    if value is None:
        return 1
    text = str(value).strip().lower()
    if text in {"1", "anomaly", "anom"}:
        return 1
    if text in {"0", "normal", "norm"}:
        return 0
    return 1


def _build_prediction_status(score: float) -> dict[str, str]:
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

    return {
        "status": status,
        "color": color,
        "message": message,
        "margin": f"{margin:.6f}",
    }


def compute_anomaly_score(image_bytes: bytes, filename: str) -> float:
    arr = _load_image_array_from_upload(image_bytes, filename)
    x = _to_chw_float_tensor(arr).to(DEVICE)
    out = _predict_with_uncertainty(model, x, CALIBRATION_T)
    return float(out["score_mean"])


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
    return render_template(
        "web3_hero.html",
        paper_title=PAPER_TITLE,
        paper_authors=PAPER_AUTHORS,
        paper_affiliation=PAPER_AFFILIATION,
        paper_contacts=PAPER_CONTACTS,
        paper_abstract=PAPER_ABSTRACT,
        paper_keywords=PAPER_KEYWORDS,
    )


@app.route("/model-card", methods=["GET"])
def model_card():
    return render_template(
        "model_card.html",
        paper_title=PAPER_TITLE,
        paper_authors=PAPER_AUTHORS,
        paper_affiliation=PAPER_AFFILIATION,
        paper_contacts=PAPER_CONTACTS,
        paper_abstract=PAPER_ABSTRACT,
        paper_keywords=PAPER_KEYWORDS,
        benchmarks=_build_benchmarks(),
        model_path=MODEL_PATH,
        threshold=ANOMALY_THRESHOLD,
        device=str(DEVICE),
        quantum_active=MODEL_USES_QML,
        tta_active=USE_TTA,
        decision_margin=DECISION_MARGIN,
    )


@app.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist("files")
    if not files:
        single = request.files.get("file")
        files = [single] if single is not None else []

    valid_files = [f for f in files if f is not None and f.filename]
    if not valid_files:
        return jsonify({"error": "No selected files"}), 400

    try:
        results: list[dict[str, object]] = []
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for file in valid_files:
            image_bytes = file.read()
            if not image_bytes:
                continue

            arr = _load_image_array_from_upload(image_bytes, file.filename)
            x = _to_chw_float_tensor(arr).to(device=DEVICE, dtype=torch.float32)
            pred = _predict_with_uncertainty(model, x, CALIBRATION_T)
            score = float(pred["score_mean"])
            explain_anomaly = compute_explainability_from_tensor(x, target_class=1, model_ref=model)
            explain_normal = compute_explainability_from_tensor(x, target_class=0, model_ref=model)
            channel_rows = _compute_channel_contribution(x, target_class=1)
            rollout = _compute_attention_rollout_map(x)
            rollout_rgb = _heatmap_to_rgb(rollout)
            input_rgb = _tensor_chw_to_rgb_uint8(x)
            rollout_overlay = np.clip(0.55 * input_rgb + 0.45 * rollout_rgb, 0, 255).astype(np.uint8)
            ab_compare = _compute_quantum_classical_ab(x)
            status_pack = _build_prediction_status(score)

            results.append(
                {
                    "filename": file.filename,
                    "score": f"{score:.6f}",
                    "score_float": round(score, 6),
                    "sigma": f"{float(pred['sigma']):.6f}",
                    "sigma_float": round(float(pred["sigma"]), 6),
                    "ci_low": f"{float(pred['ci_low']):.6f}",
                    "ci_high": f"{float(pred['ci_high']):.6f}",
                    "tta_views": int(pred["views"]),
                    "calibration_t": CALIBRATION_T,
                    "threshold": f"{ANOMALY_THRESHOLD:.6f}",
                    "threshold_float": ANOMALY_THRESHOLD,
                    "margin": status_pack["margin"],
                    "status": status_pack["status"],
                    "color": status_pack["color"],
                    "message": status_pack["message"],
                    "input_image_base64": explain_anomaly.get("input_png_base64", ""),
                    "heatmap_anomaly_base64": explain_anomaly.get("heatmap_png_base64", ""),
                    "heatmap_normal_base64": explain_normal.get("heatmap_png_base64", ""),
                    "rollout_heatmap_base64": _encode_png_base64(rollout_rgb),
                    "rollout_overlay_base64": _encode_png_base64(rollout_overlay),
                    "channel_contributions": channel_rows,
                    "ab_compare": ab_compare,
                    "explain_method": explain_anomaly.get("method", "unavailable"),
                }
            )

        if not results:
            return jsonify({"error": "Uploaded files are empty or unsupported."}), 400

        results.sort(key=lambda item: float(item["score_float"]), reverse=True)

        history = session.get("prediction_history", [])
        for row in results:
            history.insert(
                0,
                {
                    "timestamp": generated_at,
                    "filename": row["filename"],
                    "status": row["status"],
                    "score": row["score"],
                    "margin": row["margin"],
                },
            )
        session["prediction_history"] = history[:10]
        session.modified = True

        export_rows = [
            {
                "timestamp": generated_at,
                "filename": row["filename"],
                "status": row["status"],
                "score": row["score_float"],
                "sigma": row["sigma_float"],
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "tta_views": row["tta_views"],
                "calibration_t": row["calibration_t"],
                "threshold": row["threshold_float"],
                "margin": float(row["margin"]),
                "explainability_method": row["explain_method"],
                "quantum_score": row["ab_compare"]["quantum_score"],
                "classical_score": row["ab_compare"]["classical_score"],
                "quantum_delta": row["ab_compare"]["quantum_delta"],
            }
            for row in results
        ]

        return render_template(
            "web3_dashboard.html",
            results=results,
            threshold=f"{ANOMALY_THRESHOLD:.6f}",
            threshold_float=ANOMALY_THRESHOLD,
            history=session.get("prediction_history", []),
            export_rows=export_rows,
            generated_at=generated_at,
        )
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@app.route("/explain", methods=["POST"])
def explain():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    try:
        image_bytes = file.read()
        target_class = _parse_target_class(request.form.get("target_class") or request.args.get("target_class"))
        explain_out = compute_explainability(image_bytes, file.filename, target_class=target_class)
        explain_out["target_class_name"] = "anomaly" if target_class == 1 else "normal"
        return jsonify(explain_out)
    except Exception as e:
        return jsonify({"error": f"Explainability failed: {e}"}), 500


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
            "captum_available": HAS_CAPTUM,
            "calibration_temperature": CALIBRATION_T,
            "ab_quantum_available": AB_MODEL_QUANTUM is not None,
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
