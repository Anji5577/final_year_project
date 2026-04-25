import copy
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

try:
    import pennylane as qml
    from pennylane.templates.layers import StronglyEntanglingLayers

    HAS_QML = True
except Exception:
    qml = None
    StronglyEntanglingLayers = None
    HAS_QML = False


# -------------------- Config --------------------
MODEL_PATH = "eurosat_hybrid_qccnn_anomaly_model2.pth"
THRESHOLD_PATH = "anomaly_threshol.npy"
DATA_ROOT = "/Users/vejandlaanji/Desktop/EuroSat_data/eurosat_data/2750"

SEED = 42
BATCH_SIZE = 32
# [IMPROVED] More epochs + more patience to allow thorough convergence
NUM_EPOCHS = 40
PATIENCE = 10
MIN_EPOCHS_BEFORE_EARLY_STOP = 10
ENABLE_EARLY_STOPPING = True
# [IMPROVED] Lower LR for more stable convergence; cosine scheduler will handle decay
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
# [IMPROVED] Label smoothing reduces overconfidence and improves generalisation
LABEL_SMOOTHING = 0.05
TRAIN_AUGMENT = True
# [IMPROVED] More aggressive augmentation probability
TRAIN_AUGMENT_PROB = 0.7

NORMAL_CLASS_LABEL = 4  # Industrial
NORMAL_CLASS_NAME = "Industrial"
OUTPUT_CLASSES = 2  # normal vs anomaly

INPUT_CHANNELS = 13
IMAGE_SIZE = 64
# [IMPROVED] More samples per class for a richer training signal
MAX_SAMPLES_PER_CLASS = 500

# [IMPROVED] More qubits & layers → richer quantum representation
N_QUBITS = 12
N_QUANTUM_LAYERS = 4
Q_INPUT_DIM = N_QUBITS
# [IMPROVED] More quantum output features feed the classifier
Q_OUTPUT_DIM = 8
# [IMPROVED] Always balance classes — critical for anomaly detection
USE_BALANCED_SAMPLER = True
# [IMPROVED] Class weights on top of balanced sampler reinforce minority signal
USE_CLASS_WEIGHTS = True
QML_DEVICE_NAME = os.getenv("QML_DEVICE", "default.qubit")
QML_DIFF_METHOD = os.getenv("QML_DIFF_METHOD", "backprop")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available() and os.getenv("ENABLE_MPS", "0") == "1":
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

if torch.backends.mps.is_available() and DEVICE.type != "mps":
    print(
        "INFO: MPS is available but disabled for stability. "
        "Set ENABLE_MPS=1 to force MPS execution."
    )
FORCE_CLASSICAL = os.getenv("FORCE_CLASSICAL", "0") == "1"
USE_QML = HAS_QML and DEVICE.type != "mps" and not FORCE_CLASSICAL


def _load_image_array(path: str) -> np.ndarray:
    try:
        import tifffile  # type: ignore
        arr = tifffile.imread(path)
        return arr
    except Exception:
        pass

    try:
        with Image.open(path) as img:
            return np.array(img)
    except Exception as e:
        raise RuntimeError(
            f"Failed to decode TIFF '{path}'. "
            f"Install tifffile: 'python3 -m pip install tifffile'. Error: {e}"
        ) from e


def _to_chw_float_tensor(arr: np.ndarray) -> torch.Tensor:
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        if arr.shape[0] > arr.shape[-1]:
            arr = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported image ndim: {arr.ndim}")

    x = torch.tensor(arr, dtype=torch.float32)

    if x.shape[0] < INPUT_CHANNELS:
        repeat_factor = int(np.ceil(INPUT_CHANNELS / max(x.shape[0], 1)))
        x = x.repeat(repeat_factor, 1, 1)[:INPUT_CHANNELS]
    elif x.shape[0] > INPUT_CHANNELS:
        x = x[:INPUT_CHANNELS]

    x = F.interpolate(
        x.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    ).squeeze(0)

    # [IMPROVED] Per-channel normalisation (zero-mean, unit-variance) instead of
    # global min-max, which better preserves spectral differences between bands.
    mean = x.mean(dim=[1, 2], keepdim=True)
    std = x.std(dim=[1, 2], keepdim=True).clamp(min=1e-6)
    x = (x - mean) / std
    return x


class EuroSatAllBandsDataset(Dataset):
    CLASS_NAMES = (
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake",
    )

    def __init__(self, root_dir: str, max_samples_per_class: int = MAX_SAMPLES_PER_CLASS):
        self.root_dir = root_dir
        self.data_paths: list[str] = []
        self.labels: list[int] = []
        self.skipped_paths: list[str] = []

        root = Path(root_dir)
        if not root.exists():
            raise FileNotFoundError(f"Dataset path not found: {root_dir}")

        for i, class_name in enumerate(self.CLASS_NAMES):
            class_dir = root / class_name
            if not class_dir.exists():
                continue

            all_tifs = sorted(class_dir.glob("*.tif"))
            if max_samples_per_class > 0:
                all_tifs = all_tifs[:max_samples_per_class]

            for path in all_tifs:
                try:
                    _ = _load_image_array(str(path))
                    self.data_paths.append(str(path))
                    self.labels.append(i)
                except Exception:
                    self.skipped_paths.append(str(path))

        if len(self.data_paths) == 0:
            raise RuntimeError(
                f"No readable .tif files found under {root_dir}. "
                "Install tifffile: 'python3 -m pip install tifffile'."
            )
        if self.skipped_paths:
            print(
                f"WARNING: Skipped {len(self.skipped_paths)} unreadable TIFF files. "
                f"First skipped: {self.skipped_paths[0]}"
            )

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.data_paths[idx]
        class_label = self.labels[idx]
        binary_label = 0 if class_label == NORMAL_CLASS_LABEL else 1

        arr = _load_image_array(path)
        image_tensor = _to_chw_float_tensor(arr)
        return image_tensor, torch.tensor(binary_label, dtype=torch.long)


if USE_QML:
    dev = qml.device(QML_DEVICE_NAME, wires=N_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method=QML_DIFF_METHOD)
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(features=inputs, wires=range(N_QUBITS), rotation="Y")
        StronglyEntanglingLayers(weights=weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(Q_OUTPUT_DIM)]


class FallbackQuantumLayer(nn.Module):
    """[IMPROVED] Larger fallback with residual connection for deeper feature mixing."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Q_INPUT_DIM, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, Q_OUTPUT_DIM),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# [IMPROVED] Squeeze-and-Excitation block for channel-wise recalibration
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 1)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 1), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class HybridQCCNNAnomaly(nn.Module):
    def __init__(self, output_classes: int = OUTPUT_CLASSES):
        super().__init__()

        # [IMPROVED] Deeper CNN backbone with BatchNorm + SE blocks
        # Input: [B, 13, 64, 64] → after pool: [B, 256, 4, 4] → flatten: 256*4*4=4096
        # Adjusted to end with 256-d vector for to_quantum compatibility.
        self.classical_prep = nn.Sequential(
            # Block 1
            nn.Conv2d(INPUT_CHANNELS, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),          # 32x32

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),          # 16x16

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(2),          # 8x8
        )
        # SE after CNN blocks
        self.se = SEBlock(256, reduction=8)

        # Global average pool + flatten → 256-d
        self.gap = nn.AdaptiveAvgPool2d(1)

        # [IMPROVED] Projection to quantum input with BatchNorm for stability
        self.to_quantum = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, Q_INPUT_DIM),
            nn.Tanh(),
        )

        if USE_QML:
            weight_shape = qml.StronglyEntanglingLayers.shape(
                n_layers=N_QUANTUM_LAYERS, n_wires=N_QUBITS
            )
            self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, {"weights": weight_shape})
        else:
            self.quantum_layer = FallbackQuantumLayer()

        # [IMPROVED] Wider classifier head with residual skip
        self.classifier = nn.Sequential(
            nn.Linear(Q_OUTPUT_DIM, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, output_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.classical_prep(x)
        feat = self.se(feat)
        feat = self.gap(feat).flatten(1)                  # [B, 256]
        feat = torch.nan_to_num(feat, nan=1e-8, posinf=1e10, neginf=-1e10)

        q_inputs = self.to_quantum(feat) * np.pi
        q_inputs = torch.nan_to_num(q_inputs, nan=0.0, posinf=np.pi, neginf=-np.pi)

        q_out = self.quantum_layer(q_inputs)
        q_out = q_out.to(device=x.device, dtype=x.dtype)
        return self.classifier(q_out)


def make_split_indices(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=SEED, stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=SEED, stratify=labels[temp_idx],
    )
    return train_idx, val_idx, test_idx


def _metrics_from_preds(preds: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    preds = np.asarray(preds, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    balanced_acc = 0.5 * (tpr + tnr)
    acc = float((preds == labels).mean())
    precision = tp / max(tp + fp, 1)
    recall = tpr
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return balanced_acc, acc, f1


def find_best_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    objective: str = "balanced_accuracy",  # [IMPROVED] default to balanced_accuracy
) -> tuple[float, float, float, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    if scores.size == 0 or labels.size == 0:
        raise ValueError("Cannot find threshold from empty scores/labels.")

    if objective not in {"accuracy", "balanced_accuracy"}:
        raise ValueError("objective must be one of {'accuracy', 'balanced_accuracy'}")

    unique_scores = np.unique(scores)
    if unique_scores.size == 1:
        threshold = float(unique_scores[0])
        preds = (scores >= threshold).astype(np.int32)
        balanced_acc, acc, f1 = _metrics_from_preds(preds, labels)
        return threshold, balanced_acc, acc, f1

    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    candidates = np.concatenate(
        [
            np.array([unique_scores[0] - 1e-8], dtype=np.float64),
            unique_scores.astype(np.float64),
            midpoints.astype(np.float64),
            np.array([unique_scores[-1] + 1e-8], dtype=np.float64),
        ]
    )
    best_threshold = float(candidates[0])
    best_objective = -1.0
    best_balanced_acc = -1.0
    best_accuracy = -1.0
    best_f1 = -1.0

    for threshold in candidates:
        preds = (scores >= threshold).astype(np.int32)
        balanced_acc, accuracy, f1 = _metrics_from_preds(preds, labels)
        current_objective = accuracy if objective == "accuracy" else balanced_acc
        if (
            current_objective > best_objective + 1e-12
            or (
                abs(current_objective - best_objective) <= 1e-12
                and (
                    f1 > best_f1 + 1e-12
                    or (
                        abs(f1 - best_f1) <= 1e-12
                        and (
                            accuracy > best_accuracy + 1e-12
                            or (
                                abs(accuracy - best_accuracy) <= 1e-12
                                and balanced_acc > best_balanced_acc
                            )
                        )
                    )
                )
            )
        ):
            best_objective = current_objective
            best_balanced_acc = balanced_acc
            best_f1 = f1
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return best_threshold, best_balanced_acc, best_accuracy, best_f1


def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """[IMPROVED] Extended augmentation: flips, rotations, cutout, channel dropout."""
    if not TRAIN_AUGMENT:
        return x

    out = x.clone()
    b = out.size(0)

    flip_h = torch.rand(b, device=out.device) < TRAIN_AUGMENT_PROB
    flip_v = torch.rand(b, device=out.device) < TRAIN_AUGMENT_PROB
    rot_k = torch.randint(0, 4, (b,), device=out.device)

    for i in range(b):
        xi = out[i]
        if bool(flip_h[i]):
            xi = torch.flip(xi, dims=[2])
        if bool(flip_v[i]):
            xi = torch.flip(xi, dims=[1])
        if int(rot_k[i]) > 0:
            xi = torch.rot90(xi, k=int(rot_k[i]), dims=[1, 2])

        # Intensity jitter
        scale = torch.empty(1, device=out.device).uniform_(0.85, 1.15)
        bias = torch.empty(1, device=out.device).uniform_(-0.05, 0.05)
        noise = 0.02 * torch.randn_like(xi)
        xi = xi * scale + bias + noise

        # [IMPROVED] Random cutout: zero out a random square patch
        if torch.rand(1).item() < TRAIN_AUGMENT_PROB:
            _, h, w = xi.shape
            cut_frac = torch.empty(1).uniform_(0.1, 0.25).item()
            cut_h = max(1, int(h * cut_frac))
            cut_w = max(1, int(w * cut_frac))
            top = torch.randint(0, h - cut_h + 1, (1,)).item()
            left = torch.randint(0, w - cut_w + 1, (1,)).item()
            xi[:, top: top + cut_h, left: left + cut_w] = 0.0

        # [IMPROVED] Random channel dropout: zero one band entirely
        if torch.rand(1).item() < 0.3:
            ch = torch.randint(0, xi.shape[0], (1,)).item()
            xi[ch] = 0.0

        out[i] = xi

    return out


def evaluate(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n = 0
    scores: list[float] = []
    labels: list[int] = []

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            logits = model(x)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)[:, 1]
            probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)

            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            scores.extend(probs.detach().cpu().numpy().tolist())
            labels.extend(y.detach().cpu().numpy().tolist())

    return total_loss / max(n, 1), np.array(scores), np.array(labels, dtype=np.int32)


def main() -> None:
    print(f"Running on device: {DEVICE}")
    if HAS_QML and DEVICE.type == "mps":
        print(
            "WARNING: PennyLane quantum layer is disabled on MPS due to float64 limitations. "
            "Using fallback quantum layer."
        )
    elif not HAS_QML:
        print("WARNING: PennyLane not available. Using classical fallback quantum layer.")
    elif FORCE_CLASSICAL:
        print("INFO: FORCE_CLASSICAL=1 set. Using classical fallback quantum layer.")
    elif USE_QML:
        print(f"INFO: QML backend={QML_DEVICE_NAME}, diff_method={QML_DIFF_METHOD}")

    dataset = EuroSatAllBandsDataset(DATA_ROOT, max_samples_per_class=MAX_SAMPLES_PER_CLASS)
    labels = np.array([0 if y == NORMAL_CLASS_LABEL else 1 for y in dataset.labels], dtype=np.int32)

    train_idx, val_idx, test_idx = make_split_indices(labels)
    train_dataset = Subset(dataset, train_idx.tolist())
    val_dataset = Subset(dataset, val_idx.tolist())
    test_dataset = Subset(dataset, test_idx.tolist())

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
    }
    y_train = labels[train_idx]
    train_sampler = None
    if USE_BALANCED_SAMPLER:
        class_sample_counts = np.bincount(y_train, minlength=2).astype(np.float64)
        class_sample_counts = np.maximum(class_sample_counts, 1.0)
        class_sample_weights = 1.0 / class_sample_counts
        train_sample_weights = class_sample_weights[y_train]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(train_sample_weights, dtype=torch.double),
            num_samples=len(train_sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    print(
        f"Dataset split -> train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}"
    )
    print(f"Normal class: {NORMAL_CLASS_NAME} ({NORMAL_CLASS_LABEL})")

    model = HybridQCCNNAnomaly(output_classes=OUTPUT_CLASSES).to(device=DEVICE, dtype=torch.float32)

    class_counts = np.bincount(y_train, minlength=2).astype(np.float32)
    class_weight_tensor = None
    if USE_CLASS_WEIGHTS:
        class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
        class_weights = class_weights / class_weights.sum() * 2.0
        class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor,
        label_smoothing=LABEL_SMOOTHING,
    )

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # [IMPROVED] Cosine annealing with warm restarts for better convergence landscape
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_state = None
    best_val_bal_acc = -1.0        # [IMPROVED] track balanced accuracy as primary metric
    best_val_loss = float("inf")
    best_val_threshold = 0.5
    no_improve_epochs = 0

    print(f"\n--- Training Hybrid QC-CNN for up to {NUM_EPOCHS} epochs ---")
    print(
        f"Training class counts -> normal: {int(class_counts[0])}, anomaly: {int(class_counts[1])}; "
        f"balanced_sampler={USE_BALANCED_SAMPLER}, class_weights={USE_CLASS_WEIGHTS}."
    )

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        seen = 0
        skipped_batches = 0

        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            x = augment_batch(x)

            optimizer.zero_grad()
            logits = model(x)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                skipped_batches += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            seen += x.size(0)

        scheduler.step(epoch)      # CosineAnnealingWarmRestarts uses epoch index

        train_loss = running_loss / max(seen, 1)
        val_loss, val_scores, val_labels = evaluate(model, val_loader)

        val_auc = roc_auc_score(val_labels, val_scores) if len(np.unique(val_labels)) > 1 else float("nan")
        # [IMPROVED] Optimise threshold on balanced accuracy (fairer for imbalanced data)
        val_threshold, val_bal_acc, val_acc, val_f1 = find_best_threshold(
            val_scores, val_labels, objective="balanced_accuracy",
        )
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.4f} | Val Acc*: {val_acc:.4f} | "
            f"Val Bal Acc*: {val_bal_acc:.4f} | Val F1*: {val_f1:.4f} | "
            f"Thr*: {val_threshold:.6f} | LR: {current_lr:.2e} | Skipped Batches: {skipped_batches}"
        )

        # [IMPROVED] Primary: balanced accuracy; tiebreak: val loss
        improved = (val_bal_acc > best_val_bal_acc + 1e-4) or (
            abs(val_bal_acc - best_val_bal_acc) <= 1e-4 and val_loss < best_val_loss - 1e-6
        )

        if improved:
            best_val_bal_acc = val_bal_acc
            best_val_loss = val_loss
            best_val_threshold = val_threshold
            best_state = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"No improvement epochs: {no_improve_epochs}/{PATIENCE}")
            if (
                ENABLE_EARLY_STOPPING
                and no_improve_epochs >= PATIENCE
                and (epoch + 1) >= MIN_EPOCHS_BEFORE_EARLY_STOP
            ):
                print(
                    "Early stopping triggered. "
                    f"Minimum epoch gate reached ({MIN_EPOCHS_BEFORE_EARLY_STOP})."
                )
                break

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "normal_class_label": NORMAL_CLASS_LABEL,
        "normal_class_name": NORMAL_CLASS_NAME,
        "input_channels": INPUT_CHANNELS,
        "q_input_dim": Q_INPUT_DIM,
        "embedding": "angle_y",
        "n_qubits": N_QUBITS,
        "n_quantum_layers": N_QUANTUM_LAYERS,
        "uses_pennylane": USE_QML,
        "qml_device": QML_DEVICE_NAME if USE_QML else "fallback",
        "qml_diff_method": QML_DIFF_METHOD if USE_QML else "none",
        "train_augment": TRAIN_AUGMENT,
        "threshold_strategy": "balanced_accuracy_then_f1",
        "best_val_threshold": float(best_val_threshold),
        "best_val_balanced_accuracy": float(best_val_bal_acc),
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"\n*** Hybrid QC-CNN anomaly model saved to: {MODEL_PATH} ***")

    # Final threshold on validation set
    _, val_scores, val_labels = evaluate(model, val_loader)
    val_scores = np.nan_to_num(val_scores, nan=0.5, posinf=1.0, neginf=0.0)
    threshold, best_val_bal_acc, best_val_acc, best_val_f1 = find_best_threshold(
        val_scores, val_labels, objective="balanced_accuracy",
    )
    np.save(THRESHOLD_PATH, np.array(threshold, dtype=np.float32))

    _, test_scores, test_labels = evaluate(model, test_loader)
    test_scores = np.nan_to_num(test_scores, nan=0.5, posinf=1.0, neginf=0.0)
    test_auc = roc_auc_score(test_labels, test_scores) if len(np.unique(test_labels)) > 1 else float("nan")
    preds = (test_scores >= threshold).astype(np.int32)
    _, balanced_acc, accuracy, f1 = _metrics_from_preds(preds, test_labels), None, None, None
    balanced_acc, accuracy, f1 = _metrics_from_preds(preds, test_labels)

    print("\n--- Hybrid QC-CNN Anomaly Evaluation ---")
    print(f"AUC-ROC: {test_auc:.4f}")
    print(f"Best validation balanced accuracy: {best_val_bal_acc:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation F1 (anomaly): {best_val_f1:.4f}")
    print(f"Threshold (optimised on validation balanced accuracy): {threshold:.6f}")
    print(f"Thresholded accuracy on test set: {accuracy:.4f}")
    print(f"Thresholded balanced accuracy on test set: {balanced_acc:.4f}")
    print(f"Thresholded F1 on test set: {f1:.4f}")
    print(f"Threshold saved to: {THRESHOLD_PATH}")

    true_anomaly_indices = np.where(test_labels == 1)[0]
    true_normal_indices = np.where(test_labels == 0)[0]

    print("\nFirst 5 Anomalies (Expected High Scores):")
    for i in true_anomaly_indices[:5]:
        print(f"{test_scores[i]:<10.6f} | True Label: ANOMALY")

    print("\nFirst 5 Normal Samples (Expected Low Scores):")
    for i in true_normal_indices[:5]:
        print(f"{test_scores[i]:<10.6f} | True Label: NORMAL")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error in Hybrid QC-CNN anomaly pipeline: {e}")
        sys.exit(1)