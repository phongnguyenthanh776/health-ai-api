from __future__ import annotations

from copy import deepcopy
from io import BytesIO
from pathlib import Path
import json
import random
from typing import Iterable, Sequence

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


DEFAULT_IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BinaryImageDataset(Dataset):
    def __init__(self, samples: Sequence[tuple[str | Path, int]], image_size=DEFAULT_IMAGE_SIZE, training: bool = False):
        self.samples = [(str(path), int(label)) for path, label in samples]
        resize_size = tuple(int(v) for v in image_size)
        if training:
            self.transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.RandomRotation(degrees=7),
                transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return tensor, torch.tensor([float(label)], dtype=torch.float32)


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def build_eval_transform(image_size=DEFAULT_IMAGE_SIZE):
    resize_size = tuple(int(v) for v in image_size)
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def initialize_binary_model(architecture: str = "resnet18", pretrained: bool = True):
    architecture = (architecture or "resnet18").strip().lower()
    pretrained_used = False

    if architecture != "resnet18":
        raise ValueError(f"Unsupported architecture: {architecture}")

    weights = None
    if pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT
        except Exception:
            weights = None

    try:
        model = models.resnet18(weights=weights)
        pretrained_used = weights is not None
    except Exception:
        model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, 1)
    return model, "layer4", pretrained_used


def freeze_backbone(model: nn.Module):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def freeze_early_layers(model: nn.Module):
    """Freeze early/mid backbone; fine-tune layer4 and fc.

    This allows the deepest feature maps (layer4) to specialise on
    medical images so that Grad-CAM highlights medically relevant regions
    instead of generic ImageNet texture patterns.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in ("layer4", "fc")):
            parameter.requires_grad = True


def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray, start: float = 0.10, stop: float = 0.90, steps: int = 161):
    best = {
        "threshold": 0.5,
        "accuracy": 0.0,
        "balanced_accuracy": 0.0,
        "f1": 0.0,
    }

    for threshold in np.linspace(start, stop, steps):
        pred = (probabilities >= threshold).astype(np.int32)
        acc = accuracy_score(y_true, pred)
        bacc = balanced_accuracy_score(y_true, pred)
        f1 = f1_score(y_true, pred, zero_division=0)

        if (bacc, f1, acc) > (best["balanced_accuracy"], best["f1"], best["accuracy"]):
            best = {
                "threshold": float(threshold),
                "accuracy": float(acc),
                "balanced_accuracy": float(bacc),
                "f1": float(f1),
            }

    return best


def safe_roc_auc(y_true: np.ndarray, probabilities: np.ndarray):
    try:
        return float(roc_auc_score(y_true, probabilities))
    except Exception:
        return 0.5


def evaluate_binary_classifier(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            probabilities = torch.sigmoid(logits)

            all_probabilities.extend(probabilities.squeeze(1).cpu().numpy().tolist())
            all_labels.extend(labels.squeeze(1).cpu().numpy().astype(np.int32).tolist())

    return np.asarray(all_labels, dtype=np.int32), np.asarray(all_probabilities, dtype=np.float32)


def train_binary_classifier(
    train_samples: Sequence[tuple[str | Path, int]],
    val_samples: Sequence[tuple[str | Path, int]],
    class_names: Sequence[str],
    model_output_path: str | Path,
    labels_output_path: str | Path,
    meta_output_path: str | Path,
    *,
    disease_type: str,
    architecture: str = "resnet18",
    image_size=DEFAULT_IMAGE_SIZE,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    val_source: str | None = None,
):
    if not train_samples or not val_samples:
        raise RuntimeError("Du lieu train/val khong du de train CNN.")

    set_seed(SEED)
    device = get_device()
    model, target_layer_name, pretrained_used = initialize_binary_model(architecture=architecture, pretrained=True)
    feature_extract_mode = False  # Fine-tune layer4/fc for better Grad-CAM spatial attention
    if pretrained_used:
        freeze_early_layers(model)  # Freeze early layers; fine-tune layer4/fc

    model = model.to(device)

    train_dataset = BinaryImageDataset(train_samples, image_size=image_size, training=True)
    val_dataset = BinaryImageDataset(val_samples, image_size=image_size, training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    train_labels = np.asarray([label for _, label in train_samples], dtype=np.int32)
    positive_count = max(1, int((train_labels == 1).sum()))
    negative_count = max(1, int((train_labels == 0).sum()))
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Differential learning rates: fine-tuned backbone uses 10x lower LR than FC head
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc")]
    head_params = list(model.fc.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": learning_rate * 0.1},
            {"params": head_params, "lr": learning_rate},
        ],
        weight_decay=weight_decay,
    )

    best_state = None
    best_metrics = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * inputs.size(0)

        train_loss = running_loss / max(1, len(train_dataset))
        y_val, val_probabilities = evaluate_binary_classifier(model, val_loader, device)
        threshold_metrics = find_best_threshold(y_val, val_probabilities)
        val_auc = safe_roc_auc(y_val, val_probabilities)
        metrics = {
            "epoch": int(epoch + 1),
            "train_loss": float(train_loss),
            "val_accuracy": float(threshold_metrics["accuracy"]),
            "val_balanced_accuracy": float(threshold_metrics["balanced_accuracy"]),
            "val_f1": float(threshold_metrics["f1"]),
            "val_auc": float(val_auc),
            "threshold": float(threshold_metrics["threshold"]),
        }

        print(
            f"Epoch {epoch + 1}/{epochs} - loss={train_loss:.4f} "
            f"bacc={metrics['val_balanced_accuracy']:.4f} f1={metrics['val_f1']:.4f} auc={metrics['val_auc']:.4f}"
        )

        if best_metrics is None or (
            metrics["val_balanced_accuracy"],
            metrics["val_f1"],
            metrics["val_auc"],
            metrics["val_accuracy"],
        ) > (
            best_metrics["val_balanced_accuracy"],
            best_metrics["val_f1"],
            best_metrics["val_auc"],
            best_metrics["val_accuracy"],
        ):
            best_metrics = metrics
            best_state = {name: tensor.detach().cpu() for name, tensor in deepcopy(model.state_dict()).items()}

    if best_state is None or best_metrics is None:
        raise RuntimeError("Khong train duoc CNN model tot nhat.")

    model_output_path = Path(model_output_path)
    labels_output_path = Path(labels_output_path)
    meta_output_path = Path(meta_output_path)

    torch.save(
        {
            "state_dict": best_state,
            "architecture": architecture,
            "target_layer": target_layer_name,
        },
        model_output_path,
    )
    joblib.dump(list(class_names), labels_output_path)

    meta = {
        "classes": list(class_names),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "val_accuracy": float(best_metrics["val_accuracy"]),
        "val_balanced_accuracy": float(best_metrics["val_balanced_accuracy"]),
        "val_f1": float(best_metrics["val_f1"]),
        "val_auc": float(best_metrics["val_auc"]),
        "threshold": float(best_metrics["threshold"]),
        "model_type": "pytorch_cnn_gradcam",
        "model_name": architecture,
        "target_layer": target_layer_name,
        "device": str(device),
        "pretrained": bool(pretrained_used),
        "feature_extract_mode": bool(feature_extract_mode),
        "train_size": int(len(train_samples)),
        "val_size": int(len(val_samples)),
        "disease_type": disease_type,
        "visualization_type": "gradcam",
    }
    if val_source:
        meta["val_source"] = str(val_source)

    meta_output_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_binary_cnn_model(model_path: str | Path, meta: dict, device: torch.device | None = None):
    checkpoint = torch.load(model_path, map_location=(device or get_device()))
    architecture = str(checkpoint.get("architecture") or meta.get("model_name") or "resnet18")
    target_layer_name = str(checkpoint.get("target_layer") or meta.get("target_layer") or "layer4")

    model, _, _ = initialize_binary_model(architecture=architecture, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device or get_device())
    model.eval()
    return model, target_layer_name


def resolve_target_layer(model: nn.Module, target_layer_name: str):
    module = model
    for part in str(target_layer_name).split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def predict_probability_and_gradcam(
    model: nn.Module,
    image_bytes: bytes,
    *,
    image_size=DEFAULT_IMAGE_SIZE,
    target_layer_name: str = "layer4",
    device: torch.device | None = None,
):
    device = device or get_device()
    image = load_image_from_bytes(image_bytes)
    transform = build_eval_transform(image_size=image_size)
    input_tensor = transform(image).unsqueeze(0).to(device)
    target_layer = resolve_target_layer(model, target_layer_name)

    activations = {}
    gradients = {}

    def forward_hook(_, __, output):
        activations["value"] = output.detach()

    def backward_hook(_, __, grad_output):
        gradients["value"] = grad_output[0].detach()

    handles = [
        target_layer.register_forward_hook(forward_hook),
        target_layer.register_full_backward_hook(backward_hook),
    ]

    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        probability = float(torch.sigmoid(logits)[0, 0].detach().cpu().item())
        logits[0, 0].backward()

        activation_map = activations["value"]
        gradient_map = gradients["value"]
        weights = gradient_map.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation_map).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        cam /= max(float(cam.max()), 1e-6)

        return probability, cam.astype(np.float32)
    finally:
        for handle in handles:
            handle.remove()