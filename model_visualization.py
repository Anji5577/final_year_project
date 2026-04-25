from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any

import torch
import torch.nn as nn


QUANTUM_KEYWORDS = (
    "quantum",
    "qubit",
    "qiskit",
    "pennylane",
    "qml",
    "qnode",
    "torchquantum",
    "circuit",
)


@dataclass
class LayerInfo:
    name: str
    layer_type: str
    params: int
    trainable_params: int
    input_shape: str
    output_shape: str


def _shape_str(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, torch.Tensor):
        return str(list(value.shape))
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            if isinstance(item, torch.Tensor):
                parts.append(str(list(item.shape)))
            else:
                parts.append(type(item).__name__)
        return "[" + ", ".join(parts) + "]"
    return type(value).__name__


def _iter_leaf_modules(model: nn.Module):
    for name, module in model.named_modules():
        if name == "":
            continue
        if len(list(module.children())) == 0:
            yield name, module


def _infer_input_shape(model: nn.Module, fallback: tuple[int, ...]) -> tuple[int, ...]:
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return (fallback[0], module.in_channels, fallback[2], fallback[3])
    return fallback


def collect_layer_info(model: nn.Module, device: torch.device, input_shape: tuple[int, ...] = (1, 3, 64, 64)) -> list[LayerInfo]:
    hooks = []
    io_shapes: dict[str, tuple[str, str]] = {}
    resolved_input_shape = _infer_input_shape(model, input_shape)

    def make_hook(module_name: str):
        def _hook(_, inputs, outputs):
            in_shape = _shape_str(inputs[0] if inputs else None)
            out_shape = _shape_str(outputs)
            io_shapes[module_name] = (in_shape, out_shape)

        return _hook

    for name, module in _iter_leaf_modules(model):
        hooks.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(*resolved_input_shape, device=device)
        _ = model(dummy)
    if was_training:
        model.train()

    for hook in hooks:
        hook.remove()

    layers: list[LayerInfo] = []
    for name, module in _iter_leaf_modules(model):
        params = sum(p.numel() for p in module.parameters(recurse=False))
        trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        in_shape, out_shape = io_shapes.get(name, ("-", "-"))
        layers.append(
            LayerInfo(
                name=name,
                layer_type=module.__class__.__name__,
                params=params,
                trainable_params=trainable,
                input_shape=in_shape,
                output_shape=out_shape,
            )
        )
    return layers


def _is_quantum_module(module: nn.Module) -> bool:
    path = f"{module.__class__.__module__}.{module.__class__.__name__}".lower()
    return any(key in path for key in QUANTUM_KEYWORDS)


def collect_quantum_layers(model: nn.Module) -> list[tuple[str, str]]:
    quantum_layers: list[tuple[str, str]] = []
    for name, module in model.named_modules():
        if name == "":
            continue
        if _is_quantum_module(module):
            quantum_layers.append((name, f"{module.__class__.__module__}.{module.__class__.__name__}"))
    return quantum_layers


def collect_quantum_circuits(model: nn.Module) -> list[tuple[str, str]]:
    circuits: list[tuple[str, str]] = []
    for module_name, module in model.named_modules():
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            if not any(k in attr_name.lower() for k in ("circuit", "qnode", "quantum")):
                continue
            try:
                attr_value = getattr(module, attr_name)
            except Exception:
                continue
            if callable(attr_value):
                attr_type = "callable"
            else:
                attr_type = type(attr_value).__name__
            label = f"{module_name or 'model'}.{attr_name}"
            text = repr(attr_value)
            if len(text) > 160:
                text = text[:157] + "..."
            circuits.append((label, f"{attr_type}: {text}"))
    return circuits


def split_layer_groups(layers: list[LayerInfo]) -> tuple[list[LayerInfo], list[LayerInfo], list[LayerInfo]]:
    enc: list[LayerInfo] = []
    dec: list[LayerInfo] = []
    other: list[LayerInfo] = []

    for layer in layers:
        lname = layer.name.lower()
        if lname.startswith("enc") or "encoder" in lname:
            enc.append(layer)
        elif lname.startswith("dec") or "decoder" in lname:
            dec.append(layer)
        else:
            other.append(layer)
    return enc, dec, other


def _render_layer_table(title: str, layers: list[LayerInfo]) -> str:
    if not layers:
        return f"<h3>{escape(title)}</h3><p class='empty'>No layers found.</p>"

    rows = []
    for layer in layers:
        rows.append(
            "<tr>"
            f"<td>{escape(layer.name)}</td>"
            f"<td>{escape(layer.layer_type)}</td>"
            f"<td>{layer.params:,}</td>"
            f"<td>{layer.trainable_params:,}</td>"
            f"<td>{escape(layer.input_shape)}</td>"
            f"<td>{escape(layer.output_shape)}</td>"
            "</tr>"
        )

    return (
        f"<h3>{escape(title)}</h3>"
        "<table>"
        "<thead><tr>"
        "<th>Layer</th><th>Type</th><th>Params</th><th>Trainable</th><th>Input</th><th>Output</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _render_flow(title: str, layers: list[LayerInfo]) -> str:
    if not layers:
        return ""

    cards = []
    for layer in layers:
        cards.append(
            "<div class='node'>"
            f"<div class='node-name'>{escape(layer.name)}</div>"
            f"<div class='node-type'>{escape(layer.layer_type)}</div>"
            "</div>"
        )

    return f"<h3>{escape(title)}</h3><div class='flow'>{''.join(cards)}</div>"


def _render_quantum_block(title: str, values: list[tuple[str, str]]) -> str:
    if not values:
        return f"<h3>{escape(title)}</h3><p class='empty'>No quantum components detected in this model.</p>"

    items = []
    for name, desc in values:
        items.append(f"<li><b>{escape(name)}</b><br><code>{escape(desc)}</code></li>")
    return f"<h3>{escape(title)}</h3><ul class='quantum-list'>{''.join(items)}</ul>"


def render_model_visualization_html(
    model: nn.Module, device: torch.device, input_shape: tuple[int, ...] = (1, 3, 64, 64)
) -> str:
    layers = collect_layer_info(model, device=device, input_shape=input_shape)
    encoder_layers, decoder_layers, other_layers = split_layer_groups(layers)
    quantum_layers = collect_quantum_layers(model)
    quantum_circuits = collect_quantum_circuits(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model Visualization</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #f4f7fb; color: #1f2a37; margin: 0; }}
    .wrap {{ max-width: 1200px; margin: 28px auto; padding: 0 16px 32px; }}
    .card {{ background: #fff; border: 1px solid #dbe2ea; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
    h1 {{ margin-top: 0; }}
    h2 {{ margin-bottom: 6px; }}
    h3 {{ margin: 18px 0 8px; }}
    p {{ margin: 8px 0; }}
    .meta {{ display: flex; gap: 18px; flex-wrap: wrap; }}
    .meta span {{ background: #eef3ff; border: 1px solid #d4e1ff; border-radius: 8px; padding: 8px 10px; }}
    .flow {{ display: flex; gap: 10px; overflow-x: auto; padding: 8px 2px 6px; }}
    .node {{ min-width: 155px; background: #f8fafc; border: 1px solid #d9e2ec; border-radius: 8px; padding: 10px; }}
    .node-name {{ font-weight: 700; font-size: 13px; }}
    .node-type {{ font-size: 12px; color: #4b5563; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #e5eaf0; text-align: left; padding: 8px; vertical-align: top; }}
    th {{ background: #f0f4f8; }}
    .quantum-list {{ margin: 8px 0; padding-left: 20px; }}
    .quantum-list li {{ margin: 10px 0; }}
    code {{ background: #f8fafc; border: 1px solid #d9e2ec; border-radius: 4px; padding: 2px 4px; }}
    .empty {{ color: #6b7280; }}
    a {{ color: #1a73e8; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Model Layer & Quantum Visualization</h1>
      <p>This page is generated automatically from the loaded model object.</p>
      <div class="meta">
        <span><b>Total params:</b> {total_params:,}</span>
        <span><b>Trainable params:</b> {trainable_params:,}</span>
        <span><b>Total leaf layers:</b> {len(layers)}</span>
      </div>
      <p><a href="/">Back to detection page</a></p>
    </div>

    <div class="card">
      <h2>Architecture Flow</h2>
      {_render_flow("Encoder Flow", encoder_layers)}
      {_render_flow("Decoder Flow", decoder_layers)}
      {_render_flow("Other Layer Flow", other_layers)}
    </div>

    <div class="card">
      <h2>Layer Tables</h2>
      {_render_layer_table("Encoder Layers", encoder_layers)}
      {_render_layer_table("Decoder Layers", decoder_layers)}
      {_render_layer_table("Other Layers", other_layers)}
    </div>

    <div class="card">
      <h2>Quantum Components</h2>
      {_render_quantum_block("Quantum Layers", quantum_layers)}
      {_render_quantum_block("Quantum Circuits / QNodes", quantum_circuits)}
    </div>
  </div>
</body>
</html>
"""
