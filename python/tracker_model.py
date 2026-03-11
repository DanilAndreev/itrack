import os
import time
from collections import OrderedDict

import cv2  # type: ignore[reportMissingImports]
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # type: ignore[reportMissingImports]

# Hardcoded tracker settings from siamrpn_alex_dwxcorr.yaml
EXEMPLAR_SIZE = 127
INSTANCE_SIZE = 666
BASE_SIZE = 0
ANCHOR_STRIDE = 8
ANCHOR_RATIOS = (0.33, 0.5, 1.0, 2.0, 3.0)
ANCHOR_SCALES = (8,)

# Central debug-runner config (edit directly in code)
RUN_MODE = "inspect_weights"  # full_track | backbone_only | rpn_only | export_parts | inspect_weights
DEBUG_LEVEL = "verbose"  # compact | verbose
CHECKPOINT_PATH = "python/model.pth"  # Example: "model.pth"
BACKBONE_SAVE_PATH = "python/backbone_weights.pth"
RPN_SAVE_PATH = "python/rpn_head_weights.pth"
TEMPLATE_IMAGE_PATH = "dataset/template.png"
SEARCH_IMAGE_PATH = "dataset/search.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_RUNS = 10
BENCHMARK_RUNS = 100
REPORT_KEY_LIMIT = 20
WEIGHT_EDGE_VALUES = 3
OUTPUT_EDGE_VALUES = 10


def xcorr_depthwise(x, kernel):
    """Depthwise cross-correlation used in SiamRPN."""
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class AlexNetLegacy(nn.Module):
    def __init__(self):
        super().__init__()
        # Hardcoded from config: width_mult = 1.0
        c0, c1, c2, c3, c4, c5 = 3, 96, 256, 384, 384, 256
        self.features = nn.Sequential(
            nn.Conv2d(c0, c1, kernel_size=11, stride=2),
            nn.BatchNorm2d(c1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=5),
            nn.BatchNorm2d(c2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, kernel_size=3),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, c5, kernel_size=3),
            nn.BatchNorm2d(c5),
        )

    def forward(self, x):
        return self.features(x)


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super().__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        return self.head(feature)


class DepthwiseRPN(nn.Module):
    def __init__(self, anchor_num, in_channels, out_channels):
        super().__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc


class TrackerModel(nn.Module):
    """
    Standalone SiamRPN model with hardcoded settings:
    - backbone: AlexNetLegacy
    - rpn: DepthwiseRPN
    - width_mult: 1.0
    - anchor_num: 5
    - in_channels: 256
    - out_channels: 256
    """

    def __init__(self):
        super().__init__()
        self.backbone = AlexNetLegacy()
        self.rpn_head = DepthwiseRPN(
            anchor_num=5,
            in_channels=256,
            out_channels=256,
        )
        self.zf = None

    def forward(self, template, search):
        z_f = self.backbone(template)
        x_f = self.backbone(search)
        cls, loc = self.rpn_head(z_f, x_f)
        return {"cls": cls, "loc": loc}

    def template(self, template):
        self.zf = self.backbone(template)

    def track(self, search):
        if self.zf is None:
            raise RuntimeError("Call template() before track().")
        x_f = self.backbone(search)
        cls, loc = self.rpn_head(self.zf, x_f)
        return {"cls": cls, "loc": loc}


def load_image_as_tensor(image_path, size_hw, device):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(
        image,
        (size_hw[1], size_hw[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    tensor = torch.from_numpy(np.asarray(image)).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(device)


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_benchmark(fn, warmup, runs, device, desc):
    for _ in tqdm(range(warmup), desc=f"{desc} warmup", unit="iter"):
        fn()
    sync_if_cuda(device)

    run_times_ms = []
    last_output = None
    for _ in tqdm(range(runs), desc=f"{desc} benchmark", unit="iter"):
        sync_if_cuda(device)
        start = time.perf_counter()
        last_output = fn()
        sync_if_cuda(device)
        run_times_ms.append((time.perf_counter() - start) * 1000.0)

    times = np.asarray(run_times_ms, dtype=np.float64)
    stats = {
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "avg_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
    }
    return last_output, stats


def tensor_stats(tensor):
    arr = tensor.detach().float().cpu()
    return {
        "shape": tuple(arr.shape),
        "min": float(arr.min().item()),
        "max": float(arr.max().item()),
        "mean": float(arr.mean().item()),
        "std": float(arr.std(unbiased=False).item()),
    }


def tensor_edges(tensor, edge_values):
    arr = tensor.detach().flatten().float().cpu()
    head = arr[:edge_values].tolist()
    tail = arr[-edge_values:].tolist()
    return head, tail


def print_tensor_observability(name, tensor, debug_level):
    stats = tensor_stats(tensor)
    head, tail = tensor_edges(tensor, OUTPUT_EDGE_VALUES)
    if debug_level == "compact":
        print(f"[compact] {name} shape={stats['shape']}, head={[f"{x:.2e}" for x in head]}, tail={[f"{x:.2e}" for x in tail]}")
        return
    print(
        f"[verbose] {name} shape={stats['shape']}, "
        f"min={stats['min']:.6f}, max={stats['max']:.6f}, "
        f"mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
        f"head={head}, tail={tail}"
    )


def print_layer_weight_edges(module, edge_values, debug_level):
    if debug_level != "verbose":
        return
    for name, param in module.named_parameters():
        if param.numel() == 0:
            continue
        flat = param.detach().flatten().float().cpu()
        head = flat[:edge_values].tolist()
        tail = flat[-edge_values:].tolist()
        print(
            f"[verbose] weight={name}, shape={tuple(param.shape)}, "
            f"head={head}, tail={tail}"
        )
        # os.makedirs("weights", exist_ok=True)
        # dump_float_list_to_file_binary(os.path.join("weights", f"{name}.bytes"), flat)
        # with open(os.path.join("weights", f"{name}.meta.txt"), "w") as f:
        #     f.write(f"{tuple(param.shape)}")


def _collect_output_tensors(value, prefix):
    if isinstance(value, torch.Tensor):
        return [(prefix, value)]
    if isinstance(value, (list, tuple)):
        items = []
        for idx, item in enumerate(value):
            items.extend(_collect_output_tensors(item, f"{prefix}[{idx}]"))
        return items
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            items.extend(_collect_output_tensors(item, f"{prefix}.{key}"))
        return items
    return []


def _pretty_layer_path(raw_path):
    parts = raw_path.split(".")
    pretty_parts = []
    for part in parts:
        if part.isdigit() and pretty_parts:
            pretty_parts[-1] = f"{pretty_parts[-1]}[{part}]"
        else:
            pretty_parts.append(part)
    return ".".join(pretty_parts)


def _layer_display_name(module_label, sub_name, sub_module):
    pretty_sub_name = _pretty_layer_path(sub_name)
    layer_type = sub_module.__class__.__name__
    return f"{module_label} > {pretty_sub_name} ({layer_type})"


def run_layer_output_observability(run_fn, modules_to_hook, section_title, debug_level):
    records = []
    call_counts = {}
    handles = []

    for module_label, module in modules_to_hook:
        for sub_name, sub_module in module.named_modules():
            # Log every leaf layer output (including non-param layers like ReLU/Pool)
            # to keep layer indices contiguous and easy to follow.
            if sub_name == "":
                continue
            is_leaf = len(list(sub_module.children())) == 0
            if not is_leaf:
                continue
            full_name = _layer_display_name(module_label, sub_name, sub_module)

            def _hook(_module, _inputs, output, layer_name=full_name):
                call_index = call_counts.get(layer_name, 0) + 1
                call_counts[layer_name] = call_index
                output_tensors = _collect_output_tensors(output, prefix=layer_name)
                for output_name, output_tensor in output_tensors:
                    if not isinstance(output_tensor, torch.Tensor):
                        continue
                    records.append((f"{output_name}#call{call_index}", output_tensor))

            handles.append(sub_module.register_forward_hook(_hook))

    try:
        run_fn()
    finally:
        for handle in handles:
            handle.remove()

    print(f"Layer output observability [{section_title}]")
    if not records:
        print("  No layer outputs captured.")
        return

    for layer_name, tensor in records:
        print_tensor_observability(layer_name, tensor, debug_level)


def normalize_checkpoint_keys(state_dict):
    prefix_variants = (
        "module.",
        "model.",
        "state_dict.",
        "net.",
        "tracker.",
    )
    normalized = OrderedDict()
    key_map = {}
    for original_key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        candidate = original_key
        changed = True
        while changed:
            changed = False
            for prefix in prefix_variants:
                if candidate.startswith(prefix):
                    candidate = candidate[len(prefix) :]
                    changed = True
        if candidate.startswith("features."):
            candidate = f"backbone.{candidate}"
        if candidate.startswith("rpn."):
            candidate = candidate.replace("rpn.", "rpn_head.", 1)
        normalized[candidate] = value
        key_map[original_key] = candidate
    return normalized, key_map


def select_state_dict_from_checkpoint(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            candidate = checkpoint_obj.get(key)
            if isinstance(candidate, dict):
                return candidate, key
        if all(isinstance(v, torch.Tensor) for v in checkpoint_obj.values()):
            return checkpoint_obj, "state_dict_root"
    raise ValueError("Unsupported checkpoint format. Expected dict-like state_dict.")


def report_load_result(report, limit):
    print(f"Checkpoint: {report['checkpoint_path']}")
    print(f"Detected format: {report['format']}")
    print(
        "Key stats: "
        f"raw={report['raw_keys']}, normalized={report['normalized_keys']}, "
        f"matched={report['matched']}, missing={report['missing']}, "
        f"unexpected={report['unexpected']}, shape_mismatch={report['shape_mismatch']}"
    )
    if report["missing_keys"]:
        print(f"Missing keys (first {limit}): {report['missing_keys'][:limit]}")
    if report["unexpected_keys"]:
        print(f"Unexpected keys (first {limit}): {report['unexpected_keys'][:limit]}")
    if report["shape_mismatch_details"]:
        print(
            f"Shape mismatch keys (first {limit}): "
            f"{report['shape_mismatch_details'][:limit]}"
        )


def load_checkpoint_auto(model, checkpoint_path, device, report_key_limit):
    if not checkpoint_path:
        print("Checkpoint loading skipped: CHECKPOINT_PATH is not set.")
        return None
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw_checkpoint = torch.load(checkpoint_path, map_location=device)
    raw_state_dict, detected_format = select_state_dict_from_checkpoint(raw_checkpoint)
    normalized_state, key_map = normalize_checkpoint_keys(raw_state_dict)

    model_state = model.state_dict()
    filtered = OrderedDict()
    missing_keys = []
    shape_mismatch = []
    used_keys = set()

    for model_key, model_tensor in model_state.items():
        source_tensor = normalized_state.get(model_key)
        if source_tensor is None:
            missing_keys.append(model_key)
            continue
        if tuple(source_tensor.shape) != tuple(model_tensor.shape):
            shape_mismatch.append(
                {
                    "key": model_key,
                    "checkpoint_shape": tuple(source_tensor.shape),
                    "model_shape": tuple(model_tensor.shape),
                }
            )
            continue
        filtered[model_key] = source_tensor
        used_keys.add(model_key)

    unexpected_keys = [k for k in normalized_state.keys() if k not in used_keys]
    load_result = model.load_state_dict(filtered, strict=False)

    report = {
        "checkpoint_path": checkpoint_path,
        "format": detected_format,
        "raw_keys": len(raw_state_dict),
        "normalized_keys": len(normalized_state),
        "matched": len(filtered),
        "missing": len(missing_keys),
        "unexpected": len(unexpected_keys),
        "shape_mismatch": len(shape_mismatch),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatch_details": shape_mismatch,
        "torch_missing_after_load": list(load_result.missing_keys),
        "torch_unexpected_after_load": list(load_result.unexpected_keys),
        "key_map_size": len(key_map),
    }
    report_load_result(report, report_key_limit)
    return report


def export_backbone_and_rpn(model, backbone_save_path, rpn_save_path, device):
    torch.save(model.backbone.state_dict(), backbone_save_path)
    torch.save(model.rpn_head.state_dict(), rpn_save_path)
    print(f"Saved backbone weights: {backbone_save_path}")
    print(f"Saved rpn_head weights: {rpn_save_path}")

    fresh = TrackerModel().to(device)
    backbone_state = torch.load(backbone_save_path, map_location=device)
    rpn_state = torch.load(rpn_save_path, map_location=device)
    backbone_result = fresh.backbone.load_state_dict(backbone_state, strict=True)
    rpn_result = fresh.rpn_head.load_state_dict(rpn_state, strict=True)
    print(
        "Export validation: "
        f"backbone_missing={len(backbone_result.missing_keys)}, "
        f"backbone_unexpected={len(backbone_result.unexpected_keys)}, "
        f"rpn_missing={len(rpn_result.missing_keys)}, "
        f"rpn_unexpected={len(rpn_result.unexpected_keys)}"
    )
    return {
        "backbone_path": backbone_save_path,
        "rpn_path": rpn_save_path,
        "backbone_missing": len(backbone_result.missing_keys),
        "backbone_unexpected": len(backbone_result.unexpected_keys),
        "rpn_missing": len(rpn_result.missing_keys),
        "rpn_unexpected": len(rpn_result.unexpected_keys),
    }


def run_full_track(model, template_tensor, search_tensor, warmup, runs, device, debug_level):
    model.eval()
    with torch.no_grad():
        model.template(template_tensor)
        output, timing_stats = timed_benchmark(
            lambda: model.track(search_tensor),
            warmup=warmup,
            runs=runs,
            device=device,
            desc="full_track",
        )
        run_layer_output_observability(
            run_fn=lambda: model.template(template_tensor),
            modules_to_hook=[("full_track.template.backbone", model.backbone)],
            section_title="full_track template",
            debug_level=debug_level,
        )
        run_layer_output_observability(
            run_fn=lambda: model.track(search_tensor),
            modules_to_hook=[
                ("full_track.track.backbone", model.backbone),
                ("full_track.track.rpn_head", model.rpn_head),
            ],
            section_title="full_track track",
            debug_level=debug_level,
        )
    print_tensor_observability("full_track.cls", output["cls"], debug_level)
    print_tensor_observability("full_track.loc", output["loc"], debug_level)
    return {"output": output, "timing": timing_stats}


def run_backbone_only(model, template_tensor, search_tensor, warmup, runs, device, debug_level):
    model.eval()
    with torch.no_grad():
        template_feature, template_stats = timed_benchmark(
            lambda: model.backbone(template_tensor),
            warmup=warmup,
            runs=runs,
            device=device,
            desc="backbone(template)",
        )
        search_feature, search_stats = timed_benchmark(
            lambda: model.backbone(search_tensor),
            warmup=warmup,
            runs=runs,
            device=device,
            desc="backbone(search)",
        )
        run_layer_output_observability(
            run_fn=lambda: model.backbone(template_tensor),
            modules_to_hook=[("backbone.template", model.backbone)],
            section_title="backbone template",
            debug_level=debug_level,
        )
        run_layer_output_observability(
            run_fn=lambda: model.backbone(search_tensor),
            modules_to_hook=[("backbone.search", model.backbone)],
            section_title="backbone search",
            debug_level=debug_level,
        )
    print_tensor_observability("backbone.template_feature", template_feature, debug_level)
    print_tensor_observability("backbone.search_feature", search_feature, debug_level)
    return {
        "zf": template_feature,
        "xf": search_feature,
        "timing_template": template_stats,
        "timing_search": search_stats,
    }


def run_rpn_only(model, zf, xf, warmup, runs, device, debug_level):
    model.eval()
    with torch.no_grad():
        output, timing_stats = timed_benchmark(
            lambda: model.rpn_head(zf, xf),
            warmup=warmup,
            runs=runs,
            device=device,
            desc="rpn_only",
        )
        run_layer_output_observability(
            run_fn=lambda: model.rpn_head(zf, xf),
            modules_to_hook=[("rpn_only", model.rpn_head)],
            section_title="rpn_only",
            debug_level=debug_level,
        )
    cls, loc = output
    print_tensor_observability("rpn_only.cls", cls, debug_level)
    print_tensor_observability("rpn_only.loc", loc, debug_level)
    return {"cls": cls, "loc": loc, "timing": timing_stats}


def run_inspect_weights(model, debug_level):
    print("Inspecting model weights...")
    print_layer_weight_edges(model.backbone, WEIGHT_EDGE_VALUES, debug_level)
    print_layer_weight_edges(model.rpn_head, WEIGHT_EDGE_VALUES, debug_level)


def calc_runtime_shape_stats(instance_size, exemplar_size, stride, base_size, anchor_num):
    score_size = (instance_size - exemplar_size) // stride + 1 + base_size
    candidates = score_size * score_size * anchor_num
    return {
        "score_size": score_size,
        "positions": score_size * score_size,
        "anchor_num": anchor_num,
        "candidates": candidates,
    }


def main():
    if RUN_MODE not in {"full_track", "backbone_only", "rpn_only", "export_parts", "inspect_weights"}:
        raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE}")
    if DEBUG_LEVEL not in {"compact", "verbose"}:
        raise ValueError(f"Unsupported DEBUG_LEVEL: {DEBUG_LEVEL}")

    device = torch.device(DEVICE)
    anchor_num = len(ANCHOR_RATIOS) * len(ANCHOR_SCALES)
    stats = calc_runtime_shape_stats(
        instance_size=INSTANCE_SIZE,
        exemplar_size=EXEMPLAR_SIZE,
        stride=ANCHOR_STRIDE,
        base_size=BASE_SIZE,
        anchor_num=anchor_num,
    )
    model = TrackerModel().to(device)
    _ = load_checkpoint_auto(
        model=model,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
        report_key_limit=REPORT_KEY_LIMIT,
    )

    print(f"Device: {device}")
    print(f"RUN_MODE={RUN_MODE}, DEBUG_LEVEL={DEBUG_LEVEL}")
    print(
        "Tracking geometry: "
        f"instance={INSTANCE_SIZE}, exemplar={EXEMPLAR_SIZE}, stride={ANCHOR_STRIDE}, "
        f"base={BASE_SIZE}, anchor_num={anchor_num}"
    )
    print(
        "Derived workload: "
        f"score_size={stats['score_size']}x{stats['score_size']}, "
        f"positions={stats['positions']}, candidates={stats['candidates']}"
    )

    if RUN_MODE == "inspect_weights":
        run_inspect_weights(model, DEBUG_LEVEL)
        return

    template = load_image_as_tensor(
        TEMPLATE_IMAGE_PATH, size_hw=(EXEMPLAR_SIZE, EXEMPLAR_SIZE), device=device
    )
    search = load_image_as_tensor(
        SEARCH_IMAGE_PATH, size_hw=(INSTANCE_SIZE, INSTANCE_SIZE), device=device
    )
    print_tensor_observability("input.template", template, DEBUG_LEVEL)
    print_tensor_observability("input.search", search, DEBUG_LEVEL)

    if RUN_MODE == "full_track":
        result = run_full_track(
            model=model,
            template_tensor=template,
            search_tensor=search,
            warmup=WARMUP_RUNS,
            runs=BENCHMARK_RUNS,
            device=device,
            debug_level=DEBUG_LEVEL,
        )
        output = result["output"]
        timing_stats = result["timing"]
        print(f"Output cls shape: {tuple(output['cls'].shape)}")
        print(f"Output loc shape: {tuple(output['loc'].shape)}")
        print(f"Timing min:    {timing_stats['min_ms']:.3f} ms")
        print(f"Timing max:    {timing_stats['max_ms']:.3f} ms")
        print(f"Timing avg:    {timing_stats['avg_ms']:.3f} ms")
        print(f"Timing median: {timing_stats['median_ms']:.3f} ms")
        print(f"(runs={BENCHMARK_RUNS}, warmup={WARMUP_RUNS})")
        return

    backbone_result = run_backbone_only(
        model=model,
        template_tensor=template,
        search_tensor=search,
        warmup=WARMUP_RUNS,
        runs=BENCHMARK_RUNS,
        device=device,
        debug_level=DEBUG_LEVEL,
    )
    print(
        "Backbone timing template(ms): "
        f"{backbone_result['timing_template']['avg_ms']:.3f} avg / "
        f"{backbone_result['timing_template']['median_ms']:.3f} median"
    )
    print(
        "Backbone timing search(ms): "
        f"{backbone_result['timing_search']['avg_ms']:.3f} avg / "
        f"{backbone_result['timing_search']['median_ms']:.3f} median"
    )

    if RUN_MODE == "backbone_only":
        return

    if RUN_MODE == "export_parts":
        export_backbone_and_rpn(
            model=model,
            backbone_save_path=BACKBONE_SAVE_PATH,
            rpn_save_path=RPN_SAVE_PATH,
            device=device,
        )

    if RUN_MODE in {"rpn_only", "export_parts"}:
        rpn_result = run_rpn_only(
            model=model,
            zf=backbone_result["zf"],
            xf=backbone_result["xf"],
            warmup=WARMUP_RUNS,
            runs=BENCHMARK_RUNS,
            device=device,
            debug_level=DEBUG_LEVEL,
        )
        print(f"RPN cls shape: {tuple(rpn_result['cls'].shape)}")
        print(f"RPN loc shape: {tuple(rpn_result['loc'].shape)}")
        print(f"RPN timing avg: {rpn_result['timing']['avg_ms']:.3f} ms")
        print(f"RPN timing median: {rpn_result['timing']['median_ms']:.3f} ms")


if __name__ == "__main__":
    main()
