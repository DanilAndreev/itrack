import time

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

# Runtime settings (edit these directly in code)
TEMPLATE_IMAGE_PATH = "../dataset/template.png"
SEARCH_IMAGE_PATH = "../dataset/search.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_RUNS = 10
BENCHMARK_RUNS = 666


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


def benchmark(model, template_tensor, search_tensor, warmup, runs, device):
    model.eval()
    per_run_times_ms = []
    with torch.no_grad():
        model.template(template_tensor)

        for _ in tqdm(range(warmup), desc="Warmup", unit="iter"):
            _ = model.track(search_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        for _ in tqdm(range(runs), desc="Benchmark", unit="iter"):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = model.track(search_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            per_run_times_ms.append(elapsed_ms)

    times = np.array(per_run_times_ms, dtype=np.float64)
    timing_stats = {
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "avg_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
    }
    return output, timing_stats


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
    device = torch.device(DEVICE)

    anchor_num = len(ANCHOR_RATIOS) * len(ANCHOR_SCALES)
    stats = calc_runtime_shape_stats(
        instance_size=INSTANCE_SIZE,
        exemplar_size=EXEMPLAR_SIZE,
        stride=ANCHOR_STRIDE,
        base_size=BASE_SIZE,
        anchor_num=anchor_num,
    )

    template = load_image_as_tensor(
        TEMPLATE_IMAGE_PATH, size_hw=(EXEMPLAR_SIZE, EXEMPLAR_SIZE), device=device
    )
    search = load_image_as_tensor(
        SEARCH_IMAGE_PATH, size_hw=(INSTANCE_SIZE, INSTANCE_SIZE), device=device
    )

    model = TrackerModel().to(device)
    output, timing_stats = benchmark(
        model=model,
        template_tensor=template,
        search_tensor=search,
        warmup=WARMUP_RUNS,
        runs=BENCHMARK_RUNS,
        device=device,
    )

    cls_shape = tuple(output["cls"].shape)
    loc_shape = tuple(output["loc"].shape)
    print(f"Device: {device}")
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
    print(f"Output cls shape: {cls_shape}")
    print(f"Output loc shape: {loc_shape}")
    print(f"Timing min:    {timing_stats['min_ms']:.3f} ms")
    print(f"Timing max:    {timing_stats['max_ms']:.3f} ms")
    print(f"Timing avg:    {timing_stats['avg_ms']:.3f} ms")
    print(f"Timing median: {timing_stats['median_ms']:.3f} ms")
    print(f"(runs={BENCHMARK_RUNS}, warmup={WARMUP_RUNS})")


if __name__ == "__main__":
    main()
