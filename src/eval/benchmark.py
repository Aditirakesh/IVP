"""Parameter count and latency per epoch for any model class.

    python -m src.eval.benchmark --model dummy --channels 32 128
    python -m src.eval.benchmark --model src.models.cdl_mddnet:CDLMDDNet --channels 128 32 --out reports/efficiency.csv

The model class must take (n_channels, n_samples) and eat a (B, C, T) tensor.
"""
import argparse
import csv
import importlib
import os
import statistics
import time

import torch
from torch import nn

LIMIT = 100_000  # deck target, slide 40


class Dummy(nn.Module):
    # stand-in so this runs before the real model is merged
    def __init__(self, n_channels=128, n_samples=1000, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 16, 25, padding=12), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(16, n_classes))

    def forward(self, x):
        return self.net(x)


def load(name, c, ns):
    if name == "dummy":
        return Dummy(c, ns)
    mod, cls = name.split(":")
    return getattr(importlib.import_module(mod), cls)(n_channels=c, n_samples=ns)


def nparams(m):
    # trainable only, same way the paper counts its 34,812
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def latency(m, dev, c, ns, runs=200, warm=20):
    """median ms for one epoch (batch of 1)"""
    m = m.to(dev).eval()
    x = torch.randn(1, c, ns, device=dev)
    for _ in range(warm):
        m(x)
    ts = []
    for _ in range(runs):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        m(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dummy", help="'dummy' or module:Class")
    ap.add_argument("--channels", type=int, nargs="+", default=[128])
    ap.add_argument("--samples", type=int, default=1000)  # 4 s at 250 Hz
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--out", help="csv to append to")
    a = ap.parse_args()

    rows = []
    for c in a.channels:
        m = load(a.model, c, a.samples)
        n = nparams(m)
        cpu = latency(m, "cpu", c, a.samples, a.runs)
        gpu = latency(m, "cuda", c, a.samples, a.runs) if torch.cuda.is_available() else None
        rows.append([a.model, c, n, n < LIMIT, round(cpu, 3),
                     "" if gpu is None else round(gpu, 3), torch.get_num_threads()])
        print(f"{a.model} ch={c} params={n:,} under_100k={n < LIMIT} "
              f"cpu={cpu:.2f}ms gpu={'n/a' if gpu is None else f'{gpu:.2f}ms'}")

    if not a.out:
        return
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "channels", "params", "under_100k", "cpu_ms", "gpu_ms", "cpu_threads"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
