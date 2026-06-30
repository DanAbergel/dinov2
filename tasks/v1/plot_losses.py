"""Plot the DINO training losses for the v1 runs (base vs fourier).

Reads runs/v1/<run>/training_metrics.json (DINOv2 writes one JSON record per
logged iteration) and saves a PNG comparing the loss curves of the two runs.

Usage (on Moriah, in an env with matplotlib — e.g. torch_env):
    source /sci/labs/arieljaffe/dan.abergel1/torch_env/bin/activate
    python tasks/v1/plot_losses.py
    -> writes /sci/labs/arieljaffe/dan.abergel1/runs/v1/dino_losses.png
"""

import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = "/sci/labs/arieljaffe/dan.abergel1"
RUNS = ["base", "fourier"]
# DINOv2 loss components logged by train.py (we plot whichever are present).
FIELDS = ["total_loss", "dino_global_crops_loss", "dino_local_crops_loss",
          "ibot_loss", "koleo_loss"]


def load(run):
    """Return (iterations, {field: [values]}) — robust to JSON-array or JSON-lines."""
    path = f"{LAB}/runs/v1/{run}/training_metrics.json"
    txt = open(path).read()
    records = []
    try:                                   # whole file is one JSON (array or object)
        obj = json.loads(txt)
        records = obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:           # JSON-lines: one record per line
        for line in txt.splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    iters, series = [], {f: [] for f in FIELDS}
    for i, r in enumerate(records):
        iters.append(r.get("iteration", r.get("iter", i)))
        for f in FIELDS:
            series[f].append(float(r.get(f, math.nan)))
    return iters, series


def has_data(values):
    return any(v == v for v in values)     # any non-NaN


def main():
    data = {r: load(r) for r in RUNS}
    present = [f for f in FIELDS if any(has_data(data[r][1][f]) for r in RUNS)]
    if not present:
        raise RuntimeError("No known loss fields found in training_metrics.json "
                           f"(looked for {FIELDS}).")

    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    for ax, f in zip(axes[0], present):
        for run in RUNS:
            iters, series = data[run]
            if has_data(series[f]):
                ax.plot(iters, series[f], label=run, linewidth=1.3)
        ax.set_title(f)
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("v1 DINO losses — base vs fourier", fontsize=13)
    fig.tight_layout()
    out = f"{LAB}/runs/v1/dino_losses.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    # also print the final values for a quick text summary
    for run in RUNS:
        _, series = data[run]
        finals = {f: series[f][-1] for f in present if has_data(series[f])}
        print(f"  {run} final: " + "  ".join(f"{k}={v:.3f}" for k, v in finals.items()))


if __name__ == "__main__":
    main()
