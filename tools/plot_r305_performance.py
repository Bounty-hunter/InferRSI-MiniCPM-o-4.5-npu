"""Render the recorded r305 NPU points; no new benchmark or quality claim."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"


def render(rows, variants, fields, title, footnote, name):
    fig, axes = plt.subplots(1, len(fields), figsize=(5 * len(fields), 4.4), layout="constrained")
    x = [row["concurrency"] for row in rows]
    for ax, (key, label) in zip(axes, fields):
        for who, text, color, marker in variants:
            ax.plot(x, [r[who][key] for r in rows], marker=marker, color=color, linewidth=2, label=text)
        ax.set(xscale="log", xticks=x, xticklabels=[f"C{c}" for c in x], ylabel=label)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=15)
    fig.text(.5, -.04, footnote, ha="center", fontsize=8)
    OUT.mkdir(exist_ok=True)
    for suffix in ("png", "svg"):
        kwargs = {"metadata": {"Date": None}} if suffix == "svg" else {}
        path = OUT / f"{name}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight", **kwargs)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                            encoding="utf-8", newline="\n")
    plt.close(fig)


def main():
    data = json.loads((ROOT / "docs/data/ascend-r305-20261102.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "ascend-r305"})
    variants = (("omni", "vLLM-Omni main@9990f10d9", "#D55E00", "s"), ("engine", "Standalone r305 (experimental)", "#0072B2", "o"))
    caveat = "A2 / 910B1 · Omni Flow FP32/10 steps; engine FP16/6 · single-run points · speech gate NOT_RUN\nC1 Omni: same-day archive; engine C1: supplementary run. C2/C4/C8: fresh serial comparison."
    render(data["rows"], variants, (("mean_audio_ttfp_ms", "Mean TTFP (ms)"), ("mean_audio_rtf", "Mean audio RTF"), ("request_throughput", "Requests / s")),
           "Standalone NPU engine vs vLLM-Omni · means", caveat, "npu-r305-vs-omni-means")
    render(data["rows"], variants, (("p99_audio_ttfp_ms", "P99 TTFP (ms)"), ("p99_audio_rtf", "P99 audio RTF")),
           "Standalone NPU engine vs vLLM-Omni · sample P99", caveat, "npu-r305-vs-omni-p99")
    render(data["incremental"], (("control", "r292 control", "#666666", "s"), ("candidate", "r305 serial confirmation", "#0072B2", "o")),
           (("mean_audio_ttfp_ms", "Mean TTFP (ms)"), ("mean_audio_rtf", "Mean audio RTF"), ("request_throughput", "Requests / s")),
           "Profiling-guided optimization · r292 to r305", "Same engine Flow FP16/6, chunk25 · observed before/after optimization, not profiling-only causal gain\nSingle-run native points; earlier control vs later confirmation; speech gate NOT_RUN", "npu-r305-incremental")
    print("Saved r305 mean, P99 and incremental figures as SVG/PNG")


if __name__ == "__main__":
    main()
