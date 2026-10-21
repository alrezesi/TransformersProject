"""Collect the saved experiment results into one Markdown report."""
import argparse
import datetime
import json
import statistics
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CORE = "attention core (QK^T, softmax, AV)"


def load(path: Path) -> Optional[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def mean_std(values: Iterable[float]):
    values = list(values)
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def pct(x: float) -> str:
    return f"{x * 100:.1f}"


def describe(cfg: dict, keys) -> str:
    return ", ".join(f"{k}={cfg[k]}" for k in keys if k in cfg)


def image_line(tag: str, alt: str) -> list:
    """Embed the figure only if it exists next to the report."""
    return [f"![{alt}]({tag}.png)", ""] if (REPORTS / f"{tag}.png").exists() else []


ANALYSIS = ["**Analysis (write your own):** what did you see, is it larger than the "
            "noise, and why?", ""]


# ------------------------------------------------------------------ experiment 1
def section_heads(data: Optional[dict]) -> str:
    title = "## Experiment 1: number of heads"
    if data is None:
        return f"{title}\n\n_Not run yet (`reports/exp1_heads.json` not found)._\n"
    cfg, summary, runs = data["config"], data["summary"], data["runs"]
    out = [title, "",
           "Setup: " + describe(cfg, ["d_model", "d_ff", "layers", "subset", "epochs",
                                      "lr", "seeds"]) + ".", "",
           "| H | Dh | Params | Test acc % (mean ± std) | Best dev acc % | "
           "Final train loss | s/epoch |",
           "|:-:|:-:|---:|:-:|:-:|:-:|:-:|"]
    for r in summary:
        out.append(f"| {r['heads']} | {r['d_head']} | {r['params']:,} | "
                   f"{pct(r['test_acc_mean'])} ± {pct(r['test_acc_std'])} | "
                   f"{pct(r['dev_acc_mean'])} | {r['train_loss_mean']:.3f} | "
                   f"{r['seconds_per_epoch']:.0f} |")
    out += ["", "_Train loss is measured during training (dropout on)._", ""]

    means = [r["test_acc_mean"] for r in summary]
    gap = max(means) - min(means)
    max_std = max(r["test_acc_std"] for r in summary)
    out.append(f"Noise check: best-to-worst gap {gap * 100:.1f} points, largest std "
               f"across seeds {max_std * 100:.1f} points.")
    if gap <= max_std:
        out.append("The gap is not larger than the seed-to-seed spread, so differences "
                   "between head counts are not established.")
    out.append("With only a few seeds the std itself is a rough estimate.")
    out += ["", "Per-run results:", "", "| H | seed | Test acc % | Best epoch | Final train loss |",
            "|:-:|:-:|:-:|:-:|:-:|"]
    for r in sorted(runs, key=lambda r: (r["heads"], r["seed"])):
        out.append(f"| {r['heads']} | {r['seed']} | {pct(r['test_acc'])} | "
                   f"{r['best_epoch']} | {r['final_train_loss']:.3f} |")
    out.append("")
    return "\n".join(out + image_line(cfg.get("tag", "exp1_heads"), "heads") + ANALYSIS)


# ------------------------------------------------------------------ experiment 2
def _variants(runs):
    for use_pe in (True, False):
        rows = [r for r in runs if r["positional_encoding"] == use_pe]
        if rows:
            yield ("with PE" if use_pe else "without PE"), rows


def section_pe(data: Optional[dict]) -> str:
    title = "## Experiment 2: removing positional encoding"
    if data is None:
        return (f"{title}\n\n_Not run yet "
                f"(`reports/exp2_positional_encoding.json` not found)._\n")
    cfg = data["config"]
    out = [title, "",
           "Setup: " + describe(cfg, ["d_model", "heads", "d_ff", "layers", "subset",
                                      "epochs", "syn_train", "syn_epochs", "seeds"]) + ".", ""]

    out += ["### Synthetic task (label = 1 iff marker A comes before marker B; chance = 50%)", ""]
    if data.get("synthetic"):
        out += ["| Variant | Test acc % (mean ± std) | mean abs(p(original) - p(A,B swapped)) |",
                "|---|:-:|:-:|"]
        for name, rows in _variants(data["synthetic"]):
            m, s = mean_std(r["test_acc"] for r in rows)
            d, _ = mean_std(r["swap_prob_diff"] for r in rows)
            out.append(f"| {name} | {pct(m)} ± {pct(s)} | {d:.2e} |")
        out.append("")
    else:
        out += ["_Not run in this experiment file._", ""]

    out += ["### SST-2", ""]
    if data.get("sst2"):
        out += ["| Variant | Test acc % (mean ± std) | Acc % on word-shuffled test | "
                "Drop (points) |", "|---|:-:|:-:|:-:|"]
        for name, rows in _variants(data["sst2"]):
            m, s = mean_std(r["test_acc"] for r in rows)
            sh, _ = mean_std(r["shuffled_test_acc"] for r in rows)
            out.append(f"| {name} | {pct(m)} ± {pct(s)} | {pct(sh)} | {(m - sh) * 100:.1f} |")
        out += ["", "Without PE the model cannot see word order, so its drop should be "
                "about 0 (up to float noise).", ""]
    else:
        out += ["_Not run in this experiment file._", ""]
    return "\n".join(out + ANALYSIS)


# ------------------------------------------------------------------ experiment 3
def section_seqlen(data: Optional[dict]) -> str:
    title = "## Experiment 3: sequence length and attention cost"
    if data is None:
        return f"{title}\n\n_Not run yet (`reports/exp3_sequence_length.json` not found)._\n"
    cfg = data["config"]
    lengths = cfg["lengths"]
    fwd, train = data["forward_seconds"][CORE], data["train_seconds"][CORE]
    out = [title, "",
           "Setup: " + describe(cfg, ["batch", "d_model", "heads", "repeats"]) + ".", "",
           "| T | Attention matrix (MB) | Core forward (ms) | Core fwd+bwd (ms) | "
           "x vs previous T |", "|---:|---:|---:|---:|---:|"]
    for i, T in enumerate(lengths):
        mb = cfg["batch"] * cfg["heads"] * T * T * 4 / 1024 ** 2
        ratio = "-" if i == 0 else f"{train[i] / train[i - 1]:.2f}"
        out.append(f"| {T} | {mb:.2f} | {fwd[i] * 1e3:.2f} | {train[i] * 1e3:.2f} | {ratio} |")
    out += ["", "_The matrix size is the theoretical size of the attention-weights tensor "
            "(B·H·T²·4 bytes), not measured process memory._", "",
            "Log-log slope of forward+backward time (1 = linear, 2 = quadratic):", "",
            "| Component | All points | Last 3 points |", "|---|:-:|:-:|"]
    for name, s in data["slopes"].items():
        out.append(f"| {name} | {s['all']:.2f} | {s['last3']:.2f} |")
    out.append("")
    return "\n".join(out + image_line(cfg.get("tag", "exp3_sequence_length"), "seq length")
                     + ANALYSIS)


# ------------------------------------------------------------------------ report
def build_report(exp1: Optional[dict], exp2: Optional[dict], exp3: Optional[dict]) -> str:
    parts = [
        "# Experiments report", "",
        f"_Generated on {datetime.date.today().isoformat()} from the JSON files in `reports/`._", "",
        "How to read the numbers:", "",
        "- Experiments 1 and 2 use a smaller model and less data than the full model of "
        "phase 5, so their accuracies are **not comparable** with its 82.1%; compare the "
        "variants only with each other.",
        "- Dev and test have 436 sentences each, so one accuracy has a standard error of "
        "about 2 points. Differences below the seed-to-seed std are not established.",
        "- Timings are CPU measurements on one machine: read growth rates and orders of "
        "magnitude, not absolute values.", "",
        section_heads(exp1), section_pe(exp2), section_seqlen(exp3),
    ]
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp1", default=str(REPORTS / "exp1_heads.json"))
    p.add_argument("--exp2", default=str(REPORTS / "exp2_positional_encoding.json"))
    p.add_argument("--exp3", default=str(REPORTS / "exp3_sequence_length.json"))
    p.add_argument("--out", default=str(REPORTS / "experiments_report.md"))
    args = p.parse_args()

    report = build_report(load(Path(args.exp1)), load(Path(args.exp2)), load(Path(args.exp3)))
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()