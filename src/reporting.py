"""
Reporting utilities: publication-ready plots and CSV tables.

Generates:
  plots/
    auroc_comparison.png       — bar chart: AUROC per model
    pr_curve.png               — PR curves: main model vs best baseline
    calibration_curve.png      — reliability diagram: predicted vs observed
    domain_generalization.png  — in-dist vs OOD AUROC bar chart
    ablation_study.png         — AUROC per ablation variant
  tables/
    main_results_table.csv     — full model + baselines metrics
    ablation_table.csv         — clean ablation names
    baseline_table.csv         — baselines only
  metrics.csv                  — flat per-model metric rows
"""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from utils import get_logger

logger = get_logger("synthica.reporting")

# ---------------------------------------------------------------------------
# Ablation canonical name mapping
# ---------------------------------------------------------------------------

ABLATION_DISPLAY_NAMES: Dict[str, str] = {
    "no_hosp_adv":         "w/o Hospital Adversary",
    "no_treat_adv":        "w/o Treatment Adversary",
    "no_contrastive":      "w/o Contrastive",
    "no_irm":              "w/o IRM",
    "no_adv_heads":        "w/o Adversarial",
    "no_s_t_et_only":      "e_t only (no s_t)",
    "full_model":          "Full Model",
}


def _display_name(raw: str) -> str:
    return ABLATION_DISPLAY_NAMES.get(raw, raw.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Matplotlib helper — import lazily so import errors are surfaced clearly
# ---------------------------------------------------------------------------

def _get_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        logger.warning("matplotlib not available — plots will be skipped: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Plot 1: AUROC comparison bar chart
# ---------------------------------------------------------------------------

def plot_auroc_comparison(
    model_auroc: float,
    baseline_results: Dict[str, Dict],
    ablation_results: Dict[str, Dict],
    output_path: str,
) -> None:
    plt = _get_matplotlib()
    if plt is None:
        return

    labels: List[str] = ["Full Model"]
    values: List[float] = [model_auroc]
    colors: List[str] = ["#2196F3"]

    for name, res in sorted(baseline_results.items(), key=lambda x: -x[1].get("auroc", 0.0)):
        if isinstance(res, dict) and "auroc" in res:
            labels.append(_display_name(name))
            values.append(res["auroc"])
            colors.append("#9E9E9E")

    for name, res in sorted(ablation_results.items(), key=lambda x: -x[1].get("auroc", 0.0)):
        if isinstance(res, dict) and "auroc" in res:
            labels.append(_display_name(name))
            values.append(res["auroc"])
            colors.append("#FF9800")

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC Comparison — All Models")
    ax.set_ylim(max(0, min(values) - 0.05), min(1.0, max(values) + 0.05))
    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=0.8, label="Random baseline")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    Patch = plt.matplotlib.patches.Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="Main model"),
        Patch(facecolor="#9E9E9E", label="Baseline"),
        Patch(facecolor="#FF9800", label="Ablation"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved → %s", output_path)


# ---------------------------------------------------------------------------
# Plot 2: Precision-Recall curve
# ---------------------------------------------------------------------------

def plot_pr_curve(
    main_y_true: np.ndarray,
    main_y_score: np.ndarray,
    baseline_y_true: np.ndarray,
    baseline_y_score: np.ndarray,
    baseline_name: str,
    output_path: str,
) -> None:
    plt = _get_matplotlib()
    if plt is None:
        return

    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score
    except ImportError:
        logger.warning("sklearn not available — PR curve skipped.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    for y_t, y_s, name, color in [
        (main_y_true, main_y_score, "Full Model", "#2196F3"),
        (baseline_y_true, baseline_y_score, baseline_name, "#9E9E9E"),
    ]:
        if len(np.unique(y_t)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_t, y_s)
        ap = average_precision_score(y_t, y_s)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=color, linewidth=1.5)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved → %s", output_path)


# ---------------------------------------------------------------------------
# Plot 3: Calibration reliability diagram
# ---------------------------------------------------------------------------

def plot_calibration_curve(
    ece_json_paths: Dict[str, str],
    output_path: str,
) -> None:
    plt = _get_matplotlib()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    colors = ["#2196F3", "#9E9E9E", "#FF9800", "#4CAF50", "#F44336"]
    for i, (name, path) in enumerate(ece_json_paths.items()):
        if not os.path.exists(path):
            continue
        try:
            with open(path) as fh:
                data = json.load(fh)
            ece = data.get("ece", 0.0)
            bins = data.get("bins", [])
            conf_vals = [b["avg_conf"] for b in bins if b.get("avg_conf") is not None]
            acc_vals = [b["avg_acc"] for b in bins if b.get("avg_acc") is not None]
            if conf_vals and acc_vals:
                color = colors[i % len(colors)]
                ax.plot(conf_vals, acc_vals, "o-", color=color,
                        label=f"{_display_name(name)} (ECE={ece:.3f})", linewidth=1.5, markersize=4)
        except Exception as ex:
            logger.debug("Calibration plot: could not load %s: %s", path, ex)

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Reliability Diagram")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved → %s", output_path)


# ---------------------------------------------------------------------------
# Plot 4: Domain generalization (in-dist vs OOD AUROC)
# ---------------------------------------------------------------------------

def plot_domain_generalization(
    domain_gen_metrics: Dict[str, Any],
    output_path: str,
) -> None:
    plt = _get_matplotlib()
    if plt is None:
        return

    models_data = domain_gen_metrics.get("models", {})
    if not models_data:
        logger.warning("No domain generalization data — skipping plot.")
        return

    names = list(models_data.keys())
    in_vals = [models_data[n]["in_dist_auroc"] for n in names]
    ood_vals = [models_data[n]["ood_auroc"] for n in names]
    display = [_display_name(n) for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9), 5))

    bars1 = ax.bar(x - width / 2, in_vals, width, label="In-distribution", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, ood_vals, width, label="Out-of-hospital", color="#FF5722", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_title("Domain Generalization: In-Distribution vs OOD AUROC")
    ax.legend(fontsize=9)
    ax.set_ylim(max(0, min(ood_vals + in_vals) - 0.05), min(1.0, max(in_vals) + 0.05))
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in list(zip(bars1, in_vals)) + list(zip(bars2, ood_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved → %s", output_path)


# ---------------------------------------------------------------------------
# Plot 5: Ablation study
# ---------------------------------------------------------------------------

def plot_ablation_study(
    full_model_auroc: float,
    ablation_results: Dict[str, Dict],
    output_path: str,
) -> None:
    plt = _get_matplotlib()
    if plt is None:
        return

    # Canonical ablation order for paper
    ordered_keys = [
        "no_adv_heads", "no_hosp_adv", "no_treat_adv",
        "no_contrastive", "no_irm", "no_s_t_et_only",
    ]
    labels: List[str] = ["Full Model"]
    values: List[float] = [full_model_auroc]
    colors: List[str] = ["#2196F3"]

    for key in ordered_keys:
        if key in ablation_results and "auroc" in ablation_results[key]:
            labels.append(_display_name(key))
            values.append(ablation_results[key]["auroc"])
            colors.append("#FF9800")

    # Any extra ablations not in canonical list
    for key, res in ablation_results.items():
        if key not in ordered_keys and isinstance(res, dict) and "auroc" in res:
            labels.append(_display_name(key))
            values.append(res["auroc"])
            colors.append("#FF9800")

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.8), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5)

    ax.axhline(y=full_model_auroc, color="#2196F3", linestyle="--",
               linewidth=1.0, label="Full Model AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_title("Ablation Study")
    ax.set_ylim(max(0, min(values) - 0.05), min(1.0, max(values) + 0.05))
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved → %s", output_path)


# ---------------------------------------------------------------------------
# CSV table generation
# ---------------------------------------------------------------------------

def _safe(v: Any, decimals: int = 4) -> str:
    if v is None or v == "":
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        logger.warning("No rows for CSV %s — skipping.", path)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Table saved → %s", path)


def generate_main_results_table(
    model_auroc: float,
    model_auprc: float,
    model_ece: float,
    model_in_dist: float,
    model_ood: Optional[float],
    model_gap: Optional[float],
    model_mean_auroc: Optional[float],
    model_std_auroc: Optional[float],
    baseline_results: Dict[str, Dict],
    output_path: str,
) -> None:
    rows: List[Dict[str, Any]] = []

    # Main model row
    rows.append({
        "Model": "Full Model (ours)",
        "AUROC": _safe(model_auroc),
        "AUPRC": _safe(model_auprc),
        "ECE": _safe(model_ece),
        "In-Dist AUROC": _safe(model_in_dist),
        "OOD AUROC": _safe(model_ood),
        "Gen Gap": _safe(model_gap),
        "Mean AUROC (3-seed)": _safe(model_mean_auroc),
        "Std AUROC (3-seed)": _safe(model_std_auroc),
    })

    for name, res in sorted(baseline_results.items()):
        if not isinstance(res, dict) or "auroc" not in res:
            continue
        rows.append({
            "Model": _display_name(name),
            "AUROC": _safe(res.get("auroc")),
            "AUPRC": _safe(res.get("auprc")),
            "ECE": _safe(res.get("ece")),
            "In-Dist AUROC": _safe(res.get("auroc")),
            "OOD AUROC": _safe(res.get("ooh_auroc")),
            "Gen Gap": "N/A",
            "Mean AUROC (3-seed)": "N/A",
            "Std AUROC (3-seed)": "N/A",
        })

    _write_csv(output_path, rows)


def generate_ablation_table(
    full_model_auroc: float,
    ablation_results: Dict[str, Dict],
    output_path: str,
) -> None:
    rows: List[Dict[str, Any]] = []
    rows.append({
        "Ablation": "Full Model",
        "AUROC": _safe(full_model_auroc),
        "AUPRC": "N/A",
        "ECE": "N/A",
        "ΔAUROC vs Full": "—",
        "Notes": "Reference",
    })

    ordered_keys = [
        "no_adv_heads", "no_hosp_adv", "no_treat_adv",
        "no_contrastive", "no_irm", "no_s_t_et_only",
    ]
    written = set()
    for key in ordered_keys + [k for k in ablation_results if k not in ordered_keys]:
        if key in written:
            continue
        res = ablation_results.get(key, {})
        if not isinstance(res, dict) or "auroc" not in res:
            continue
        auroc = res["auroc"]
        delta = auroc - full_model_auroc
        rows.append({
            "Ablation": _display_name(key),
            "AUROC": _safe(auroc),
            "AUPRC": _safe(res.get("auprc")),
            "ECE": _safe(res.get("ece")),
            "ΔAUROC vs Full": _safe(delta),
            "Notes": res.get("note", ""),
        })
        written.add(key)

    _write_csv(output_path, rows)


def generate_baseline_table(
    baseline_results: Dict[str, Dict],
    output_path: str,
) -> None:
    rows: List[Dict[str, Any]] = []
    for name in sorted(baseline_results.keys()):
        res = baseline_results[name]
        if not isinstance(res, dict) or "auroc" not in res:
            continue
        best_cfg = res.get("best_config", {})
        rows.append({
            "Baseline": _display_name(name),
            "AUROC": _safe(res.get("auroc")),
            "AUPRC": _safe(res.get("auprc")),
            "ECE": _safe(res.get("ece")),
            "OOD AUROC": _safe(res.get("ooh_auroc")),
            "Best hidden_dim": str(best_cfg.get("hidden_dim", "N/A")),
            "Best lr": str(best_cfg.get("lr", "N/A")),
            "Best dropout": str(best_cfg.get("dropout", "N/A")),
        })
    _write_csv(output_path, rows)


def generate_metrics_csv(results: Dict[str, Any], output_path: str) -> None:
    """Flat per-model metric rows for quick import into analysis tools."""
    rows: List[Dict[str, Any]] = []

    main = results.get("main_model", {})
    rows.append({
        "model": "full_model",
        "auroc": _safe(main.get("auroc")),
        "auprc": _safe(main.get("auprc")),
        "accuracy": _safe(main.get("accuracy")),
        "ece": _safe(main.get("ece")),
        "in_dist_auroc": _safe(main.get("auroc")),
        "ood_auroc": _safe(main.get("ooh_metrics", {}).get("auroc") if main.get("ooh_metrics") else None),
        "gen_gap": _safe(main.get("domain_shift_drop")),
        "mean_auroc_3seed": _safe(results.get("stability", {}).get("mean_auroc")),
        "std_auroc_3seed": _safe(results.get("stability", {}).get("std_auroc")),
    })

    for name, res in results.get("baselines", {}).items():
        if not isinstance(res, dict) or "auroc" not in res:
            continue
        rows.append({
            "model": name,
            "auroc": _safe(res.get("auroc")),
            "auprc": _safe(res.get("auprc")),
            "accuracy": _safe(res.get("accuracy")),
            "ece": _safe(res.get("ece")),
            "in_dist_auroc": _safe(res.get("auroc")),
            "ood_auroc": _safe(res.get("ooh_auroc")),
            "gen_gap": "N/A",
            "mean_auroc_3seed": "N/A",
            "std_auroc_3seed": "N/A",
        })

    for name, res in results.get("ablations", {}).items():
        if not isinstance(res, dict) or "auroc" not in res:
            continue
        rows.append({
            "model": f"ablation_{name}",
            "auroc": _safe(res.get("auroc")),
            "auprc": _safe(res.get("auprc")),
            "accuracy": _safe(res.get("accuracy")),
            "ece": _safe(res.get("ece")),
            "in_dist_auroc": _safe(res.get("auroc")),
            "ood_auroc": "N/A",
            "gen_gap": "N/A",
            "mean_auroc_3seed": "N/A",
            "std_auroc_3seed": "N/A",
        })

    _write_csv(output_path, rows)


# ---------------------------------------------------------------------------
# Master reporting function
# ---------------------------------------------------------------------------

def generate_all_reports(
    results: Dict[str, Any],
    output_dir: str,
    main_y_true: Optional[np.ndarray] = None,
    main_y_score: Optional[np.ndarray] = None,
    best_baseline_y_true: Optional[np.ndarray] = None,
    best_baseline_y_score: Optional[np.ndarray] = None,
    best_baseline_name: str = "best_baseline",
) -> None:
    """
    Generate all publication-ready plots and tables from results dict.
    """
    plots_dir = os.path.join(output_dir, "plots")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    main_res = results.get("main_model", {})
    baselines = results.get("baselines", {})
    ablations = results.get("ablations", {})
    domain_gen = results.get("domain_generalization", {})
    stability = results.get("stability", {})

    model_auroc = main_res.get("auroc", 0.0)
    model_auprc = main_res.get("auprc", 0.0)
    model_ece = main_res.get("ece", 0.0)
    model_ood = main_res.get("ooh_metrics", {}).get("auroc") if main_res.get("ooh_metrics") else None
    model_gap = main_res.get("domain_shift_drop")

    logger.info("=== GENERATING REPORTS ===")

    # Plot 1: AUROC comparison
    plot_auroc_comparison(
        model_auroc=model_auroc,
        baseline_results=baselines,
        ablation_results=ablations,
        output_path=os.path.join(plots_dir, "auroc_comparison.png"),
    )

    # Plot 2: PR curve (requires raw scores)
    if main_y_true is not None and best_baseline_y_true is not None:
        plot_pr_curve(
            main_y_true=main_y_true,
            main_y_score=main_y_score,
            baseline_y_true=best_baseline_y_true,
            baseline_y_score=best_baseline_y_score,
            baseline_name=best_baseline_name,
            output_path=os.path.join(plots_dir, "pr_curve.png"),
        )

    # Plot 3: Calibration curve
    ece_paths: Dict[str, str] = {}
    for name in ["main_model"] + list(baselines.keys()):
        stem = "main_model" if name == "main_model" else name
        p = os.path.join(output_dir, f"ece_{stem}.json")
        if os.path.exists(p):
            ece_paths[name] = p
    if ece_paths:
        plot_calibration_curve(
            ece_json_paths=ece_paths,
            output_path=os.path.join(plots_dir, "calibration_curve.png"),
        )

    # Plot 4: Domain generalization
    if domain_gen.get("models"):
        plot_domain_generalization(
            domain_gen_metrics=domain_gen,
            output_path=os.path.join(plots_dir, "domain_generalization.png"),
        )

    # Plot 5: Ablation study
    if ablations:
        plot_ablation_study(
            full_model_auroc=model_auroc,
            ablation_results=ablations,
            output_path=os.path.join(plots_dir, "ablation_study.png"),
        )

    # Tables
    generate_main_results_table(
        model_auroc=model_auroc,
        model_auprc=model_auprc,
        model_ece=model_ece,
        model_in_dist=model_auroc,
        model_ood=model_ood,
        model_gap=model_gap,
        model_mean_auroc=stability.get("mean_auroc"),
        model_std_auroc=stability.get("std_auroc"),
        baseline_results=baselines,
        output_path=os.path.join(tables_dir, "main_results_table.csv"),
    )
    generate_ablation_table(
        full_model_auroc=model_auroc,
        ablation_results=ablations,
        output_path=os.path.join(tables_dir, "ablation_table.csv"),
    )
    generate_baseline_table(
        baseline_results=baselines,
        output_path=os.path.join(tables_dir, "baseline_table.csv"),
    )
    generate_metrics_csv(
        results=results,
        output_path=os.path.join(output_dir, "metrics.csv"),
    )

    logger.info("=== REPORTS COMPLETE ===")
    logger.info("  Plots  → %s", plots_dir)
    logger.info("  Tables → %s", tables_dir)
    logger.info("  Metrics CSV → %s", os.path.join(output_dir, "metrics.csv"))
