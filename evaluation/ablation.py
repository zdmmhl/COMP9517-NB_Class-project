"""Reusable loading, validation, aggregation, and plotting for ablations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_METRICS = [
    "top1_accuracy",
    "top5_accuracy",
    "macro_f1",
]
OPTIONAL_METRICS = [
    "macro_precision",
    "macro_recall",
    "balanced_accuracy",
    "training_time_seconds",
    "inference_time_seconds",
]
DEFAULT_IGNORED_CONFIG_KEYS = {
    "method_key",
    "method_name",
    "run_id",
    "output_dir",
    "checkpoint_path",
}
HISTORY_ALIASES = {
    "val_loss": ["val_loss"],
    "val_top1": ["val_top1", "val_top1_accuracy"],
    "val_macro_f1": ["val_macro_f1"],
}


class AblationValidationError(ValueError):
    """Raised when an ablation does not satisfy its controlled-study contract."""


@dataclass(frozen=True)
class AblationStudy:
    """Validated raw runs and metadata for one controlled factor."""

    root: Path
    spec: dict[str, Any]
    runs: pd.DataFrame
    configs: dict[str, dict[str, str]]
    histories: dict[str, pd.DataFrame]

    @property
    def variants(self) -> list[str]:
        return list(self.spec["variant_order"])

    @property
    def baseline_variant(self) -> str:
        return str(self.spec["baseline_variant"])


def _load_study_spec(study_dir: Path) -> dict[str, Any]:
    spec_path = study_dir / "study.json"
    if not spec_path.is_file():
        raise AblationValidationError(f"Missing ablation study file: {spec_path}")
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AblationValidationError(
            f"Could not read {spec_path}: {exc}"
        ) from exc

    required = ["study_name", "factor", "baseline_variant", "variant_order"]
    missing = [key for key in required if key not in spec]
    if missing:
        raise AblationValidationError(
            f"study.json is missing required fields: {missing}"
        )
    variants = spec["variant_order"]
    if not isinstance(variants, list) or len(variants) < 2:
        raise AblationValidationError(
            "study.json variant_order must contain at least two variants"
        )
    if len(set(variants)) != len(variants):
        raise AblationValidationError("study.json variant_order has duplicates")
    if spec["baseline_variant"] not in variants:
        raise AblationValidationError(
            "study.json baseline_variant must appear in variant_order"
        )

    allowed = spec.get("allowed_config_differences", [spec["factor"]])
    if not isinstance(allowed, list) or not allowed:
        raise AblationValidationError(
            "allowed_config_differences must be a non-empty list"
        )
    spec["allowed_config_differences"] = allowed
    spec.setdefault("ignored_config_keys", [])
    spec.setdefault("display_names", {})
    return spec


def _read_configuration(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise AblationValidationError(f"Missing configuration.csv: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != ["parameter", "value"]:
        raise AblationValidationError(
            f"{path} must have exactly the columns parameter,value"
        )
    if frame["parameter"].duplicated().any():
        raise AblationValidationError(f"{path} has duplicate parameters")
    return dict(zip(frame["parameter"], frame["value"]))


def _read_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AblationValidationError(f"Missing metrics.csv: {path}")
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise AblationValidationError(f"{path} must contain exactly one row")
    missing = [metric for metric in REQUIRED_METRICS if metric not in frame]
    if missing:
        raise AblationValidationError(
            f"{path} is missing required metrics: {missing}"
        )
    row = frame.iloc[0].to_dict()
    for metric in [*REQUIRED_METRICS, *OPTIONAL_METRICS]:
        if metric not in row or pd.isna(row[metric]) or row[metric] == "":
            continue
        try:
            row[metric] = float(row[metric])
        except (TypeError, ValueError) as exc:
            raise AblationValidationError(
                f"{path} field {metric} must be numeric"
            ) from exc
    return row


def _find_run_directories(variant_dir: Path) -> list[Path]:
    if (
        (variant_dir / "configuration.csv").is_file()
        and (variant_dir / "metrics.csv").is_file()
    ):
        return [variant_dir]
    return sorted(
        path.parent
        for path in variant_dir.rglob("metrics.csv")
        if (path.parent / "configuration.csv").is_file()
    )


def _run_seed(config: dict[str, str], metrics: dict[str, Any], path: Path) -> str:
    seed = config.get("seed") or metrics.get("random_seed")
    if seed is None or str(seed).strip() == "":
        raise AblationValidationError(
            f"Run {path} must record seed in configuration.csv or metrics.csv"
        )
    return str(seed)


def load_ablation_study(study_dir: str | Path) -> AblationStudy:
    """Load one study without assuming anything about model implementation."""
    study_dir = Path(study_dir)
    spec = _load_study_spec(study_dir)
    rows = []
    configs: dict[str, dict[str, str]] = {}
    histories: dict[str, pd.DataFrame] = {}

    for order, variant in enumerate(spec["variant_order"]):
        variant_dir = study_dir / variant
        if not variant_dir.is_dir():
            raise AblationValidationError(
                f"Missing variant directory {variant_dir}"
            )
        run_dirs = _find_run_directories(variant_dir)
        if not run_dirs:
            raise AblationValidationError(
                f"No complete runs found under {variant_dir}"
            )
        for run_dir in run_dirs:
            config = _read_configuration(run_dir / "configuration.csv")
            metrics = _read_metrics(run_dir / "metrics.csv")
            seed = _run_seed(config, metrics, run_dir)
            run_id = f"{variant}/seed_{seed}"
            if run_id in configs:
                raise AblationValidationError(f"Duplicate ablation run {run_id}")

            display_names = spec["display_names"]
            row = {
                "run_id": run_id,
                "variant": variant,
                "display_name": display_names.get(variant, variant),
                "variant_order": order,
                "seed": seed,
                "run_dir": run_dir.as_posix(),
            }
            for field in (
                "split_id",
                "num_classes",
                "num_test_samples",
                *REQUIRED_METRICS,
                *OPTIONAL_METRICS,
            ):
                row[field] = metrics.get(field)
            rows.append(row)
            configs[run_id] = config

            history_path = run_dir / "history.csv"
            if history_path.is_file():
                history = pd.read_csv(history_path)
                if "epoch" not in history:
                    raise AblationValidationError(
                        f"{history_path} is missing epoch"
                    )
                histories[run_id] = history

    runs = pd.DataFrame(rows).sort_values(
        ["variant_order", "seed"]
    ).reset_index(drop=True)
    study = AblationStudy(study_dir, spec, runs, configs, histories)
    validate_ablation_study(study)
    return study


def _unique_nonempty(frame: pd.DataFrame, column: str) -> set[str]:
    if column not in frame:
        return set()
    return {
        str(value)
        for value in frame[column]
        if not pd.isna(value) and str(value).strip() != ""
    }


def validate_ablation_study(study: AblationStudy) -> dict[str, Any]:
    """Verify data consistency and one-factor-at-a-time configuration changes."""
    runs = study.runs
    for column in ("split_id", "num_classes", "num_test_samples"):
        values = _unique_nonempty(runs, column)
        if len(values) > 1:
            raise AblationValidationError(
                f"Ablation runs use different {column} values: {sorted(values)}"
            )

    seed_sets = {
        variant: set(runs.loc[runs["variant"] == variant, "seed"])
        for variant in study.variants
    }
    baseline_seeds = seed_sets[study.baseline_variant]
    for variant, seeds in seed_sets.items():
        if seeds != baseline_seeds:
            raise AblationValidationError(
                f"Variant {variant} uses seeds {sorted(seeds)}, expected "
                f"{sorted(baseline_seeds)}"
            )

    ignored = DEFAULT_IGNORED_CONFIG_KEYS | set(
        study.spec["ignored_config_keys"]
    )
    allowed = set(study.spec["allowed_config_differences"])
    checked_pairs = []
    for variant in study.variants:
        if variant == study.baseline_variant:
            continue
        for seed in sorted(baseline_seeds):
            baseline_id = f"{study.baseline_variant}/seed_{seed}"
            candidate_id = f"{variant}/seed_{seed}"
            baseline = study.configs[baseline_id]
            candidate = study.configs[candidate_id]
            changed = {
                key
                for key in set(baseline) | set(candidate)
                if baseline.get(key, "") != candidate.get(key, "")
                and key not in ignored
            }
            unexpected = changed - allowed
            if unexpected:
                details = {
                    key: [baseline.get(key, ""), candidate.get(key, "")]
                    for key in sorted(unexpected)
                }
                raise AblationValidationError(
                    f"{candidate_id} changes uncontrolled parameters: {details}"
                )
            if not changed & allowed:
                raise AblationValidationError(
                    f"{candidate_id} does not change the declared factor "
                    f"{study.spec['factor']}"
                )
            checked_pairs.append(
                {
                    "baseline_run": baseline_id,
                    "candidate_run": candidate_id,
                    "changed_parameters": sorted(changed & allowed),
                }
            )

    return {
        "study_name": study.spec["study_name"],
        "factor": study.spec["factor"],
        "baseline_variant": study.baseline_variant,
        "variants": study.variants,
        "seeds": sorted(baseline_seeds),
        "num_runs": len(runs),
        "checked_pairs": checked_pairs,
    }


def aggregate_ablation_metrics(study: AblationStudy) -> pd.DataFrame:
    """Aggregate repeated seeds into report-ready means and standard deviations."""
    metrics = [
        metric
        for metric in [*REQUIRED_METRICS, *OPTIONAL_METRICS]
        if metric in study.runs and study.runs[metric].notna().any()
    ]
    grouped = study.runs.groupby(
        ["variant", "display_name", "variant_order"],
        sort=False,
    )
    summary = grouped[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    counts = grouped.size().rename("n_runs").reset_index()
    summary = summary.merge(
        counts,
        on=["variant", "display_name", "variant_order"],
    )
    std_columns = [column for column in summary if column.endswith("_std")]
    summary[std_columns] = summary[std_columns].fillna(0.0)
    numeric_columns = summary.select_dtypes(include="number").columns
    summary[numeric_columns] = summary[numeric_columns].round(12)
    return summary.sort_values("variant_order").reset_index(drop=True)


def compute_ablation_deltas(
    summary: pd.DataFrame,
    baseline_variant: str,
) -> pd.DataFrame:
    """Compute each variant's metric change relative to the declared baseline."""
    baseline_rows = summary[summary["variant"] == baseline_variant]
    if len(baseline_rows) != 1:
        raise AblationValidationError(
            f"Could not identify baseline summary row {baseline_variant}"
        )
    baseline = baseline_rows.iloc[0]
    output = summary[["variant", "display_name", "variant_order"]].copy()
    for metric in REQUIRED_METRICS:
        column = f"{metric}_mean"
        if column in summary:
            output[f"delta_{metric}"] = summary[column] - float(baseline[column])
    numeric_columns = output.select_dtypes(include="number").columns
    output[numeric_columns] = output[numeric_columns].round(12)
    return output


def _plot_metric_bars(
    summary: pd.DataFrame,
    output_path: Path,
    study_name: str,
) -> None:
    labels = summary["display_name"].tolist()
    x = np.arange(len(labels))
    metrics = [
        ("top1_accuracy", "Top-1"),
        ("top5_accuracy", "Top-5"),
        ("macro_f1", "Macro-F1"),
    ]
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.2), 5.5))
    upper_bound = 0.0
    for index, (metric, label) in enumerate(metrics):
        means = summary[f"{metric}_mean"].to_numpy(dtype=float)
        stds = summary[f"{metric}_std"].to_numpy(dtype=float)
        upper_bound = max(upper_bound, float(np.max(means + stds)))
        offset = (index - 1) * width
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=label,
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1 if upper_bound >= 0.2 else max(0.05, upper_bound * 1.3))
    ax.set_ylabel("Score")
    ax.set_title(f"Ablation Performance: {study_name}")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_deltas(
    deltas: pd.DataFrame,
    output_path: Path,
    study_name: str,
) -> None:
    labels = deltas["display_name"].tolist()
    x = np.arange(len(labels))
    columns = [
        ("delta_top1_accuracy", "Top-1"),
        ("delta_top5_accuracy", "Top-5"),
        ("delta_macro_f1", "Macro-F1"),
    ]
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.2), 5.5))
    for index, (column, label) in enumerate(columns):
        if column not in deltas:
            continue
        offset = (index - 1) * width
        ax.bar(
            x + offset,
            100 * deltas[column].to_numpy(dtype=float),
            width,
            label=label,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Change from baseline (percentage points)")
    ax.set_title(f"Improvement over Baseline: {study_name}")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _history_column(history: pd.DataFrame, metric: str) -> str | None:
    return next(
        (
            column
            for column in HISTORY_ALIASES[metric]
            if column in history.columns
        ),
        None,
    )


def _plot_training_curves(study: AblationStudy, output_path: Path) -> bool:
    available = []
    for metric in HISTORY_ALIASES:
        if any(
            _history_column(history, metric) is not None
            for history in study.histories.values()
        ):
            available.append(metric)
    if not available:
        return False

    fig, axes = plt.subplots(
        1,
        len(available),
        figsize=(5.2 * len(available), 4.2),
        squeeze=False,
    )
    for ax, metric in zip(axes.flat, available):
        for variant in study.variants:
            frames = []
            for run_id, history in study.histories.items():
                if not run_id.startswith(f"{variant}/"):
                    continue
                column = _history_column(history, metric)
                if column is None:
                    continue
                frames.append(
                    history[["epoch", column]].rename(columns={column: "value"})
                )
            if not frames:
                continue
            combined = pd.concat(frames, ignore_index=True)
            aggregate = combined.groupby("epoch")["value"].agg(["mean", "std"])
            aggregate["std"] = aggregate["std"].fillna(0.0)
            display = study.spec["display_names"].get(variant, variant)
            ax.plot(aggregate.index, aggregate["mean"], label=display)
            ax.fill_between(
                aggregate.index,
                aggregate["mean"] - aggregate["std"],
                aggregate["mean"] + aggregate["std"],
                alpha=0.15,
            )
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    axes.flat[0].legend()
    fig.suptitle(f"Training Dynamics: {study.spec['study_name']}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def write_ablation_outputs(
    study_dir: str | Path,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate one study and write report-ready tables and figures."""
    study = load_ablation_study(study_dir)
    report = validate_ablation_study(study)
    summary = aggregate_ablation_metrics(study)
    deltas = compute_ablation_deltas(summary, study.baseline_variant)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    study.runs.to_csv(
        output_dir / "ablation_runs.csv",
        index=False,
        lineterminator="\n",
    )
    summary.to_csv(
        output_dir / "ablation_summary.csv",
        index=False,
        lineterminator="\n",
    )
    deltas.to_csv(
        output_dir / "ablation_deltas.csv",
        index=False,
        lineterminator="\n",
    )
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    study_name = str(study.spec["study_name"])
    _plot_metric_bars(
        summary,
        output_dir / "ablation_metrics.png",
        study_name,
    )
    _plot_metric_deltas(
        deltas,
        output_dir / "ablation_deltas.png",
        study_name,
    )
    _plot_training_curves(study, output_dir / "training_curves.png")
    return summary, deltas
