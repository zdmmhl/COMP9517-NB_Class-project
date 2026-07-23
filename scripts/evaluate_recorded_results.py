"""Run evaluation over a committed result package without retraining models."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ablation import write_ablation_outputs
from evaluation.comparison import write_comparison_outputs
from evaluation.recorded_results import (
    exported_metrics_summary,
    prepare_ablation_study,
    prepare_shared_manifests,
    prepare_summary_only_entry,
    prepare_top5_artifact,
)
from scripts.evaluate_artifacts import run_evaluation


ABLATION_STUDIES = [
    {
        "key": "handcrafted_features",
        "name": "Handcrafted feature descriptor comparison",
        "factor": "feature",
        "baseline": "color",
        "variants": [
            ("color", "color_sgd_svm", "HSV colour histogram"),
            ("lbp", "lbp_sgd_svm", "Uniform LBP"),
            ("hog", "hog_sgd_svm", "HOG"),
            ("sift_bovw", "sift_bovw_sgd_svm", "SIFT-BoVW"),
        ],
    },
    {
        "key": "hog_classifiers",
        "name": "HOG classifier comparison",
        "factor": "classifier",
        "baseline": "sgd_svm",
        "variants": [
            ("sgd_svm", "hog_sgd_svm", "SGD linear SVM"),
            ("random_forest", "hog_random_forest", "Random Forest"),
        ],
    },
    {
        "key": "color_classifiers",
        "name": "Colour histogram classifier comparison",
        "factor": "classifier",
        "baseline": "sgd_svm",
        "variants": [
            ("sgd_svm", "color_sgd_svm", "SGD linear SVM"),
            ("linear_svc", "color_linear_svc", "LinearSVC"),
        ],
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt and evaluate committed experiment records "
            "without loading datasets or model checkpoints."
        )
    )
    parser.add_argument(
        "--final-results-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "final_results" / "inat500",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Run output directory. Defaults to "
            "outputs/runs/inat<num_classes>_seed<seed>."
        ),
    )
    parser.add_argument("--subset-classes", type=int, default=25)
    parser.add_argument("--top-confusions", type=int, default=30)
    return parser


def load_split_identity(final_root: Path) -> tuple[str, int, int]:
    summary_path = (
        final_root / "reproducibility" / "data_splits" / "split_summary.json"
    )
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing split summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    num_classes = int(summary["num_classes"])
    seed = int(summary["seed"])
    return f"inat{num_classes}_seed{seed}", num_classes, seed


def main() -> None:
    args = build_parser().parse_args()
    final_root = args.final_results_root
    split_id, expected_num_classes, split_seed = load_split_identity(final_root)
    output_root = args.output_root or PROJECT_ROOT / "outputs" / "runs" / split_id
    methods_root = final_root / "methods"
    standardized_root = output_root / "standardized_inputs"
    evaluation_root = output_root / "methods"
    examples_root = output_root / "examples"

    class_mapping, test_manifest, original_test = prepare_shared_manifests(
        final_root,
        standardized_root / "shared",
    )
    num_classes = len(pd.read_csv(class_mapping))
    num_test_samples = len(original_test)
    if num_classes != expected_num_classes:
        raise ValueError(
            "Class mapping count does not match split_summary.json: "
            f"{num_classes} != {expected_num_classes}"
        )
    fully_recomputed = []
    summary_only = []

    for method_dir in sorted(methods_root.iterdir()):
        if not method_dir.is_dir() or not (method_dir / "metrics.csv").is_file():
            continue
        method_key = method_dir.name
        method_output = evaluation_root / method_key
        top5_path = method_dir / "test_predictions_top5.csv"
        if top5_path.is_file():
            artifact_dir = standardized_root / "methods" / method_key
            prepare_top5_artifact(method_dir, original_test, artifact_dir)
            run_evaluation(
                Namespace(
                    prediction_dir=artifact_dir,
                    class_mapping=class_mapping,
                    test_manifest=test_manifest,
                    history=None,
                    output_dir=method_output,
                    image_root=None,
                    examples_dir=examples_root / method_key,
                    top_confusions=args.top_confusions,
                    subset_classes=args.subset_classes,
                    example_count=12,
                )
            )
            exported = exported_metrics_summary(method_dir)
            recomputed = json.loads(
                (method_output / "metrics.json").read_text(encoding="utf-8")
            )
            for metric in (
                "top1_accuracy",
                "top5_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
            ):
                if abs(float(exported[metric]) - float(recomputed[metric])) > 1e-10:
                    raise ValueError(
                        f"{method_key} recomputed {metric} does not match export"
                    )
            fully_recomputed.append(method_key)
        else:
            prepare_summary_only_entry(method_dir, method_output)
            summary_only.append(method_key)

    write_comparison_outputs(
        evaluation_root,
        output_root / "comparison",
    )

    ablation_keys = []
    for definition in ABLATION_STUDIES:
        study_input = output_root / "ablation_inputs" / definition["key"]
        prepare_ablation_study(
            final_root,
            study_input,
            study_name=definition["name"],
            factor=definition["factor"],
            baseline_variant=definition["baseline"],
            variants=definition["variants"],
        )
        write_ablation_outputs(
            study_input,
            output_root / "ablations" / definition["key"],
        )
        ablation_keys.append(definition["key"])

    manifest = {
        "source": final_root.as_posix(),
        "split_id": split_id,
        "num_classes": num_classes,
        "num_test_samples": num_test_samples,
        "random_seed": split_seed,
        "fully_recomputed_methods": fully_recomputed,
        "comparison_only_methods": summary_only,
        "ablation_studies": ablation_keys,
        "limitations": [
            "The repository does not include the image dataset, so new "
            "qualitative image grids were not rendered.",
            "Methods without per-image Top-5 records were included using "
            "their committed aggregate and per-class metrics.",
            "The recorded ablations use one random seed, so their standard "
            "deviation is zero and no cross-seed uncertainty can be claimed.",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
