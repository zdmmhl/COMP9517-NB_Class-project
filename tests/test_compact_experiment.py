import json

from evaluation.compact_experiment import _copy_sanitized_training_state


def test_training_state_export_removes_machine_specific_paths(tmp_path):
    source = tmp_path / "source.json"
    destination = tmp_path / "destination.json"
    source.write_text(
        json.dumps(
            {
                "completed_epochs": 12,
                "last_checkpoint": "C:/private/results/last_checkpoint.pt",
                "best_checkpoint": "C:/private/results/best_model.pt",
            }
        ),
        encoding="utf-8",
    )

    _copy_sanitized_training_state(source, destination)

    exported = json.loads(destination.read_text(encoding="utf-8"))
    assert exported["completed_epochs"] == 12
    assert exported["last_checkpoint"] == "last_checkpoint.pt"
    assert exported["best_checkpoint"] == "best_model.pt"
