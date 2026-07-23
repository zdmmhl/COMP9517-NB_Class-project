# Mock Ablation Study

This deterministic fixture demonstrates the expected ablation directory layout.
It compares `basic` and `strong` augmentation using the same two random seeds.
Every setting other than `augmentation` is identical.

```text
mock_ablation/
|-- study.json
|-- basic/
|   |-- seed_42/
|   `-- seed_43/
`-- strong/
    |-- seed_42/
    `-- seed_43/
```

Each seed directory contains:

```text
configuration.csv
metrics.csv
history.csv
```

Generate the report outputs from the repository root:

```powershell
python scripts\plot_ablations.py `
  --study-dir tests\fixtures\mock_ablation `
  --output-dir outputs\ablations\mock_augmentation
```
