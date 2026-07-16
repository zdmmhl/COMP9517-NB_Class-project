"""PyTorch dataset for the selected iNaturalist species subset.

The implementation will read a generated manifest rather than resampling data
inside each experiment. This keeps all methods on identical splits.
"""


class INatSpeciesDataset:
    """Placeholder for the manifest-backed classification dataset."""

    def __init__(self, manifest_path, transform=None):
        self.manifest_path = manifest_path
        self.transform = transform
        raise NotImplementedError("Dataset loading will be implemented next.")
