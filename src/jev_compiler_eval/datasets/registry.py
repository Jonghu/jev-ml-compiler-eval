"""Registry for dataset implementations."""

from jev_compiler_eval.datasets.base import BaseDataset

_DATASET_REGISTRY: dict[str, type[BaseDataset]] = {}


def register_dataset(
    dataset_type: str,
    dataset_class: type[BaseDataset],
) -> None:
    """Register a dataset class under a configuration type name."""

    if not dataset_type.strip():
        raise ValueError("dataset type must not be empty")
    if not isinstance(dataset_class, type) or not issubclass(
        dataset_class, BaseDataset
    ):
        raise TypeError("registered datasets must inherit from BaseDataset")
    if dataset_type in _DATASET_REGISTRY:
        raise ValueError(f'dataset type "{dataset_type}" is already registered')
    _DATASET_REGISTRY[dataset_type] = dataset_class


def get_dataset_class(dataset_type: str) -> type[BaseDataset]:
    """Return the dataset class registered for a type name."""

    try:
        return _DATASET_REGISTRY[dataset_type]
    except KeyError as error:
        raise KeyError(f'unknown dataset type "{dataset_type}"') from error
