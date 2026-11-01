"""Registry for decision model implementations."""

from jev_compiler_eval.models.base import BaseDecisionModel

_MODEL_REGISTRY: dict[str, type[BaseDecisionModel]] = {}


def register_model(
    model_type: str,
    model_class: type[BaseDecisionModel],
) -> None:
    """Register a decision model class under a configuration type name."""

    if not model_type.strip():
        raise ValueError("model type must not be empty")
    if not isinstance(model_class, type) or not issubclass(
        model_class, BaseDecisionModel
    ):
        raise TypeError("registered models must inherit from BaseDecisionModel")
    if model_type in _MODEL_REGISTRY:
        raise ValueError(f'model type "{model_type}" is already registered')
    _MODEL_REGISTRY[model_type] = model_class


def get_model_class(model_type: str) -> type[BaseDecisionModel]:
    """Return the decision model class registered for a type name."""

    try:
        return _MODEL_REGISTRY[model_type]
    except KeyError as error:
        raise KeyError(f'unknown model type "{model_type}"') from error
