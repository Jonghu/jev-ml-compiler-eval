"""Decision model interfaces and registration."""

from jev_compiler_eval.models.base import BaseDecisionModel
from jev_compiler_eval.models.random import RandomModel
from jev_compiler_eval.models.registry import get_model_class, register_model

register_model("random", RandomModel)

__all__ = [
    "BaseDecisionModel",
    "RandomModel",
    "get_model_class",
    "register_model",
]
