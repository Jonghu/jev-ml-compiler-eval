"""Decision model interfaces and registration."""

from jev_compiler_eval.models.base import BaseDecisionModel
from jev_compiler_eval.models.jev import JevModel
from jev_compiler_eval.models.jev_tournament import JevTournamentModel
from jev_compiler_eval.models.random import RandomModel
from jev_compiler_eval.models.registry import get_model_class, register_model
from jev_compiler_eval.models.tenset_xgb import TenSetXGBModel

register_model("random", RandomModel)
register_model("jev", JevModel)
register_model("jev-tournament", JevTournamentModel)
register_model("tenset_xgb", TenSetXGBModel)

__all__ = [
    "BaseDecisionModel",
    "JevModel",
    "JevTournamentModel",
    "RandomModel",
    "TenSetXGBModel",
    "get_model_class",
    "register_model",
]
