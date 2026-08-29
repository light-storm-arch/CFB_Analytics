"""Name -> model class mapping, so models can be flipped from the CLI/dashboard."""
from __future__ import annotations

from cfb.models.base import SpreadModel
from cfb.models.ensemble import EnsembleSpreadModel
from cfb.models.linear import ElasticNetSpreadModel, HuberSpreadModel, RidgeSpreadModel
from cfb.models.trees import (
    LightGBMSpreadModel,
    RandomForestSpreadModel,
    XGBoostSpreadModel,
)

MODEL_REGISTRY: dict[str, type[SpreadModel]] = {
    "ridge": RidgeSpreadModel,
    "elasticnet": ElasticNetSpreadModel,
    "huber": HuberSpreadModel,
    "forest": RandomForestSpreadModel,
    "xgboost": XGBoostSpreadModel,
    "lightgbm": LightGBMSpreadModel,
    "ensemble": EnsembleSpreadModel,
}

#: Members used when the user asks for "ensemble" without naming its parts.
DEFAULT_ENSEMBLE = ("ridge", "xgboost", "forest")


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


def build_model(name: str, features: list[str], members: list[str] | None = None,
                **params) -> SpreadModel:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{name}'. Available: {available_models()}")
    if key == "ensemble":
        member_names = list(members or DEFAULT_ENSEMBLE)
        built = [build_model(m, features, **params) for m in member_names]
        return EnsembleSpreadModel(features, members=built, **params)
    return MODEL_REGISTRY[key](features, **params)
