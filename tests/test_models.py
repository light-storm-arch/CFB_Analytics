import numpy as np
import pytest

from cfb.features.build import feature_columns
from cfb.models import available_models, build_model
from cfb.models.base import SpreadModel

FAST_MODELS = ["ridge", "huber", "forest", "xgboost", "lightgbm"]


@pytest.mark.parametrize("name", FAST_MODELS)
def test_model_fits_and_predicts(features, name):
    cols = feature_columns()
    train = features[features["season"] < features["season"].max()]
    test = features[features["season"] == features["season"].max()]
    model = build_model(name, cols, n_estimators=120)
    model.fit(train)
    pred = model.predict(test)
    assert set(pred.columns) >= {"game_id", "pred_margin", "pred_total"}
    assert len(pred) == len(test)
    assert np.isfinite(pred["pred_margin"]).all()
    # Sanity: predictions must live in a plausible range, not blow up.
    assert pred["pred_margin"].abs().max() < 80
    assert pred["pred_total"].between(10, 130).mean() > 0.95


def test_registry_lists_and_rejects(features):
    assert "ridge" in available_models()
    with pytest.raises(KeyError):
        build_model("does-not-exist", feature_columns())


def test_ensemble_beats_or_matches_its_worst_member(features):
    cols = feature_columns()
    train = features[features["season"] < features["season"].max()]
    test = features[features["season"] == features["season"].max()]
    ens = build_model("ensemble", cols, members=["ridge", "forest"], n_estimators=120)
    ens.fit(train)
    pred = ens.predict(test)
    y = test["margin"].to_numpy()
    mae = lambda p: np.nanmean(np.abs(p - y))  # noqa: E731
    members = [mae(pred[f"pred_margin__{m.name}"].to_numpy()) for m in ens.members]
    assert mae(pred["pred_margin"].to_numpy()) <= max(members) + 1e-9


def test_save_and_load_roundtrip(features, tmp_path):
    cols = feature_columns()
    model = build_model("ridge", cols).fit(features)
    path = model.save(tmp_path / "m.joblib")
    loaded = SpreadModel.load(path)
    a = model.predict(features.head(20))["pred_margin"].to_numpy()
    b = loaded.predict(features.head(20))["pred_margin"].to_numpy()
    np.testing.assert_allclose(a, b)


def test_all_nan_features_are_dropped(features):
    f = features.copy()
    f["rat_margin"] = np.nan
    model = build_model("ridge", feature_columns()).fit(f)
    assert "rat_margin" not in model.features_used
    assert "rat_margin" in model.report.dropped_features
