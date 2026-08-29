"""End-to-end: features -> walk-forward -> distributions -> saved artifacts."""
import numpy as np
import pytest

from cfb.evaluation.backtest import ats_record, backtest_report, walk_forward_predictions
from cfb.evaluation.calibration import crps_discrete, distribution_report
from cfb.features.build import feature_columns
from cfb.pipeline import Predictor, PredictorConfig, train_predictor


@pytest.fixture(scope="module")
def trained(features):
    cfg = PredictorConfig(model_name="ridge", refit="season", min_train_games=300,
                          mc_sims=4000)
    return train_predictor(features, cfg, progress=False)


def test_walk_forward_never_trains_on_the_future(features):
    cols = feature_columns()
    oos = walk_forward_predictions(features, cols, "ridge", refit="season",
                                   min_train_games=300, progress=False)
    assert not oos.empty
    # The earliest season cannot be predicted -- there is no prior history.
    assert oos["season"].min() > features["season"].min()
    assert oos["n_train"].is_monotonic_increasing or oos["n_train"].nunique() > 1


def test_backtest_report_is_sane(trained):
    _, oos = trained
    rep = backtest_report(oos)
    assert rep.n > 100
    assert 8 < rep.margin_mae < 20
    assert 0.55 < rep.su_accuracy < 0.85
    assert abs(rep.margin_bias) < 2.0


def test_sigma_model_is_calibrated_out_of_sample(trained):
    predictor, oos = trained
    z = predictor.sigma_model.standardized_residuals(oos)
    assert np.std(z) == pytest.approx(1.0, abs=0.1)
    sigmas = predictor.sigma_model.predict(oos)
    assert (sigmas > 5).all() and (sigmas < 30).all()
    # Variance must actually respond to the scoring environment.
    assert np.corrcoef(sigmas, oos["pred_total"])[0, 1] > 0.1


def test_distributions_are_calibrated(trained):
    predictor, oos = trained
    sample = oos.sample(min(400, len(oos)), random_state=2)
    sigmas = predictor.sigma_model.predict(sample)
    dists = [predictor.distribution(m, s, t)
             for m, s, t in zip(sample["pred_margin"], sigmas, sample["pred_total"])]
    rep = distribution_report(dists, sample["margin"].to_numpy(float))
    cov = rep["coverage"].set_index("level")["coverage"]
    assert abs(cov.loc[0.8] - 0.8) < 0.08
    assert abs(cov.loc[0.5] - 0.5) < 0.09
    assert rep["log_loss"] < 0.69


def test_lattice_prices_key_numbers_better_than_a_normal(trained):
    predictor, oos = trained
    sample = oos.sample(min(600, len(oos)), random_state=4)
    sigmas = predictor.sigma_model.predict(sample)
    y = sample["margin"].to_numpy(float)

    def mean_p(method, k):
        ds = [predictor.distribution(m, s, t, method)
              for m, s, t in zip(sample["pred_margin"], sigmas, sample["pred_total"])]
        return float(np.mean([d.p_exact(k) + d.p_exact(-k) for d in ds]))

    for k in (3, 7):
        actual = float(np.mean(np.abs(y) == k))
        assert abs(mean_p("lattice", k) - actual) < abs(mean_p("plain", k) - actual)


def test_predictor_roundtrip(trained, tmp_path):
    predictor, oos = trained
    predictor.save(tmp_path / "art")
    loaded = Predictor.load(tmp_path / "art")
    assert loaded.model.name == predictor.model.name
    assert loaded.cfg.dist_method == predictor.cfg.dist_method
    a = predictor.distribution(-3.0, 16.0, 54.0)
    b = loaded.distribution(-3.0, 16.0, 54.0)
    np.testing.assert_allclose(a.pmf, b.pmf, atol=1e-12)


def test_predict_slate_shape(trained, features):
    predictor, _ = trained
    block = features[features["season"] == features["season"].max()].head(25)
    slate, dists = predictor.predict_slate(block)
    assert len(slate) == len(dists) == len(block)
    assert slate["p_home_win"].between(0, 1).all()
    bucket_sum = (slate["p_home_by_1_3"] + slate["p_home_by_4_7"]
                  + slate["p_home_by_8_plus"])
    assert (bucket_sum <= slate["p_home_win"] + 1e-9).all()


def test_ats_record_grades_correctly():
    import pandas as pd
    d = pd.DataFrame({
        # model likes home by 10, market says 3, home wins by 14 -> win
        "pred_margin": [10.0, -10.0, 3.0],
        "market_margin": [3.0, 3.0, 3.0],
        "margin": [14.0, -5.0, 3.0],
    })
    r = ats_record(d, threshold=0.0)
    assert r["wins"] == 2 and r["losses"] == 0 and r["pushes"] == 1


def test_crps_rewards_a_sharper_correct_distribution(key_numbers):
    from cfb.distribution.margin import MarginDistribution
    sharp = MarginDistribution.from_continuous(7.0, 10.0, key_numbers=key_numbers)
    vague = MarginDistribution.from_continuous(7.0, 25.0, key_numbers=key_numbers)
    assert crps_discrete(sharp, 7) < crps_discrete(vague, 7)
    wrong = MarginDistribution.from_continuous(-20.0, 10.0, key_numbers=key_numbers)
    assert crps_discrete(wrong, 7) > crps_discrete(sharp, 7)
