"""Streamlit app tests.

These matter more than usual here: the app is the primary interface for a user
who cannot run anything locally, so a runtime error in it is not a nuisance --
it is the whole product being down, with no terminal to debug from.

``AppTest`` executes the real script in-process and exposes the rendered
elements, so these catch the errors that only appear when Streamlit actually
runs the page top to bottom.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

# AppTest resolves relative paths against the *calling* file, so anchor to the
# repo root explicitly -- otherwise this only works from one directory.
APP = str(Path(__file__).resolve().parent.parent / "cfb" / "dashboard" / "app.py")
TIMEOUT = 120


@pytest.fixture(autouse=True)
def _clear_streamlit_caches():
    """Streamlit's caches outlive an AppTest instance, so state leaks between
    tests unless they are cleared -- the same reason the app bumps a token
    after fetching data or training."""
    import streamlit as st
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def run_app(**session_state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    for k, v in session_state.items():
        at.session_state[k] = v
    return at.run()


def all_text(at: AppTest) -> str:
    parts = []
    for coll in (at.markdown, at.title, at.caption, at.info, at.warning,
                 at.error, at.subheader, at.header, at.text):
        parts.extend(str(e.value) for e in coll)
    return "\n".join(parts)


def test_app_runs_with_no_data_and_forces_setup(monkeypatch, tmp_path):
    """A brand-new deployment must land on Setup, not crash or show a bare page."""
    from cfb.config import get_config

    monkeypatch.setenv("CFB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CFB_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    get_config(refresh=True)
    try:
        at = run_app()
        assert not at.exception, at.exception
        text = all_text(at)
        assert "Setup" in text
        # It must tell a keyless user what to do, not just fail.
        assert "sample data" in text.lower() or "CFBD" in text
        # And it must not pretend there is a model.
        assert not [r for r in at.sidebar.radio if "view" in str(r.label).lower()]
    finally:
        monkeypatch.undo()
        get_config(refresh=True)


def test_app_runs_with_data_and_model(trained_artifact):
    at = run_app()
    assert not at.exception, at.exception
    # With data and a model present the app must unlock the other views.
    assert [r for r in at.sidebar.radio if "view" in str(r.label).lower()]
    assert "CFB Analytics" in all_text(at) or at.sidebar.title


@pytest.mark.parametrize("page", ["Slate", "Game", "Model", "Backtest"])
def test_each_page_renders(trained_artifact, page):
    """Every offline page must render. Live and Kalshi need the network."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, at.exception
    radios = [r for r in at.sidebar.radio if "view" in str(r.label).lower()]
    assert radios, "sidebar view selector missing -- setup may be blocking"
    radios[0].set_value(page).run()
    assert not at.exception, f"{page} raised: {at.exception}"
    assert page in all_text(at)


def test_setup_page_offers_the_actions(trained_artifact):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    radios = [r for r in at.sidebar.radio if "view" in str(r.label).lower()]
    radios[0].set_value("Setup").run()
    assert not at.exception, at.exception
    labels = " | ".join(str(b.label) for b in at.button)
    assert "Load sample data" in labels
    assert "Fetch from CollegeFootballData" in labels
    assert "Train" in labels


def test_fetch_button_disabled_without_a_key(trained_artifact, monkeypatch):
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    radios = [r for r in at.sidebar.radio if "view" in str(r.label).lower()]
    radios[0].set_value("Setup").run()
    fetch = [b for b in at.button if "CollegeFootballData" in str(b.label)]
    assert fetch and fetch[0].disabled, "fetch must be disabled with no API key"


def test_synthetic_data_is_labelled_as_such(trained_artifact):
    """A user must never mistake the sample league for real teams."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    radios = [r for r in at.sidebar.radio if "view" in str(r.label).lower()]
    radios[0].set_value("Setup").run()
    text = all_text(at).lower()
    assert "synthetic" in text
    assert "real team" in text or "fake" in text


def test_game_page_produces_a_distribution(trained_artifact):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    radios = [r for r in at.sidebar.radio if "view" in str(r.label).lower()]
    radios[0].set_value("Game").run()
    assert not at.exception, at.exception
    labels = [str(m.label) for m in at.metric]
    assert any("projected margin" in x for x in labels)
    assert any("fair spread" in x for x in labels)


def test_cloud_entry_point_runs(trained_artifact):
    """streamlit_app.py is the file Streamlit Cloud executes -- if it is broken
    the deployment is down, with no terminal to debug from."""
    entry = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
    at = AppTest.from_file(entry, default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    assert all_text(at)


def test_importing_the_entry_point_has_no_side_effects():
    import importlib
    import sys

    sys.modules.pop("streamlit_app", None)
    mod = importlib.import_module("streamlit_app")
    assert hasattr(mod, "main")


def test_cold_deploy_flow_load_then_train(monkeypatch, tmp_path):
    """The exact first-run click path on a fresh deployment.

    No data, no key -> Setup is forced -> 'Load sample data' -> 'Train' ->
    the other views unlock. This is the sequence a hosted user hits with no
    terminal to fall back on, so it is worth the seconds it costs.
    """
    from cfb.bootstrap import data_status, model_status
    from cfb.config import get_config

    monkeypatch.setenv("CFB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CFB_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    get_config(refresh=True)
    try:
        at = run_app()
        assert not at.exception, at.exception
        assert not [r for r in at.sidebar.radio if "view" in str(r.label).lower()]

        # Shrink the sample so the test stays quick, then load it.
        at = run_app()
        for ni in at.number_input:
            if str(ni.label) == "teams":
                ni.set_value(40)
            if str(ni.label) == "seasons":
                ni.set_value(4)
        at.run()
        [b for b in at.button if b.label == "Load sample data"][0].click().run()
        assert not at.exception, at.exception
        assert data_status().has_data

        at = run_app()
        [b for b in at.button if b.label == "Train"][0].click().run()
        assert not at.exception, at.exception
        assert model_status()["exists"]

        at = run_app()
        assert [r for r in at.sidebar.radio if "view" in str(r.label).lower()], \
            "views should unlock once data and a model exist"
    finally:
        monkeypatch.undo()
        get_config(refresh=True)
