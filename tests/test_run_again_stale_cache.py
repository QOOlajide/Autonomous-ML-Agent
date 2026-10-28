"""
Run again must fetch stale agent_history from session_state — not re-run the agent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


@pytest.fixture
def agent_stub(monkeypatch):
    monkeypatch.setenv("AML_AGENT_STUB", "1")
    monkeypatch.setenv("AML_AGENT_STUB_SLEEP", "0.1")
    monkeypatch.setenv("AML_AGENT_STUB_CALLS", "0")
    monkeypatch.setenv("AML_REPLAY_SLEEP", "0")


@pytest.fixture
def app_after_first_run(agent_stub):
    """AppTest after one successful Run agent (history cached in session)."""
    at = AppTest.from_file(APP, default_timeout=10)
    at.run()
    assert not at.exception, "app failed on initial load"

    at.button(key="run_agent_btn").click().run()
    assert not at.exception, "app failed on first Run agent"
    assert at.session_state["agent_run_cached"] is True
    assert os.environ.get("AML_AGENT_STUB_CALLS") == "1"
    return at


def test_first_run_saves_history_with_load_panel_proof(app_after_first_run):
    hist = list(app_after_first_run.session_state["agent_history"])
    assert len(hist) >= 1
    assert hist[0]["call"]["tool"] == "load_panel"
    assert hist[0]["result"].get("ok") is True
    assert hist[0]["result"].get("preview"), "expected proof preview in cache"


def test_run_again_does_not_call_agent_stub_again(app_after_first_run):
    at = app_after_first_run
    assert os.environ.get("AML_AGENT_STUB_CALLS") == "1"

    at.button(key="run_agent_btn").click().run()
    assert not at.exception, "app failed on Run again"
    assert os.environ.get("AML_AGENT_STUB_CALLS") == "1", (
        "agent stub ran again — data was reproduced instead of fetched from session"
    )


def test_run_again_reuses_identical_session_history(app_after_first_run):
    at = app_after_first_run
    hist_after_first = list(at.session_state["agent_history"])
    first_payload = hist_after_first[0]["result"]

    at.button(key="run_agent_btn").click().run()
    assert not at.exception, "app failed on Run again"

    hist_after_second = list(at.session_state["agent_history"])
    assert hist_after_second == hist_after_first, (
        "history changed on Run again — expected identical stale session payload"
    )
    assert hist_after_second[0]["result"] == first_payload
    assert at.session_state["agent_run_cached"] is True


def test_run_again_renders_cached_steps_complete(app_after_first_run):
    at = app_after_first_run
    at.button(key="run_agent_btn").click().run()
    assert not at.exception, "app failed on Run again"

    statuses = list(at.status)
    assert len(statuses) >= 1
    assert any(s.label == "Load panel" for s in statuses)
    assert all(
        s.state == "complete"
        for s in statuses
        if getattr(s, "label", None) in {"Load panel", "Stop"}
    ), "cached steps should render complete, not running/empty"
