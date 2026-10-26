"""AppTest: Run agent → Run again only after stubbed agent finishes."""

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


def test_button_label_flips_only_after_run(agent_stub):
    at = AppTest.from_file(APP, default_timeout=10)
    at.run()
    assert not at.exception

    btn = at.button(key="run_agent_btn")
    assert btn.label == "Run agent"

    btn.click().run()
    assert not at.exception
    assert at.button(key="run_agent_btn").label == "Run again"
    assert os.environ.get("AML_AGENT_STUB") == "1"


def test_run_again_uses_cache_without_second_stub_call(agent_stub):
    at = AppTest.from_file(APP, default_timeout=10)
    at.run()
    assert not at.exception

    at.button(key="run_agent_btn").click().run()
    assert not at.exception
    assert os.environ.get("AML_AGENT_STUB_CALLS") == "1"
    assert at.button(key="run_agent_btn").label == "Run again"

    at.button(key="run_agent_btn").click().run()
    assert not at.exception
    assert os.environ.get("AML_AGENT_STUB_CALLS") == "1"
