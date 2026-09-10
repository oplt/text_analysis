"""Tests for R feature availability vs worker readiness."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.modules.text_research.infrastructure.r_runtime import capabilities


def test_r_feature_enabled_follows_settings_only() -> None:
    with patch.object(capabilities.settings, "RESEARCH_R_ENABLED", True):
        assert capabilities.r_feature_enabled() is True
    with patch.object(capabilities.settings, "RESEARCH_R_ENABLED", False):
        assert capabilities.r_feature_enabled() is False


def test_publish_and_read_capabilities_roundtrip() -> None:
    store: dict[str, str] = {}

    fake = MagicMock()
    fake.setex.side_effect = lambda key, _ttl, value: store.__setitem__(key, value)
    fake.get.side_effect = lambda key: store.get(key)

    with patch.object(capabilities, "_sync_redis", return_value=fake):
        capabilities.publish_r_worker_capabilities(ready=True, analyses=["frequencies", "dfm"])
        caps = capabilities.get_r_worker_capabilities()

    assert caps is not None
    assert caps["ready"] is True
    assert caps["implementation_version"] == "r-quanteda-1"
    assert "frequencies" in caps["analyses"]
