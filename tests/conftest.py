from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensures a developer's shell SQUELCHGRIP_API_KEY never breaks unauthenticated test clients."""
    monkeypatch.delenv("SQUELCHGRIP_API_KEY", raising=False)
