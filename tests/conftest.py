import pytest

from closepilot import db
from data.generate import generate


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    generate(seed=42, out=d)
    return d


@pytest.fixture()
def con(raw_dir):
    c = db.connect(":memory:")
    db.init_db(c, raw_dir)
    return c


@pytest.fixture(autouse=True)
def no_network_llm(monkeypatch):
    """Tests never call Gemini, even when a real key is in .env. Tests that need it override this."""
    from closepilot import llm
    monkeypatch.setattr(llm, "available", lambda: False)
