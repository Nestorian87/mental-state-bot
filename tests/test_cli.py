from __future__ import annotations

from mental_state_bot.cli import _alembic_config_path


def test_alembic_config_path_finds_project_config() -> None:
    path = _alembic_config_path()

    assert path.name == "alembic.ini"
    assert path.exists()
