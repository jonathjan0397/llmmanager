"""Config persistence: load, save, and migrate config.toml."""

from __future__ import annotations

import stat
import tomllib
from pathlib import Path
from typing import Callable

import tomli_w
from pydantic import ValidationError

from llmmanager.config.schema import AppConfig
from llmmanager.constants import CONFIG_DIR, CONFIG_FILE
from llmmanager.exceptions import ConfigLoadError, ConfigMigrationError, ConfigSaveError

# ---------------------------------------------------------------------------
# Migration registry
# Each entry migrates from version N -> N+1.
# The function receives the raw dict (already loaded from TOML) and returns
# a modified dict. Pydantic validation runs after all migrations complete.
# ---------------------------------------------------------------------------

_MIGRATIONS: dict[int, Callable[[dict], dict]] = {
    # Example future migration:
    # 1: _migrate_v1_to_v2,
}

CURRENT_VERSION = 1


def _apply_migrations(data: dict) -> dict:
    version = data.get("version", 1)
    while version < CURRENT_VERSION:
        if version not in _MIGRATIONS:
            raise ConfigMigrationError(
                f"No migration path from version {version} to {version + 1}"
            )
        data = _MIGRATIONS[version](data)
        version += 1
        data["version"] = version
    return data


class ConfigManager:
    """Load, validate, save, and migrate the app config file."""

    def __init__(self, config_path: Path = CONFIG_FILE) -> None:
        self._path = config_path
        self._config: AppConfig | None = None

    @property
    def config(self) -> AppConfig:
        if self._config is None:
            raise RuntimeError("ConfigManager.load() has not been called yet.")
        return self._config

    def load(self) -> AppConfig:
        """Load config from disk, applying migrations. Creates defaults if missing."""
        if not self._path.exists():
            self._config = AppConfig()
            self.save()
            return self._config

        try:
            raw = tomllib.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConfigLoadError(f"Failed to parse {self._path}: {exc}") from exc

        try:
            raw = _apply_migrations(raw)
        except ConfigMigrationError:
            raise
        except Exception as exc:
            raise ConfigMigrationError(f"Migration error: {exc}") from exc

        try:
            self._config = AppConfig.model_validate(raw)
        except ValidationError as exc:
            raise ConfigLoadError(f"Config validation failed: {exc}") from exc

        return self._config

    def save(self) -> None:
        """Persist the current config to disk with secure permissions."""
        if self._config is None:
            raise ConfigSaveError("Nothing to save — config has not been loaded.")

        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        raw = self._config.model_dump(mode="json", exclude_none=True)

        try:
            tmp = self._path.with_suffix(".toml.tmp")
            tmp.write_bytes(tomli_w.dumps(raw).encode())
            tmp.replace(self._path)
            # Restrict permissions: owner read/write only (secrets live here)
            self._path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except Exception as exc:
            raise ConfigSaveError(f"Failed to write {self._path}: {exc}") from exc

    def update(self, **kwargs) -> AppConfig:
        """Mutate top-level fields and save. Returns updated config."""
        current = self.config.model_dump()
        current.update(kwargs)
        self._config = AppConfig.model_validate(current)
        self.save()
        return self._config
