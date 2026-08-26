"""Small crash-safe checkpoint primitives for long-running analyses."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def format_progress(
    completed: int,
    total: int,
    *,
    label: str = "Progress",
    width: int = 30,
) -> str:
    """Return a dependency-free text progress bar suitable for notebooks."""

    if total < 1:
        raise ValueError("total must be positive")
    if completed < 0 or completed > total:
        raise ValueError("completed must lie between zero and total")
    if width < 1:
        raise ValueError("width must be positive")
    filled = width if completed == total else int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    percentage = 100.0 * completed / total
    return f"{label}: [{bar}] {completed}/{total} ({percentage:5.1f}%)"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    scalar = getattr(value, "item", None)
    if callable(scalar):
        return scalar()
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def config_digest(config: Mapping[str, Any]) -> str:
    """Return a stable short digest for a JSON-compatible run configuration."""

    return hashlib.sha256(_canonical_json(dict(config)).encode("utf-8")).hexdigest()[:16]


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Serialize JSON and atomically replace ``path``."""

    _atomic_replace(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n").encode(
            "utf-8"
        ),
    )


def atomic_write_pickle(path: Path, payload: Any) -> None:
    """Serialize a pickle and atomically replace ``path``."""

    _atomic_replace(path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def load_pickle(path: Path) -> Any:
    """Load a pickle written by :func:`atomic_write_pickle`."""

    with path.open("rb") as handle:
        return pickle.load(handle)


@dataclass(frozen=True)
class JsonUnitCheckpointStore:
    """Persist complete independent work units under a configuration digest.

    A unit becomes resumable only after its JSON file has been atomically
    replaced. Partial or corrupt files are ignored and recomputed.
    """

    output_dir: Path
    namespace: str
    config: Mapping[str, Any]

    @property
    def digest(self) -> str:
        return config_digest(self.config)

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / ".checkpoints" / self.namespace / self.digest

    @property
    def state_path(self) -> Path:
        return self.output_dir / f"{self.namespace}_resume.json"

    def initialize(self, *, resume: bool) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if resume and self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid resume state: {self.state_path}") from error
            if state.get("config") != dict(self.config):
                raise ValueError(
                    "cannot resume with settings that differ from the saved configuration"
                )
        self._write_state(status="running")

    def _unit_path(self, unit_id: str) -> Path:
        unit_digest = hashlib.sha256(unit_id.encode("utf-8")).hexdigest()
        return self.checkpoint_dir / f"{unit_digest}.json"

    def load_rows(self, unit_id: str) -> list[dict[str, Any]] | None:
        path = self._unit_path(unit_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            payload.get("status") != "complete"
            or payload.get("unit_id") != unit_id
            or payload.get("config_digest") != self.digest
            or not isinstance(payload.get("rows"), list)
        ):
            return None
        return [dict(row) for row in payload["rows"]]

    def save_rows(self, unit_id: str, rows: Sequence[Mapping[str, Any]]) -> None:
        atomic_write_json(
            self._unit_path(unit_id),
            {
                "status": "complete",
                "unit_id": unit_id,
                "config_digest": self.digest,
                "rows": [dict(row) for row in rows],
            },
        )

    def finish(self, *, completed_units: int, total_units: int) -> None:
        self._write_state(
            status="complete",
            completed_units=completed_units,
            total_units=total_units,
        )

    def _write_state(
        self,
        *,
        status: str,
        completed_units: int = 0,
        total_units: int = 0,
    ) -> None:
        atomic_write_json(
            self.state_path,
            {
                "status": status,
                "config": dict(self.config),
                "config_digest": self.digest,
                "completed_units": completed_units,
                "total_units": total_units,
            },
        )
