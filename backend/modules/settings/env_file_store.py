from __future__ import annotations

import fcntl
import os
import tempfile
from collections.abc import Callable
from pathlib import Path


class EnvFileStore:
    """Process-safe, atomic updater for a shared dotenv file."""

    def __init__(self, path: Path):
        self.path = path
        self.lock_path = path.with_suffix(f"{path.suffix}.lock")

    def update(
        self,
        updates: dict[str, str],
        *,
        parse_line: Callable[[str], tuple[str, str] | None],
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                self._write_atomic(self._merge(updates, parse_line=parse_line))
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _merge(
        self,
        updates: dict[str, str],
        *,
        parse_line: Callable[[str], tuple[str, str] | None],
    ) -> list[str]:
        existing = self.path.read_text(encoding="utf-8").splitlines() if self.path.exists() else []
        remaining = dict(updates)
        rendered: list[str] = []
        for line in existing:
            parsed = parse_line(line)
            if not parsed:
                rendered.append(line)
                continue
            key, _ = parsed
            if key in updates:
                rendered.append(f"{key}={updates[key]}")
                remaining.pop(key, None)
            else:
                rendered.append(line)
        if remaining and rendered and rendered[-1].strip():
            rendered.append("")
        rendered.extend(f"{key}={value}" for key, value in remaining.items())
        return rendered

    def _write_atomic(self, lines: list[str]) -> None:
        contents = "\n".join(lines).rstrip()
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                temp_file.write(f"{contents}\n" if contents else "")
                temp_file.flush()
                os.fsync(temp_file.fileno())
            if self.path.exists():
                os.chmod(temp_name, self.path.stat().st_mode)
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
