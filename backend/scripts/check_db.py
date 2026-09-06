#!/usr/bin/env python3
"""Verify DATABASE_URL from backend settings before running migrations."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))


async def main() -> int:
    try:
        from core.config import settings
    except Exception as exc:
        print(f"Failed to load backend settings: {exc}", file=sys.stderr)
        return 1

    dsn = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

    try:
        import asyncpg

        conn = await asyncpg.connect(dsn)
        await conn.execute("SELECT 1")
        await conn.close()
    except Exception as exc:
        host = dsn.split("@")[-1] if "@" in dsn else dsn
        print("Database connection failed.", file=sys.stderr)
        print(f"  Target: {host}", file=sys.stderr)
        print(f"  Error:  {exc}", file=sys.stderr)
        print(file=sys.stderr)
        print("Update DATABASE_URL in backend/.env to match your Postgres credentials.", file=sys.stderr)
        print("To create the default local user/database, run as the postgres superuser:", file=sys.stderr)
        print("  sudo -u postgres psql -f backend/scripts/setup-local-db.sql", file=sys.stderr)
        return 1

    host = dsn.split("@")[-1] if "@" in dsn else dsn
    print(f"Database connection OK ({host})")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
