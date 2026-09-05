"""SQLite-backed job store.

Single writer (the worker thread) plus API readers, so WAL mode plus a short
busy timeout is enough; no ORM.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from .config import get_config

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    type          TEXT NOT NULL,
    status        TEXT NOT NULL,
    model         TEXT NOT NULL,
    quant         TEXT,
    preset        TEXT,
    prompt        TEXT,
    params        TEXT NOT NULL DEFAULT '{}',
    estimated_gb  REAL,
    peak_gb       REAL,
    progress      REAL NOT NULL DEFAULT 0,
    message       TEXT,
    error         TEXT,
    output_path   TEXT,
    log_path      TEXT,
    hold_attempts INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL,
    started_at    REAL,
    finished_at   REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(type);
"""

TERMINAL = {"completed", "failed", "cancelled"}
ACTIVE = {"queued", "held", "running"}

_local = threading.local()


def _connect() -> sqlite3.Connection:
    cfg = get_config()
    path: Path = cfg.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


def conn() -> sqlite3.Connection:
    existing = getattr(_local, "conn", None)
    existing_path = getattr(_local, "path", None)
    current_path = get_config().db_path
    if existing is None or existing_path != current_path:
        if existing is not None:
            existing.close()
        _local.conn = _connect()
        _local.path = current_path
    return _local.conn


def init_db() -> None:
    c = conn()
    with c:
        c.executescript(SCHEMA)


def close_local() -> None:
    existing = getattr(_local, "conn", None)
    if existing is not None:
        existing.close()
        _local.conn = None
        _local.path = None


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    try:
        d["params"] = json.loads(d.get("params") or "{}")
    except json.JSONDecodeError:
        d["params"] = {}
    return d


def create_job(
    *,
    job_type: str,
    model: str,
    quant: str | None,
    preset: str | None,
    prompt: str | None,
    params: dict,
    estimated_gb: float,
) -> str:
    job_id = uuid.uuid4().hex[:12]
    c = conn()
    with c:
        c.execute(
            """INSERT INTO jobs (id, type, status, model, quant, preset, prompt, params,
                                 estimated_gb, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                job_id,
                job_type,
                "queued",
                model,
                quant,
                preset,
                prompt,
                json.dumps(params),
                estimated_gb,
                time.time(),
            ),
        )
    return job_id


def get_job(job_id: str) -> dict | None:
    row = conn().execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_jobs(
    *, status: str | None = None, job_type: str | None = None, model: str | None = None, limit: int = 200
) -> list[dict]:
    sql = "SELECT * FROM jobs"
    clauses: list[str] = []
    args: list[Any] = []
    if status == "active":
        clauses.append(f"status IN ({','.join('?' * len(ACTIVE))})")
        args.extend(sorted(ACTIVE))
    elif status:
        clauses.append("status = ?")
        args.append(status)
    if job_type:
        clauses.append("type = ?")
        args.append(job_type)
    if model:
        clauses.append("model = ?")
        args.append(model)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    return [_row_to_dict(r) for r in conn().execute(sql, args).fetchall()]


def next_queued() -> dict | None:
    """Oldest runnable job. `held` jobs are retried alongside `queued` ones."""
    row = (
        conn()
        .execute(
            "SELECT * FROM jobs WHERE status IN ('queued','held') ORDER BY created_at ASC LIMIT 1"
        )
        .fetchone()
    )
    return _row_to_dict(row) if row else None


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    if "params" in fields and isinstance(fields["params"], dict):
        fields["params"] = json.dumps(fields["params"])
    cols = ", ".join(f"{k} = ?" for k in fields)
    c = conn()
    with c:
        c.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))


def delete_job(job_id: str) -> bool:
    c = conn()
    with c:
        cur = c.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    return cur.rowcount > 0


def counts_by_status() -> dict[str, int]:
    rows = conn().execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
    return {r["status"]: r["n"] for r in rows}


def reset_interrupted() -> Iterable[str]:
    """A job marked `running` at startup means the service died mid-render."""
    rows = conn().execute("SELECT id FROM jobs WHERE status = 'running'").fetchall()
    ids = [r["id"] for r in rows]
    if ids:
        c = conn()
        with c:
            c.execute(
                "UPDATE jobs SET status='failed', error=?, finished_at=? WHERE status='running'",
                ("Service restarted while this job was running.", time.time()),
            )
    return ids
