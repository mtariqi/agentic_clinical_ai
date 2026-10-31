"""Tamper-evident, append-only audit trail (ALCOA++ by design).

Each entry is a JSON line carrying who did what, when, under which policy
version, and the SHA-256 hash of the previous entry. Changing, deleting or
reordering any line breaks the chain, which ``verify_chain`` detects.

    Attributable   -> ``actor`` (agent identity, system, or named human)
    Legible        -> plain JSON lines
    Contemporaneous-> UTC timestamp written at the moment of the event
    Original       -> append-only; entries are never rewritten
    Accurate       -> payload stores inputs/evidence, not just conclusions
    + Complete, Consistent, Enduring, Available -> hash chain + verification
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _entry_hash(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(_canonical(body).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path, run_id: str, policy_version: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.policy_version = policy_version
        self._seq, self._prev = self._tail()

    def _tail(self) -> tuple[int, str]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return 0, GENESIS
        last = None
        with open(self.path) as fh:
            for line in fh:
                if line.strip():
                    last = json.loads(line)
        return (last["seq"], last["hash"]) if last else (0, GENESIS)

    def record(self, actor: str, event: str, payload: dict | None = None) -> dict:
        self._seq += 1
        entry = {
            "seq": self._seq,
            "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "run_id": self.run_id,
            "policy_version": self.policy_version,
            "actor": actor,
            "event": event,
            "payload": payload or {},
            "prev_hash": self._prev,
        }
        entry["hash"] = _entry_hash(entry)
        with open(self.path, "a") as fh:
            fh.write(_canonical(entry) + "\n")
        self._prev = entry["hash"]
        return entry


def read_log(path: str | Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def verify_chain(path: str | Path) -> dict:
    """Return {'valid': bool, 'entries': n, 'first_invalid_seq': int | None, 'reason': str}."""
    prev = GENESIS
    entries = read_log(path)
    for i, e in enumerate(entries, start=1):
        if e.get("seq") != i:
            return {"valid": False, "entries": len(entries), "first_invalid_seq": i,
                    "reason": "sequence gap or reordering"}
        if e.get("prev_hash") != prev:
            return {"valid": False, "entries": len(entries), "first_invalid_seq": i,
                    "reason": "broken link to previous entry"}
        if _entry_hash(e) != e.get("hash"):
            return {"valid": False, "entries": len(entries), "first_invalid_seq": i,
                    "reason": "entry content altered"}
        prev = e["hash"]
    return {"valid": True, "entries": len(entries), "first_invalid_seq": None, "reason": "ok"}
