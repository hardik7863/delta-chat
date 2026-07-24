"""Structured JSON logging. Every log line is a single JSON object and, when emitted
inside a request, carries a correlation `request_id`. No free-text logs.

The current request id is held in a ContextVar so any module can log with correlation
without threading the id through every function signature.
"""
from __future__ import annotations

import json
import sys
import time
from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(rid: str) -> None:
    _request_id.set(rid)


def get_request_id() -> str:
    return _request_id.get()


def log(level: str, event: str, **fields) -> None:
    rec = {
        "ts": round(time.time(), 3),
        "level": level,
        "request_id": _request_id.get(),
        "event": event,
        **fields,
    }
    sys.stderr.write(json.dumps(rec, default=str) + "\n")
    sys.stderr.flush()


def info(event: str, **f) -> None:
    log("INFO", event, **f)


def warn(event: str, **f) -> None:
    log("WARN", event, **f)


def error(event: str, **f) -> None:
    log("ERROR", event, **f)
