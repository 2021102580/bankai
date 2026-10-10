
from __future__ import annotations

import contextvars
import datetime as dt
import json
import logging
import sys
import uuid
from typing import Any

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "bankai_request_id", default=None
)
user_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "bankai_user_id", default=None
)

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = request_id_var.get()
        if rid:
            payload["request_id"] = rid
        uid = user_id_var.get()
        if uid:
            payload["user_id"] = uid
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_var.get()
        prefix = f"[{rid[:8]}] " if rid else ""
        base = super().format(record)
        return f"{prefix}{base}"


def configure_logging(
    level: str = "INFO",
    fmt: str = "json",
    *,
    stream: Any | None = None,
) -> None:

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("uvicorn.access").handlers = []  # route through our formatter
    logging.getLogger("uvicorn.access").propagate = True
    logging.getLogger("lightgbm").setLevel(logging.WARNING)
    logging.getLogger("sklearn").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def new_request_id() -> str:
    return uuid.uuid4().hex


def bind_request_id(value: str | None = None) -> str:
    rid = value or new_request_id()
    request_id_var.set(rid)
    return rid


def bind_user_id(value: str | None) -> None:
    user_id_var.set(value)
