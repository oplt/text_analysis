from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


def _attrs(kwargs: dict[str, Any]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        attr = key.replace("_", ".")
        if isinstance(value, str | bool | int | float):
            attrs[attr] = value
        else:
            attrs[attr] = str(value)
    return attrs


@contextmanager
def observed_span(name: str, **attributes: Any) -> Iterator[Any]:
    """Create a span, attach safe attrs, record failures, then re-raise."""

    try:
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode
    except Exception:
        yield None
        return

    tracer = trace.get_tracer("generic_app.observability")
    with tracer.start_as_current_span(name, attributes=_attrs(attributes)) as span:
        started = time.monotonic()
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise
        finally:
            span.set_attribute("latency_ms", (time.monotonic() - started) * 1000.0)


def structured_error(
    logger: logging.Logger,
    message: str,
    exc: BaseException,
    **fields: Any,
) -> None:
    extra = {
        key: value
        for key, value in fields.items()
        if value is not None and isinstance(value, str | bool | int | float)
    }
    extra["error_type"] = type(exc).__name__
    extra["error_message"] = str(exc)[:500]
    logger.error(message, extra=extra, exc_info=True)
