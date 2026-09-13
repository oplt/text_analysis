"""Sanitized per-request RAG timing and branch diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from time import perf_counter
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

tracer = trace.get_tracer("backend.rag")


def traced(stage: str):
    """Trace async stages without recording potentially sensitive exception text."""

    def decorate(function):
        @wraps(function)
        async def run(*args, **kwargs):
            with tracer.start_as_current_span(
                stage, record_exception=False, set_status_on_exception=False
            ) as span:
                try:
                    return await function(*args, **kwargs)
                except BaseException as exc:
                    span.set_attribute("error.type", type(exc).__name__)
                    span.set_status(trace.StatusCode.ERROR)
                    raise

        return run

    return decorate


def trace_carrier() -> dict[str, str]:
    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier)
    return carrier


def resume_trace(carrier: dict[str, str]):
    return tracer.start_as_current_span(
        "rag.synthesis.worker",
        context=TraceContextTextMapPropagator().extract(carrier),
        record_exception=False,
        set_status_on_exception=False,
    )


@dataclass(slots=True)
class RagTraceContext:
    request_id: str = field(default_factory=lambda: str(uuid4()))
    stage_timings_ms: dict[str, int] = field(default_factory=dict)
    branch_status: dict[str, str] = field(default_factory=dict)
    _spans: dict = field(default_factory=dict)

    def measure(self, stage: str) -> float:
        self._spans[stage] = tracer.start_span(
            f"rag.{stage}", attributes={"request_id": self.request_id}
        )
        return perf_counter()

    def complete(self, stage: str, started: float, *, status: str = "ok") -> None:
        self.stage_timings_ms[stage] = int((perf_counter() - started) * 1000)
        self.branch_status[stage] = status
        span = self._spans.pop(stage, None)
        if span is not None:
            span.set_attribute("status", status)
            span.end()

    def close(self) -> None:
        for span in self._spans.values():
            span.set_attribute("status", "interrupted")
            span.end()
        self._spans.clear()
