from prometheus_client import Counter, Histogram

ai_run_latency_ms = Histogram(
    "ai_run_latency_ms",
    "AI provider generation latency in milliseconds",
    buckets=(50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000),
)
ai_run_failed_total = Counter("ai_run_failed_total", "AI provider generation failures")
agent_run_latency_ms = Histogram(
    "agent_run_latency_ms",
    "End-to-end agent run latency in milliseconds",
    buckets=(100, 250, 500, 1000, 2500, 5000, 10000, 30000),
)
agent_context_degraded_total = Counter(
    "agent_context_degraded_total",
    "Agent context degradation events",
    labelnames=("source",),
)
