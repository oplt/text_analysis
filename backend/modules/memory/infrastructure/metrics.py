"""Prometheus metrics for the memory module."""

from prometheus_client import Counter, Histogram

memory_add_success = Counter("memory_add_success_total", "Successful memory writes")
memory_add_failure = Counter("memory_add_failure_total", "Failed memory writes")
memory_search_success = Counter("memory_search_success_total", "Successful memory searches")
memory_search_failure = Counter("memory_search_failure_total", "Failed memory searches")
memory_search_latency_ms = Histogram(
    "memory_search_latency_ms",
    "Memory search latency in milliseconds",
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
)
retrieved_memory_count = Counter(
    "retrieved_memory_count_total",
    "Memories returned from recall",
)
rejected_memory_count = Counter(
    "rejected_memory_count_total",
    "Memories rejected by policy",
    ["reason"],
)
mem0_unavailable_count = Counter("mem0_unavailable_count_total", "Mem0 unavailable events")
