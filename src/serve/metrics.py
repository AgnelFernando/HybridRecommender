from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency of HTTP requests in seconds",
    ["endpoint", "method"]
)

RECS_GENERATED = Counter(
    "recommendations_served_total",
    "Number of recommendations served",
    ["strategy"]  # e.g., "cf","content","hybrid","coldstart"
)

CACHE_HIT_RATIO = Gauge(
    "cache_hit_ratio",
    "Redis cache hit ratio (0..1)"
)
