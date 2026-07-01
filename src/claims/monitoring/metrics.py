import time
from datetime import datetime, timezone
from typing import Any

METRICS: dict[str, Any] = {
    "predictions_total": 0,
    "anomalies_flagged": 0,
    "batch_predictions_total": 0,
    "nlp_analyses_total": 0,
    "total_processing_ms": 0.0,
    "startup_time": None,
}

_START_TIME = time.time()

def increment(key: str, amount: float = 1) -> None:
    """Add amount to METRICS[key] if key exists."""
    if key in METRICS:
        if isinstance(METRICS[key], float):
            METRICS[key] += amount
        else:
            METRICS[key] += int(amount)

def get_all() -> dict[str, Any]:
    """Return a copy of METRICS plus computed fields like avg_processing_ms and uptime_seconds."""
    res = METRICS.copy()
    
    # Compute average processing time
    pred_total = res.get("predictions_total", 0)
    total_ms = res.get("total_processing_ms", 0.0)
    res["avg_processing_ms"] = round(total_ms / pred_total, 2) if pred_total > 0 else 0.0
    
    # Compute uptime
    res["uptime_seconds"] = int(time.time() - _START_TIME)
    
    st = res.get("startup_time")
    if isinstance(st, datetime):
        res["startup_time_iso"] = st.isoformat()
    else:
        res["startup_time_iso"] = None
        
    return res

def record_startup() -> None:
    """Set METRICS['startup_time'] to datetime.now(timezone.utc) (utcnow replacement)."""
    METRICS["startup_time"] = datetime.now(timezone.utc)
