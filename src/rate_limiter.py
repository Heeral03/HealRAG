import time
from typing import Dict, Tuple

class SlidingWindowLimiter:
    """
    Layer 1: General Abuse Protection
    Tracks request timestamps within a sliding window (default 60 seconds) per client.
    """
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = {}

    def is_allowed(self, client_id: str) -> Tuple[bool, int, float]:
        now = time.time()
        client_history = [t for t in self.requests.get(client_id, []) if now - t < self.window_seconds]
        self.requests[client_id] = client_history

        if len(client_history) >= self.max_requests:
            oldest_request = client_history[0]
            retry_after = round(self.window_seconds - (now - oldest_request), 1)
            return False, len(client_history), max(retry_after, 0.1)

        client_history.append(now)
        self.requests[client_id] = client_history
        return True, len(client_history), 0.0


class CostAwareTokenBucket:
    """
    Layer 2: Financial Cost-Aware Protection
    Token Bucket refilling over time based on ESTIMATED TOKEN COST.
    Cheap fast-path queries drain few tokens, while expensive fallback queries drain the bucket faster.
    """
    def __init__(self, capacity: int = 50000, refill_rate_per_sec: float = 83.33): # 5,000 tokens / min = ~83.33 tokens/sec
        self.capacity = capacity
        self.refill_rate = refill_rate_per_sec
        # client_id -> {"tokens": float, "last_updated": float}
        self.buckets: Dict[str, dict] = {}

    def _get_bucket(self, client_id: str) -> dict:
        now = time.time()
        if client_id not in self.buckets:
            self.buckets[client_id] = {
                "tokens": float(self.capacity),
                "last_updated": now
            }
            return self.buckets[client_id]

        bucket = self.buckets[client_id]
        elapsed = now - bucket["last_updated"]
        # Refill tokens up to max capacity
        bucket["tokens"] = min(float(self.capacity), bucket["tokens"] + (elapsed * self.refill_rate))
        bucket["last_updated"] = now
        return bucket

    def can_reserve(self, client_id: str, min_required: int = 500) -> Tuple[bool, float, float]:
        bucket = self._get_bucket(client_id)
        tokens = bucket["tokens"]

        if tokens < min_required:
            deficit = min_required - tokens
            wait_seconds = round(deficit / self.refill_rate, 1)
            return False, round(tokens, 1), max(wait_seconds, 0.1)

        return True, round(tokens, 1), 0.0

    def deduct(self, client_id: str, tokens_used: int) -> float:
        bucket = self._get_bucket(client_id)
        bucket["tokens"] = max(0.0, bucket["tokens"] - tokens_used)
        return round(bucket["tokens"], 1)


class DualLayerRateLimiter:
    """
    Combined Rate Limiter coordinating Layer 1 (Request Count) and Layer 2 (Token Cost).
    """
    def __init__(self, max_req_per_min: int = 10, bucket_capacity: int = 50000, refill_rate_per_min: int = 5000):
        self.layer1 = SlidingWindowLimiter(max_requests=max_req_per_min, window_seconds=60)
        self.layer2 = CostAwareTokenBucket(capacity=bucket_capacity, refill_rate_per_sec=refill_rate_per_min / 60.0)

    def check_pre_request(self, client_id: str, min_token_estimate: int = 500) -> Tuple[bool, str, dict]:
        # Check Layer 1: Request Count
        l1_allowed, req_count, l1_retry = self.layer1.is_allowed(client_id)
        if not l1_allowed:
            return False, f"Layer 1 Rate Limit Exceeded: Max {self.layer1.max_requests} requests/min. Retry in {l1_retry}s.", {
                "layer": 1,
                "retry_after_seconds": l1_retry,
                "current_requests": req_count
            }

        # Check Layer 2: Minimum Token Availability
        l2_allowed, remaining_tokens, l2_wait = self.layer2.can_reserve(client_id, min_token_estimate)
        if not l2_allowed:
            return False, f"Layer 2 Cost Limit Exceeded: Insufficient token budget ({remaining_tokens}/{self.layer2.capacity} remaining). Refilling... Retry in {l2_wait}s.", {
                "layer": 2,
                "retry_after_seconds": l2_wait,
                "remaining_tokens": remaining_tokens,
                "capacity": self.layer2.capacity
            }

        return True, "Allowed", {
            "request_count_60s": req_count,
            "remaining_tokens": remaining_tokens,
            "token_capacity": self.layer2.capacity
        }

    def deduct_post_response(self, client_id: str, tokens_used: int) -> float:
        return self.layer2.deduct(client_id, tokens_used)
