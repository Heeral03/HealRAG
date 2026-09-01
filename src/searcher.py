import os
import time
import hashlib
from typing import Dict, List, Optional

class WebSearchCache:
    """
    Lightweight TTL-based cache for web search results.
    Uses exact query hash matching to avoid redundant network calls
    for repeated out-of-corpus queries.
    """

    def __init__(self, ttl_seconds: float = 3600.0, max_entries: int = 500):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._store: Dict[str, Dict] = {}

    def _hash_query(self, query: str) -> str:
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[List[Dict]]:
        key = self._hash_query(query)
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.time() - entry["timestamp"] > self.ttl_seconds:
            del self._store[key]
            return None
        return entry["results"]

    def put(self, query: str, results: List[Dict]):
        if len(self._store) >= self.max_entries:
            oldest_key = min(self._store, key=lambda k: self._store[k]["timestamp"])
            del self._store[oldest_key]
        key = self._hash_query(query)
        self._store[key] = {
            "results": results,
            "timestamp": time.time()
        }

    @property
    def size(self) -> int:
        return len(self._store)


class WebSearcher:
    """
    Handles external web search fallback and query expansion when local retrieval
    is classified as INCORRECT or AMBIGUOUS by the CRAG Evaluator.

    Supports multiple backends (in order of preference):
      1. Tavily API (~200ms, requires TAVILY_API_KEY)
      2. DuckDuckGo (~600ms, no API key required)
      3. Structured mock fallback (offline/demo mode)

    Includes a TTL-based result cache to skip redundant network calls.
    """

    def __init__(self, max_results: int = 3, enable_cache: bool = True):
        self.max_results = max_results

        # Initialize result cache
        self.cache = WebSearchCache(ttl_seconds=3600.0) if enable_cache else None

        # Detect available search backends
        import config
        self._tavily_available = False
        self._tavily_api_key = config.TAVILY_API_KEY.strip()
        if self._tavily_api_key:
            try:
                from tavily import TavilyClient
                self._tavily_available = True
            except ImportError:
                self._tavily_available = False

        self._ddg_available = False
        try:
            from ddgs import DDGS
            self._ddg_available = True
        except ImportError:
            self._ddg_available = False

        # Log active backend
        if self._tavily_available:
            self._active_backend = "tavily"
        elif self._ddg_available:
            self._active_backend = "duckduckgo"
        else:
            self._active_backend = "mock"

    @property
    def active_backend(self) -> str:
        return self._active_backend

    def expand_query(self, query: str) -> str:
        """Expands industry jargon or ambiguous phrases into search-optimized terms."""
        query_clean = query.strip()
        jargon_map = {
            "break-glass": "EHDS Article 7 emergency access vital interests override",
            "hipaa": "GDPR Article 9 EU US health data compliance differences",
            "fhir r4b": "HL7 FHIR Release 4B vs Release 5 specification changes",
        }
        for jargon, expansion in jargon_map.items():
            if jargon in query_clean.lower():
                query_clean += f" ({expansion})"
        return query_clean

    def search(self, query: str) -> List[Dict]:
        """
        Executes external web search with caching.
        Order of operations:
          1. Check cache → return immediately if HIT (~0ms)
          2. Try Tavily API (~200ms) if available
          3. Try DuckDuckGo (~600ms) as fallback
          4. Return structured mock if offline
        """
        # Step 1: Cache lookup
        if self.cache:
            cached = self.cache.get(query)
            if cached is not None:
                for r in cached:
                    r["cache_hit"] = True
                return cached

        expanded_q = self.expand_query(query)
        results = []

        # Step 2: Try Tavily (fastest real backend ~200ms)
        if self._tavily_available:
            results = self._tavily_search(expanded_q)

        # Step 3: Try DuckDuckGo (slower ~600ms)
        if not results and self._ddg_available:
            results = self._ddg_search(expanded_q)

        # Step 4: Mock fallback
        if not results:
            results = self._mock_web_search(expanded_q)

        # Store in cache
        if self.cache and results:
            self.cache.put(query, results)

        return results

    def _tavily_search(self, query: str) -> List[Dict]:
        """Search using Tavily API (~200ms typical latency)."""
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=self._tavily_api_key)
            response = client.search(query=query, max_results=self.max_results)
            results = []
            for item in response.get("results", []):
                results.append({
                    "source": f"Web: {item.get('title', 'Tavily Result')}",
                    "text": f"{item.get('title', '')}: {item.get('content', '')}",
                    "url": item.get("url", ""),
                    "similarity_score": 0.75,
                    "eval_status": "EXTERNAL_SEARCH",
                    "search_backend": "tavily"
                })
            return results
        except Exception as e:
            print(f"[WebSearcher Warning] Tavily search failed ({e}), falling back to DuckDuckGo.")
            return []

    def _ddg_search(self, query: str) -> List[Dict]:
        """Search using DuckDuckGo (~600ms typical latency)."""
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                ddg_res = list(ddgs.text(query, max_results=self.max_results))
                results = []
                for item in ddg_res:
                    results.append({
                        "source": f"Web: {item.get('title', 'DuckDuckGo Result')}",
                        "text": f"{item.get('title', '')}: {item.get('body', '')}",
                        "url": item.get("href", ""),
                        "similarity_score": 0.75,
                        "eval_status": "EXTERNAL_SEARCH",
                        "search_backend": "duckduckgo"
                    })
                return results
        except Exception as e:
            print(f"[WebSearcher Warning] DuckDuckGo search failed ({e}), falling back to structured web mock.")
            return []

    def _mock_web_search(self, query: str) -> List[Dict]:
        """Structured mock web search fallback when offline or package is unavailable."""
        q_lower = query.lower()
        if "hipaa" in q_lower:
            snippet = "US HIPAA Privacy Rule establishes national standards to protect individuals' medical records, enforcing penalty tiers up to $1.5M per year."
        elif "ai act" in q_lower or "2024/1689" in q_lower:
            snippet = "EU Artificial Intelligence Act (Regulation EU 2024/1689) was officially adopted in 2024, classifying AI medical devices as high-risk systems under Annex III."
        elif "break-glass" in q_lower or "article 7" in q_lower:
            snippet = "Under EHDS Chapter II Article 7, break-glass emergency access allows clinicians to override patient restrictions when vital interests or life are threatened."
        else:
            snippet = f"External web reference: General healthcare regulatory index details for query '{query}'."

        return [{
            "source": "WebSearch: Official Regulatory Registry",
            "text": snippet,
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689",
            "similarity_score": 0.75,
            "eval_status": "EXTERNAL_SEARCH",
            "search_backend": "mock"
        }]


if __name__ == "__main__":
    searcher = WebSearcher()
    print(f"Active search backend: {searcher.active_backend}")
    print(f"Cache enabled: {searcher.cache is not None}")

    q = "What are the penalty fine tiers under US HIPAA Privacy Rule?"

    # First call — cache MISS (real network call)
    t0 = time.perf_counter()
    res1 = searcher.search(q)
    t1 = (time.perf_counter() - t0) * 1000
    print(f"\n[1st call - MISS] Latency: {t1:.2f}ms | Backend: {res1[0].get('search_backend', '?')}")

    # Second call — cache HIT (should be ~0ms)
    t0 = time.perf_counter()
    res2 = searcher.search(q)
    t2 = (time.perf_counter() - t0) * 1000
    print(f"[2nd call - HIT]  Latency: {t2:.2f}ms | Cache hit: {res2[0].get('cache_hit', False)}")

    print(f"\nCache size: {searcher.cache.size}")
    print(f"Speedup: {t1/t2:.0f}x" if t2 > 0 else "Instant cache hit")
