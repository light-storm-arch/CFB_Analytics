"""Shared HTTP session with retry/backoff and a simple on-disk response cache."""
from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class HttpError(RuntimeError):
    def __init__(self, status: int, url: str, body: str = ""):
        self.status = status
        self.url = url
        self.body = body[:500]
        super().__init__(f"HTTP {status} for {url}: {self.body}")


class JsonClient:
    """Thin JSON GET client: retries, backoff, optional disk cache."""

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        cache_dir: Path | None = None,
        max_retries: int = 5,
        timeout: float = 45.0,
        min_interval: float = 0.0,
        user_agent: str = "cfb-analytics/0.1",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = min_interval
        self._last_call = 0.0
        # Guards the rate limiter so concurrent workers still share one global
        # request rate. Threads overlap network *latency*, they do not raise the
        # rate we hit the API at -- CFBD is a small free service.
        self._rate_lock = threading.Lock()
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        if headers:
            self.session.headers.update(headers)

    # -- cache -------------------------------------------------------------
    def _cache_path(self, path: str, params: dict[str, Any] | None) -> Path | None:
        if not self.cache_dir:
            return None
        key = json.dumps({"u": f"{self.base_url}{path}", "p": params or {}}, sort_keys=True)
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        safe = path.strip("/").replace("/", "_") or "root"
        return self.cache_dir / f"{safe}.{digest}.json"

    def _throttle(self) -> None:
        """Block until at least ``min_interval`` has passed since the last start."""
        if not self.min_interval:
            return
        with self._rate_lock:
            delta = time.monotonic() - self._last_call
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last_call = time.monotonic()

    # -- request -----------------------------------------------------------
    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        use_cache: bool = True,
        cache_ttl: float | None = None,
    ) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        cpath = self._cache_path(path, params) if use_cache else None
        if cpath and cpath.exists():
            fresh = cache_ttl is None or (time.time() - cpath.stat().st_mtime) < cache_ttl
            if fresh:
                try:
                    return json.loads(cpath.read_text())
                except json.JSONDecodeError:
                    cpath.unlink(missing_ok=True)

        url = f"{self.base_url}/{path.lstrip('/')}"
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code in RETRY_STATUS:
                    raise HttpError(resp.status_code, url, resp.text)
                if resp.status_code >= 400:
                    raise HttpError(resp.status_code, url, resp.text)
                data = resp.json()
                if cpath:
                    cpath.write_text(json.dumps(data))
                return data
            except (requests.RequestException, HttpError, ValueError) as exc:
                last_exc = exc
                status = getattr(exc, "status", None)
                if status is not None and status not in RETRY_STATUS:
                    raise
                sleep = min(2 ** attempt, 30) + random.random()
                log.warning("GET %s failed (%s); retry %d/%d in %.1fs",
                            url, exc, attempt + 1, self.max_retries, sleep)
                time.sleep(sleep)
        raise RuntimeError(f"GET {url} failed after {self.max_retries} attempts") from last_exc
