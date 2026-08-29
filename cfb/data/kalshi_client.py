"""Kalshi market-data client (read-only) with RSA-PSS request signing.

Setup (see docs/DATA_SETUP.md):
  1. Kalshi account -> Settings -> API Keys -> "Create New API Key".
  2. Kalshi shows you an *Access Key ID* and downloads a private key .pem once.
  3. Put the ID in KALSHI_ACCESS_KEY and the .pem path in
     KALSHI_PRIVATE_KEY_PATH.

This client never places, modifies, or cancels orders.  It only reads markets
and order books so the model's fair prices can be compared against them.
"""
from __future__ import annotations

import base64
import datetime as dt
import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
import requests

from cfb.config import CONFIG, Config

log = logging.getLogger(__name__)

# Kalshi's fee schedule: ceil(multiplier * C * P * (1-P)) cents.
DEFAULT_FEE_MULTIPLIER = 0.07


@dataclass
class KalshiMarket:
    ticker: str
    event_ticker: str
    series_ticker: str
    title: str
    subtitle: str
    yes_sub_title: str
    strike_type: str | None
    floor_strike: float | None
    cap_strike: float | None
    yes_bid: int | None
    yes_ask: int | None
    no_bid: int | None
    no_ask: int | None
    last_price: int | None
    volume: int | None
    open_interest: int | None
    status: str
    close_time: str | None

    @property
    def mid(self) -> float | None:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return (self.yes_bid + self.yes_ask) / 200.0

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["mid"] = self.mid
        return d


class KalshiClient:
    def __init__(self, cfg: Config | None = None, timeout: float = 30.0):
        self.cfg = cfg or CONFIG
        self.base = self.cfg.kalshi_base.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json",
                                     "User-Agent": "cfb-analytics/0.1"})
        self._private_key = None
        if self.cfg.kalshi_access_key and self.cfg.kalshi_private_key_path:
            self._private_key = self._load_key(self.cfg.kalshi_private_key_path)

    @property
    def authenticated(self) -> bool:
        return self._private_key is not None

    # -- auth -------------------------------------------------------------
    @staticmethod
    def _load_key(path):
        try:
            from cryptography.hazmat.primitives import serialization
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Kalshi signing needs `cryptography`: pip install 'cfb-analytics[kalshi]'"
            ) from exc
        with open(path, "rb") as fh:
            return serialization.load_pem_private_key(fh.read(), password=None)

    def _sign(self, method: str, path: str) -> dict[str, str]:
        """Kalshi signs `timestamp_ms + METHOD + path` with RSA-PSS/SHA256."""
        if not self._private_key:
            return {}
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        ts = str(int(time.time() * 1000))
        # The signed path must include the API prefix but not the query string.
        sign_path = path.split("?")[0]
        msg = f"{ts}{method.upper()}{sign_path}".encode()
        sig = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.cfg.kalshi_access_key or "",
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        from urllib.parse import urlparse

        url = f"{self.base}{path}"
        sign_path = urlparse(self.base).path.rstrip("/") + path
        last: Exception | None = None
        for attempt in range(4):
            try:
                resp = self.session.get(url, params=params,
                                        headers=self._sign("GET", sign_path),
                                        timeout=self.timeout)
                if resp.status_code == 401:
                    raise RuntimeError(
                        f"Kalshi returned 401 for {path}. Check KALSHI_ACCESS_KEY / "
                        "KALSHI_PRIVATE_KEY_PATH, and that the key is active."
                    )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Kalshi GET {path} failed") from last

    # -- reads ------------------------------------------------------------
    def _paged(self, path: str, key: str, params: dict | None = None,
               max_pages: int = 25) -> list[dict]:
        params = dict(params or {})
        params.setdefault("limit", 1000)
        out: list[dict] = []
        cursor = None
        for _ in range(max_pages):
            if cursor:
                params["cursor"] = cursor
            payload = self._get(path, params)
            batch = payload.get(key) or []
            out.extend(batch)
            cursor = payload.get("cursor")
            if not cursor or not batch:
                break
        return out

    def series_list(self, category: str = "Sports") -> pd.DataFrame:
        try:
            rows = self._paged("/series", "series", {"category": category})
        except Exception as exc:  # noqa: BLE001
            log.warning("series list failed (%s); falling back to /events scan", exc)
            rows = []
        return pd.DataFrame(rows)

    def events(self, series_ticker: str | None = None, status: str = "open",
               with_nested_markets: bool = True) -> list[dict]:
        params = {"status": status,
                  "with_nested_markets": str(with_nested_markets).lower()}
        if series_ticker:
            params["series_ticker"] = series_ticker
        return self._paged("/events", "events", params)

    def markets(self, series_ticker: str | None = None, event_ticker: str | None = None,
                status: str = "open") -> list[KalshiMarket]:
        params: dict[str, Any] = {"status": status}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        raw = self._paged("/markets", "markets", params)
        return [self._to_market(m) for m in raw]

    def orderbook(self, ticker: str, depth: int = 10) -> dict:
        return self._get(f"/markets/{ticker}/orderbook", {"depth": depth})

    @staticmethod
    def _to_market(m: dict) -> KalshiMarket:
        return KalshiMarket(
            ticker=m.get("ticker", ""),
            event_ticker=m.get("event_ticker", ""),
            series_ticker=m.get("series_ticker") or m.get("ticker", "").split("-")[0],
            title=m.get("title", ""),
            subtitle=m.get("subtitle", "") or "",
            yes_sub_title=m.get("yes_sub_title", "") or "",
            strike_type=m.get("strike_type"),
            floor_strike=_f(m.get("floor_strike")),
            cap_strike=_f(m.get("cap_strike")),
            yes_bid=_i(m.get("yes_bid")), yes_ask=_i(m.get("yes_ask")),
            no_bid=_i(m.get("no_bid")), no_ask=_i(m.get("no_ask")),
            last_price=_i(m.get("last_price")),
            volume=_i(m.get("volume")), open_interest=_i(m.get("open_interest")),
            status=m.get("status", ""), close_time=m.get("close_time"),
        )

    # -- discovery --------------------------------------------------------
    def discover_football(self, keywords: Iterable[str] = ("football", "ncaa", "cfb",
                                                           "college")) -> pd.DataFrame:
        """Find candidate CFB series/events without hardcoding ticker guesses.

        Kalshi renames and adds series between seasons, so rather than baking in
        a ticker that may be stale, scan open events and match on title text.
        """
        kws = [k.lower() for k in keywords]
        rows = []
        for ev in self.events(status="open"):
            title = f"{ev.get('title','')} {ev.get('sub_title','')}".lower()
            ticker = str(ev.get("event_ticker", ""))
            hay = f"{title} {ticker.lower()}"
            if any(k in hay for k in kws):
                rows.append({
                    "event_ticker": ticker,
                    "series_ticker": ev.get("series_ticker"),
                    "title": ev.get("title"),
                    "sub_title": ev.get("sub_title"),
                    "n_markets": len(ev.get("markets") or []),
                })
        df = pd.DataFrame(rows)
        return df.sort_values("n_markets", ascending=False) if not df.empty else df

    def markets_frame(self, series_ticker: str | None = None,
                      event_ticker: str | None = None,
                      status: str = "open") -> pd.DataFrame:
        mkts = self.markets(series_ticker=series_ticker, event_ticker=event_ticker,
                            status=status)
        if not mkts:
            return pd.DataFrame()
        return pd.DataFrame([m.to_dict() for m in mkts])


def trading_fee_cents(price: float, contracts: int = 1,
                      multiplier: float = DEFAULT_FEE_MULTIPLIER) -> float:
    """Kalshi fee in cents: ceil(multiplier * C * P * (1-P) * 100)."""
    import math
    p = min(max(float(price), 0.0), 1.0)
    return math.ceil(multiplier * contracts * p * (1.0 - p) * 100.0)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
