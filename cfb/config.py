"""Environment + path configuration.

Reads a .env file at the repo root if present (no external dependency), then
falls back to real environment variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None, override: bool = False) -> dict[str, str]:
    """Minimal .env parser. Returns the values it set."""
    path = path or (REPO_ROOT / ".env")
    found: dict[str, str] = {}
    if not path.exists():
        return found
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if not val:
            continue
        if override or key not in os.environ:
            os.environ[key] = val
        found[key] = val
    return found


def load_streamlit_secrets() -> dict[str, str]:
    """Copy Streamlit Cloud secrets into the environment.

    On Streamlit Community Cloud there is no ``.env`` -- credentials live in the
    app's Secrets manager and arrive as ``st.secrets``.  Reading them here means
    every other module keeps using ``os.environ`` and neither knows nor cares
    where the app is running.

    Deliberately defensive: ``streamlit`` is an optional dependency, and
    ``st.secrets`` raises rather than returning empty when no secrets file
    exists, so both cases must be swallowed.
    """
    found: dict[str, str] = {}
    try:
        import streamlit as st  # noqa: PLC0415 - optional, and only on Cloud
    except ImportError:
        return found
    try:
        secrets = dict(st.secrets)
    except Exception:  # noqa: BLE001 - no secrets configured is normal
        return found
    for key in ("CFBD_API_KEY", "CFBD_API_BASE", "KALSHI_ACCESS_KEY",
                "KALSHI_PRIVATE_KEY", "KALSHI_PRIVATE_KEY_PATH", "KALSHI_API_BASE",
                "CFB_DATA_DIR", "CFB_ARTIFACTS_DIR"):
        val = secrets.get(key)
        if val and not os.environ.get(key):
            os.environ[key] = str(val)
            found[key] = str(val)
    # Kalshi hands you a .pem file; pasting its contents into secrets is the
    # only option on a host with no filesystem you control, so materialise it.
    pem = secrets.get("KALSHI_PRIVATE_KEY")
    if pem and not os.environ.get("KALSHI_PRIVATE_KEY_PATH"):
        try:
            import tempfile
            path = Path(tempfile.gettempdir()) / "kalshi_private_key.pem"
            path.write_text(str(pem))
            path.chmod(0o600)
            os.environ["KALSHI_PRIVATE_KEY_PATH"] = str(path)
            found["KALSHI_PRIVATE_KEY_PATH"] = str(path)
        except OSError:
            pass
    return found


load_dotenv()
load_streamlit_secrets()


def _path_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


@dataclass(frozen=True)
class Config:
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    cfbd_api_key: str | None
    cfbd_base: str
    espn_base: str
    kalshi_base: str
    kalshi_access_key: str | None
    kalshi_private_key_path: Path | None

    @classmethod
    def load(cls) -> "Config":
        data_dir = _path_env("CFB_DATA_DIR", REPO_ROOT / "data")
        pk = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        return cls(
            data_dir=data_dir,
            raw_dir=data_dir / "raw",
            processed_dir=data_dir / "processed",
            artifacts_dir=_path_env("CFB_ARTIFACTS_DIR", REPO_ROOT / "artifacts"),
            cfbd_api_key=os.environ.get("CFBD_API_KEY") or None,
            cfbd_base=os.environ.get("CFBD_API_BASE", "https://api.collegefootballdata.com"),
            espn_base=os.environ.get(
                "ESPN_API_BASE",
                "https://site.api.espn.com/apis/site/v2/sports/football/college-football",
            ),
            kalshi_base=os.environ.get(
                "KALSHI_API_BASE", "https://api.elections.kalshi.com/trade-api/v2"
            ),
            kalshi_access_key=os.environ.get("KALSHI_ACCESS_KEY") or None,
            kalshi_private_key_path=Path(pk).expanduser() if pk else None,
        )

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.artifacts_dir):
            d.mkdir(parents=True, exist_ok=True)


CONFIG = Config.load()


def get_config(refresh: bool = False) -> Config:
    global CONFIG
    if refresh:
        load_dotenv(override=True)
        load_streamlit_secrets()
        CONFIG = Config.load()
        try:
            from cfb.data.teams import load_aliases
            load_aliases.cache_clear()   # keyed on the data dir, which may have moved
        except ImportError:
            pass
    return CONFIG
