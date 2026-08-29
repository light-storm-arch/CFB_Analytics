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


load_dotenv()


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
        CONFIG = Config.load()
    return CONFIG
