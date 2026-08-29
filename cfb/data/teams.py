"""Team-name normalisation and matching.

Three sources, three naming conventions.  CFBD says "Miami", ESPN says
"Miami (FL)" or just "Hurricanes", and Kalshi writes whatever fits in a market
title.  Every live workflow dies on this if it is not handled deliberately, so
matching is: exact -> normalised -> alias table -> fuzzy, with a similarity
floor and an explicit ``None`` when nothing is confident enough.  Silently
matching the wrong team is far worse than reporting no match.

Extend ``aliases.json`` in your data directory to teach it new mappings; it is
merged over the built-ins.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
from functools import lru_cache
from pathlib import Path

from cfb.config import CONFIG

log = logging.getLogger(__name__)

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_SPACE = re.compile(r"\s+")

# Words that carry no identifying information once normalised.
STOPWORDS = {"university", "of", "the", "college", "state"}
# "state" is deliberately kept in KEEP_STATE cases -- Ohio State vs Ohio is a
# different team, so we never drop it; it is listed above only for reference.
STOPWORDS = {"university", "of", "the", "college"}

BUILTIN_ALIASES: dict[str, str] = {
    "ole miss": "Mississippi",
    "miami fl": "Miami",
    "miami florida": "Miami",
    "miami oh": "Miami (OH)",
    "miami ohio": "Miami (OH)",
    "pitt": "Pittsburgh",
    "uconn": "Connecticut",
    "umass": "Massachusetts",
    "usc": "USC",
    "ucf": "UCF",
    "smu": "SMU",
    "tcu": "TCU",
    "byu": "BYU",
    "lsu": "LSU",
    "utep": "UTEP",
    "utsa": "UTSA",
    "unlv": "UNLV",
    "fiu": "Florida International",
    "fau": "Florida Atlantic",
    "usf": "South Florida",
    "san jose st": "San JosÃ© State",
    "hawaii": "Hawai'i",
    "louisiana": "Louisiana",
    "ul monroe": "Louisiana Monroe",
    "ul lafayette": "Louisiana",
    "southern miss": "Southern Mississippi",
    "app state": "Appalachian State",
    "nc state": "NC State",
    "north carolina state": "NC State",
    "n c state": "NC State",
    "texas am": "Texas A&M",
    "texas a m": "Texas A&M",
    "sam houston": "Sam Houston State",
    "army west point": "Army",
    "st francis": "St. Francis",
}

ABBREV_EXPANSIONS = {
    " st": " State", " univ": " University",
}


def normalize(name: str | None) -> str:
    """Lowercase, strip punctuation and noise words, collapse whitespace."""
    if not name:
        return ""
    s = str(name).lower().strip()
    s = s.replace("&", " and ")
    s = _PUNCT.sub(" ", s)
    tokens = [t for t in _SPACE.sub(" ", s).split() if t not in STOPWORDS]
    # "st" is ambiguous (State / Saint); expand to "state" which is far commoner.
    tokens = ["state" if t == "st" else t for t in tokens]
    return " ".join(tokens)


@lru_cache(maxsize=1)
def load_aliases() -> dict[str, str]:
    aliases = dict(BUILTIN_ALIASES)
    path = CONFIG.data_dir / "aliases.json"
    if path.exists():
        try:
            user = json.loads(path.read_text())
            aliases.update({normalize(k): v for k, v in user.items()})
            log.info("loaded %d team aliases from %s", len(user), path)
        except json.JSONDecodeError:
            log.warning("could not parse %s; ignoring", path)
    return {normalize(k): v for k, v in aliases.items()}


def save_alias(source_name: str, canonical: str) -> Path:
    """Persist a learned mapping so the next run does not have to guess."""
    path = CONFIG.data_dir / "aliases.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            data = {}
    data[source_name] = canonical
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    load_aliases.cache_clear()
    return path


class TeamMatcher:
    """Match arbitrary team strings onto a canonical roster."""

    def __init__(self, canonical: list[str], cutoff: float = 0.82):
        self.canonical = list(dict.fromkeys(c for c in canonical if c))
        self.cutoff = cutoff
        self._by_norm = {normalize(c): c for c in self.canonical}
        self._aliases = load_aliases()

    def match(self, name: str | None) -> str | None:
        if not name:
            return None
        if name in self._by_norm.values():
            return name
        n = normalize(name)
        if n in self._by_norm:
            return self._by_norm[n]
        alias = self._aliases.get(n)
        if alias:
            return self._by_norm.get(normalize(alias), alias)
        # Drop a trailing mascot word ("Alabama Crimson Tide" -> "Alabama").
        parts = n.split()
        for cut in range(len(parts) - 1, 0, -1):
            cand = " ".join(parts[:cut])
            if cand in self._by_norm:
                return self._by_norm[cand]
        close = difflib.get_close_matches(n, list(self._by_norm), n=1, cutoff=self.cutoff)
        if close:
            return self._by_norm[close[0]]
        return None

    def match_all(self, names) -> dict[str, str | None]:
        return {n: self.match(n) for n in names}

    def unmatched(self, names) -> list[str]:
        return sorted({n for n in names if self.match(n) is None and n})
