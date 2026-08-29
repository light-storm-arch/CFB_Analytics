import pytest

from cfb.data.teams import TeamMatcher, normalize

CANON = ["Alabama", "Miami", "Miami (OH)", "Ohio State", "Ohio", "NC State",
         "Texas A&M", "Southern Mississippi", "Mississippi", "Appalachian State",
         "San Jose State", "Louisiana", "Louisiana Monroe"]


@pytest.fixture(scope="module")
def matcher():
    return TeamMatcher(CANON)


@pytest.mark.parametrize("raw,expected", [
    ("Alabama Crimson Tide", "Alabama"),
    ("Miami (FL)", "Miami"),
    ("Miami OH", "Miami (OH)"),
    ("Ohio St", "Ohio State"),
    ("Ohio State Buckeyes", "Ohio State"),
    ("North Carolina State", "NC State"),
    ("Texas A&M Aggies", "Texas A&M"),
    ("Ole Miss", "Mississippi"),
    ("Southern Miss", "Southern Mississippi"),
    ("App State", "Appalachian State"),
])
def test_known_aliases(matcher, raw, expected):
    assert matcher.match(raw) == expected


def test_ambiguous_names_are_kept_distinct(matcher):
    """Ohio and Ohio State are different programmes; so are the two Miamis."""
    assert matcher.match("Ohio") == "Ohio"
    assert matcher.match("Ohio State") == "Ohio State"
    assert matcher.match("Miami (OH)") != matcher.match("Miami (FL)")


def test_unknown_team_returns_none_rather_than_guessing(matcher):
    assert matcher.match("Nonexistent Tech") is None
    assert matcher.match("") is None
    assert matcher.match(None) is None


def test_normalize_is_idempotent():
    for name in ("Texas A&M", "Miami (FL)", "  Ohio  State "):
        once = normalize(name)
        assert normalize(once) == once
