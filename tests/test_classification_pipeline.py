import pytest


@pytest.fixture
def matches():
    return {"sample search query": 235}


@pytest.fixture
def raw_string_matcher(matches):
    return RawStringMatcher(matches)


def test_raw_string_matcher(raw_string_matcher):
    assert raw_string_matcher.match("sample search query") == 235