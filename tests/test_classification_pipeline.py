import pytest
from src.matchers.raw_string_matcher import RawStringMatcher


class TestRawStringMatcher:

    @pytest.fixture
    def matches(self):
        return {"sample search query": 235}

    @pytest.fixture
    def raw_string_matcher(self, matches):
        return RawStringMatcher(matches)

    def test_raw_string_matcher(self, raw_string_matcher):
        assert raw_string_matcher.match("sample search query") == 235

    def test_raw_string_matcher_no_match(self, raw_string_matcher):
        assert raw_string_matcher.match("no match") is None

    def test_raw_string_matcher_add_query(self, raw_string_matcher):
        raw_string_matcher.add_query("another sample search query", 357)
        assert raw_string_matcher.match("another sample search query") == 357

    def test_raw_string_matcher_remove_query(self, raw_string_matcher):
        raw_string_matcher.remove_query("sample search query")
        assert raw_string_matcher.match("sample search query") is None
