import pytest
from src.classification_pipeline import ClassificationPipeline
from src.matchers.raw_string_matcher import RawStringMatcher


@pytest.fixture
def query_index():
    return {"sample search query": 235}


@pytest.fixture
def raw_string_matcher(query_index):
    return RawStringMatcher(query_index)


@pytest.fixture
def classification_pipeline(query_index):
    return ClassificationPipeline(query_index)


class TestClassificationPipeline:

    def test_classification_pipeline_returns_none_when_query_is_none(self, classification_pipeline):
        assert classification_pipeline.classify(None) is None        


class TestRawStringMatcher:

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

    def test_raw_string_matcher_returns_none_when_query_is_none(self, raw_string_matcher):
        assert raw_string_matcher.match(None) is None
