import pytest
import csv


from src.indexers.full_text_indexer import FullTextIndexer
from src.classifiers.classification_pipeline import ClassificationPipeline
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


class TestFullTextIndexer:

    @pytest.fixture(scope="session")
    def temp_csv(self, tmp_path_factory):
        temp_file = tmp_path_factory.mktemp(".data") / "temp.csv"
        with open(temp_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["sample search query", 257])
            writer.writerow(["another sample search query", 357])
        return temp_file
    
    @pytest.fixture(scope="session")
    def empty_temp_csv(self, tmp_path_factory):
        temp_file = tmp_path_factory.mktemp(".data") / "empty_temp.csv"
        open(temp_file, 'w').close()
        return temp_file

    @pytest.fixture
    def full_text_indexer_with_data(self, temp_csv):
        return FullTextIndexer(temp_csv)
    
    @pytest.fixture
    def full_text_indexer_without_data(self, empty_temp_csv):
        return FullTextIndexer(empty_temp_csv)

    def test_full_text_indexer_when_data_has_rows(self, full_text_indexer_with_data):
        expected = {"sample search query": 257, "another sample search query": 357}
        assert full_text_indexer_with_data.index_data() == expected

    def test_full_text_indexer_no_rows(self, full_text_indexer_without_data):
        assert full_text_indexer_without_data.index_data() == {}

    def test_full_text_indexer_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            full_text_indexer = FullTextIndexer("path_to/non_existent_file.csv")
            full_text_indexer.index_data()
