import pytest
import unittest.mock
import csv


from src.data_handlers.labelled_data_reader import LabelledDataReader
from src.classifiers.classification_pipeline import ClassificationPipeline
from src.matchers.raw_string_matcher import RawStringMatcher
from src.app_controllers.predictor import Predictor


@pytest.fixture
def query_index():
    return {"sample search query": 235, "sample search query 2": 237}


@pytest.fixture
def raw_string_matcher(query_index):
    return RawStringMatcher(query_index)


@pytest.fixture
def classification_pipeline(query_index):
    return ClassificationPipeline(query_index)


class TestClassificationPipeline:

    def test_classification_pipeline_returns_category_when_query_exists_in_index(self, classification_pipeline):
        assert classification_pipeline.classify("sample search query") == 235
    
    def test_classification_pipeline_returns_none_when_query_does_not_exist_in_index(self, classification_pipeline):
        assert classification_pipeline.classify("no match") is None
    
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


@pytest.fixture(scope="session")
def temp_csv(tmp_path_factory):
    temp_file = tmp_path_factory.mktemp(".data") / "temp.csv"
    with open(temp_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sample search query", 257])
        writer.writerow(["another sample search query", 357])
    return temp_file

@pytest.fixture(scope="session")
def empty_temp_csv(tmp_path_factory):
    temp_file = tmp_path_factory.mktemp(".data") / "empty_temp.csv"
    open(temp_file, 'w').close()
    return temp_file

@pytest.fixture
def data_reader(temp_csv):
    return LabelledDataReader(temp_csv)

@pytest.fixture
def full_text_indexer_without_data(empty_temp_csv):
    return LabelledDataReader(empty_temp_csv)


class TestLabelDataSetReader:


    def test_full_text_indexer_when_data_has_rows(self, data_reader):
        expected = {"sample search query": 257, "another sample search query": 357}
        assert data_reader.read_labelled_data() == expected

    def test_full_text_indexer_no_rows(self, full_text_indexer_without_data):
        assert full_text_indexer_without_data.read_labelled_data() == {}

    def test_full_text_indexer_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            labelled_data_reader = LabelledDataReader("path_to/non_existent_file.csv")
            labelled_data_reader.read_labelled_data()


@pytest.fixture
def sample_validation_data():
    return {'query1': 0, 'query2': 1, 'query3': 3}


@pytest.fixture
def sample_train_data():
    return {'query1': 0, 'query2': 1, 'query3': 3, 'query4': 2}


@pytest.fixture
def predictor():
    return Predictor()


@pytest.fixture
def predictor_with_validation_data(sample_validation_data):
    predictor = Predictor(validate=True)
    predictor.data = sample_validation_data
    return predictor


@pytest.fixture
def classification_pipeline_mock():
    mock = unittest.mock.MagicMock()
    mock.classify.side_effect = lambda query: int(query[-1])
    return mock


class TestPredictor:


    @pytest.mark.focus
    def test_predict_basic_functionality(self, predictor, classification_pipeline_mock):
        predictor.classification_pipeline = classification_pipeline_mock
        predictor.data = {'query1': 0, 'query2': 1, 'query3': 2}
        
        predictions = predictor.predict()
        
        expected_predictions = {'query1': 1, 'query2': 2, 'query3': 3}
        assert predictions == expected_predictions


    def test_predict_with_validate(self, predictor_with_validation_data, classification_pipeline_mock):
        predictor_with_validation_data.classification_pipeline = classification_pipeline_mock
        
        predictions = predictor_with_validation_data.predict()
        
        expected_predictions = {'query1': 1, 'query2': 2, 'query3': 3}
        assert predictions == expected_predictions
        assert predictor_with_validation_data.scorer is not None
        assert predictor_with_validation_data.scorer.f1_score is not None


    def test_predict_no_data(self, predictor, classification_pipeline_mock):
        predictor.classification_pipeline = classification_pipeline_mock
        predictor.data = {}
        
        predictions = predictor.predict()
        
        assert predictions == {}


    def test_predict_invalid_category(self, predictor, classification_pipeline_mock):
        predictor.classification_pipeline = classification_pipeline_mock
        predictor.data = {'query1': 0, 'query2': 1, 'query3': 2}
        
        # Make the classify function return None for query2
        predictor.classification_pipeline.classify.side_effect = lambda query: None if query == "query2" else int(query[-1])
        
        predictions = predictor.predict()
        
        expected_predictions = {'query1': 1, 'query2': -1, 'query3': 3}
        assert predictions == expected_predictions
