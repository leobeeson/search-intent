import os
import re
import time


from sklearn.metrics import f1_score


from src.classifiers.classification_pipeline import ClassificationPipeline
from src.data_handlers.labelled_data_reader import LabelledDataReader
from src.data_handlers.unlabelled_data_reader import UnlabelledDataReader
from src.loggers.log_utils import setup_logger


logger = setup_logger(__name__)


class SymbolicClassifier:

    def __init__(self, data_filepath: str = None, validate: bool = False) -> None:
        logger.info("Instantiating Predictor...")
        self.validate: bool = validate
        self.data_filepath: str = data_filepath
        self.data: dict[str, int] = self._load_data()
        self.classification_pipeline: ClassificationPipeline = ClassificationPipeline(self._load_train_data())
        self.predictions: dict[str, int] = {}
        self.scorer: Scorer = None
        self.f1_score: float = None


    def predict(self, continuous: bool = True, discreet: bool = False) -> int:
        logger.info("Predicting queries...")
        for query in self.data:
            category: int = self.predict_query(query)
            self.predictions[query] = category
        if self.validate:
            f1_score: float = self.score()
            logger.info(f"F1 score: {f1_score}")
        self._save_predictions()


    def predict_query(self, query: str) -> int:
        category: int = self.classification_pipeline.classify(query)
        return category


    def score(self) -> float:
        self.scorer: Scorer = Scorer(self.predictions, self.data)
        f1_score: float = self.scorer.calculate_f1_score()
        self.f1_score: float = f1_score
        return f1_score


    def _load_data(self) -> dict[str, int]:
        if self.validate or self.data_filepath is None:
            logger.warning("No data filepath provided. Loading validation data...")
            self.data_filepath: str = os.environ["PATH_LABELLED_TEST_DATA"]
            data_reader: LabelledDataReader = LabelledDataReader(self.data_filepath)    
        else:
            logger.info("Loading unlabelled test data...")
            data_reader: UnlabelledDataReader = UnlabelledDataReader(self.data_filepath)
        labelled_data: dict[str, int] = data_reader.read_data()
        return labelled_data


    def _load_train_data(self) -> dict[str, int]:
        data_filepath: str = os.environ["PATH_LABELLED_TRAIN_DATA"]
        data_reader: LabelledDataReader = LabelledDataReader(data_filepath)
        labelled_train_data: dict[str, int] = data_reader.read_data()
        return labelled_train_data
    

    def _save_predictions(self):
        filename: str = self._filename_to_save_string(self.data_filepath)
        filepath: str = os.path.join(os.environ["PATH_PREDICTIONS"], filename)
        logger.info(f"Saving predictions to {filepath}")
        try:
            with open(filepath, "w") as csv_file:
                for query in self.predictions:
                    category: int = self.predictions[query]
                    csv_file.write(f"{query},{category}\n")
            logger.info(f"Predictions saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save predictions. Exception: {e}")
         
    
    def _filename_to_save_string(self, filepath):
        base_name = os.path.basename(filepath)
        name_without_extension, _ = os.path.splitext(base_name)
        save_string = re.sub(r'\W+', '_', name_without_extension)
        timestamp = int(time.time())
        save_string = f"predictions-{timestamp}-{save_string}.csv"
        return save_string


class Scorer:

    def __init__(self, predictions: dict[str, int], validation_data: dict[str, int]) -> None:
        self.predictions: dict[str, int] = predictions
        self.validation_data: dict[str, int] = validation_data
        self.f1_score: float = None

    
    def calculate_f1_score(self) -> float:
        true_labels: list[int] = [self.validation_data[query] for query in self.predictions]
        predicted_labels: list[int] = [self.predictions[query] for query in self.predictions]
        self.f1_score: float = f1_score(true_labels, predicted_labels, average='macro')
        return self.f1_score
