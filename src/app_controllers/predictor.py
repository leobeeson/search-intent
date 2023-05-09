import os


from src.classifiers.classification_pipeline import ClassificationPipeline
from src.data_handlers.labelled_data_reader import LabelledDataReader
from src.app_controllers.scorer import Scorer
from src.loggers.log_utils import setup_logger


logger = setup_logger(__name__)


class Predictor:

    def __init__(self, data_filepath: str = None, validate: bool = False) -> None:
        logger.info("Instantiating Predictor...")
        self.validate: bool = validate
        self.data_filepath: str = data_filepath
        self.data: dict[str, int] = self._load_data()
        self.classification_pipeline: ClassificationPipeline = ClassificationPipeline(self._load_train_data())
        self.predictions: dict[str, int] = {}
        self.scorer: Scorer = None
        self.f1_score: float = None


    def predict(self) -> int:
        logger.info("Predicting queries...")
        for query in self.data:
            category: int = self.predict_query(query)
            if category is None:
                category = -1
            self.predictions[query] = category
        if self.validate:
            f1_score: float = self.score()
            logger.info(f"F1 score: {f1_score}")
        return self.predictions


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
            self.data_filepath: str = os.environ["PATH_VALIDATION_DATA"]
        #TODO: Load data from text file for unlabelled (test) data.
        data_reader: LabelledDataReader = LabelledDataReader(self.data_filepath)    
        labelled_data: dict[str, int] = data_reader.read_labelled_data()
        return labelled_data


    def _load_train_data(self) -> dict[str, int]:
        data_filepath: str = os.environ["PATH_TRAIN_DATA"]
        data_reader: LabelledDataReader = LabelledDataReader(data_filepath)
        labelled_train_data: dict[str, int] = data_reader.read_labelled_data()
        return labelled_train_data