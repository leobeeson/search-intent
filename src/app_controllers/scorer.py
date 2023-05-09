from sklearn.metrics import f1_score


from src.data_handlers.labelled_data_reader import LabelledDataReader
from src.loggers.log_utils import setup_logger


logger = setup_logger(__name__)


class Scorer:

    def __init__(self, predictions: dict[str, int], validation_data: dict[str, int]) -> None:
        logger.info("Instantiating Predictor...")
        self.predictions: dict[str, int] = predictions
        self.validation_data: dict[str, int] = validation_data
        self.f1_score: float = None

    
    def calculate_f1_score(self) -> float:
        true_labels: list[int] = [self.validation_data[query] for query in self.predictions]
        predicted_labels: list[int] = [self.predictions[query] for query in self.predictions]
        self.f1_score: float = f1_score(true_labels, predicted_labels, average='macro')
        return self.f1_score
