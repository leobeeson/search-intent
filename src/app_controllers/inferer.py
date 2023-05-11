import logging


from src.classifiers.transformer_classifier import Predictor


logger = logging.getLogger(__name__)


class Inferer:


    def __init__(self) -> None:
        self.predictor: Predictor = None


    def set_up_predictor(self) -> None:
        self.predictor = Predictor()


    def predict_unlabelled_data(self, filepath: str = None) -> None:
        self.predictor.load_unlabelled_data(filepath)
        self.predictor.tokenize_data()
        self.predictor.predict()
        
    
    def evaluate_holdout_data(self) -> None:
        self.predictor.load_labelled_data()
        self.predictor.tokenize_holdout_data()
        self.predictor.evaluate()

