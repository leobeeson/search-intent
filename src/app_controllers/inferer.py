import logging


from src.classifiers.transformer_classifier import Evaluator, Predictor


logger = logging.getLogger(__name__)


class Inferer:


    def __init__(self) -> None:
        pass


    def set_up_predictor(self, filepath: str = None) -> None:
        self.predictor = Predictor()
        self.predictor.load_data(filepath)
        self.predictor.tokenize_data()


    def predict_unlabelled_data(self) -> None:
        self.predictor.predict()
        

    def set_up_evaluator(self) -> None:
        self.evaluator = Evaluator()
        self.evaluator.load_data()
        self.evaluator.tokenize_data()

    
    def evaluate_holdout_data(self) -> None:
        self.evaluator.evaluate()
        logger.info(f"F1-Score: {self.evaluator.get_f1_score()}")
