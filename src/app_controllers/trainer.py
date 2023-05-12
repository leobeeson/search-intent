import logging


from src.app_controllers.data_handler import DataHandler
from src.trainers.transformer_trainer import train_distilbert_model
from src.trainers.hyperparameter_optimizator import perform_hyperparameter_optimization


logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self) -> None:
        logger.info("Instantiating Trainer...")
        self.data_handler: DataHandler = DataHandler()


    def build_arrow_dataset(self) -> None:
        self.data_handler.convert_to_arrow()


    def train_transformer(self) -> None:
        logger.info("Training transformer model...")
        train_distilbert_model()


    def optimize_hyperparameters(self) -> None:
        logger.info("Performing hyperparameter search...This may take a day or two...")
        perform_hyperparameter_optimization()