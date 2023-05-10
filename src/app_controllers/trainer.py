import logging


from src.app_controllers.data_handler import DataHandler


logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self) -> None:
        logger.info("Instantiating Trainer...")
        self.data_handler: DataHandler = DataHandler()


    def build_arrow_dataset(self) -> None:
        self.data_handler.convert_to_arrow()


    def train(self) -> None:
        logger.info("Training...")


