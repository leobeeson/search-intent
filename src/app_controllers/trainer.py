import logging


from src.data_handlers.dataset_splitter import DatasetSplitter


logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self, stratified: bool = True) -> None:
        self.stratified = stratified
        logger.info("Instantiating Trainer...")


    def split_data(self) -> None:
        dataset_splitter = DatasetSplitter(self.stratified)
        dataset_splitter.create_test_and_validation_sets()
        dataset_splitter.save_test_and_validation_sets()


    def train(self) -> None:
        logger.info("Training...")


