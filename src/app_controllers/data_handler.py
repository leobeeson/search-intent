import logging


from src.data_handlers.dataset_splitter import DatasetSplitter


logger = logging.getLogger(__name__)


class DataHandler:

    def __init__(self, stratified: bool) -> None:
        self.stratified = stratified
        self.dataset_splitter = DatasetSplitter(self.stratified)


    def split_labelled_data(self) -> None:
        self.dataset_splitter.split_train_validation_test_sets()
        self.dataset_splitter.save_train_validation_test_sets()

