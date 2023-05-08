

from src.data_handlers.dataset_splitter import DatasetSplitter


class Trainer:

    def __init__(self, stratified: bool = True) -> None:
        self.stratified = stratified
        print("Instantiating Trainer...")


    def split_data(self) -> None:
        dataset_splitter = DatasetSplitter(self.stratified)
        dataset_splitter.create_test_and_validation_sets()
        dataset_splitter.save_test_and_validation_sets()


    def train(self) -> None:
        print("Training...")


