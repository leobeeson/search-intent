import os
import logging


from src.classifiers.classification_pipeline import ClassificationPipeline
from src.indexers.labelled_data_set import LabelledDataSet


logger = logging.getLogger(__name__)


class Predictor:

    def __init__(self) -> None:
        logger.info("Instantiating Predictor...")
        self.load_indices()


    def load_indices(self) -> None:
        train_data_index = self.load_train_data_index()
        validation_data_index = self.load_validation_data_index()
        self.train_data_index = train_data_index
        self.validation_data_index = validation_data_index


    def load_train_data_index(self) -> None:
        train_data_filepath = os.environ["PATH_TRAIN_DATA"]
        train_data_indexer = LabelledDataSet(train_data_filepath)
        train_data = train_data_indexer.index_data()
        return train_data

    def load_validation_data_index(self) -> None:
        validation_data_filepath = os.environ["PATH_VALIDATION_DATA"]
        validation_data_indexer = LabelledDataSet(validation_data_filepath)
        validation_data = validation_data_indexer.index_data()
        return validation_data


    def predict(self) -> int:
        classification_pipeline = ClassificationPipeline(self.train_data_index)
        category = classification_pipeline.classify("sample search query")
        return category
