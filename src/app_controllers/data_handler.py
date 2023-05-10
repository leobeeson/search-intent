import os
import logging


import pandas as pd


from src.data_handlers.dataset_splitter import DatasetSplitter
from src.data_handlers.dataset_augmenter import DatasetAugmenter
from src.data_handlers.arrow_dataset_converter import ArrowDatasetConverter


logger = logging.getLogger(__name__)


class DataHandler:

    def __init__(self) -> None:
        self.dataset_splitter= DatasetSplitter()
        self.dataset_augmenter = DatasetAugmenter()
        self.arrow_dataset_converter = ArrowDatasetConverter()


    def split_labelled_data(self) -> None:
        self.dataset_splitter.split_train_validation_test_sets()


    def toggle_stratified(self) -> None:
        self.dataset_splitter.stratified = not self.dataset_splitter.stratified


    def augment_train_data(self) -> None:
        self.dataset_augmenter.augment_right_skewness()

    
    def set_augment_percentile(self, percentile: int) -> None:
        self.dataset_augmenter.right_skewness_percentile = percentile


    def convert_to_arrow(self) -> None:
        self.arrow_dataset_converter.convert_to_arrow()
    

    def toggle_use_augmented(self) -> None:
        self.arrow_dataset_converter.use_augmented = not self.arrow_dataset_converter.use_augmented
