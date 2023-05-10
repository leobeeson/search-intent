import os
import pandas as pd


class DatasetSplitter:


    def __init__(self, stratified=True):
        self.stratified: bool = stratified
        self.dataset_filepath: str = os.environ["PATH_LABELLED_DATA"]
        self.train_fraction: float = 0.8
        self.validation_fraction: float = 0.1
        self.random_state: int = 235
        self.train_data: pd.DataFrame = None
        self.validation_data: pd.DataFrame = None


    def split_train_validation_test_sets(self):
        full_labelled_data: pd.DataFrame =  pd.read_csv(self.dataset_filepath, header=None, names=["query", "category"])
        if self.stratified:
            self.train_data: pd.DataFrame = full_labelled_data.groupby("category").sample(frac=self.train_fraction, random_state=self.random_state)
            available_data: pd.DataFrame = full_labelled_data.drop(self.train_data.index)
            adjusted_val_fraction: float = self.validation_fraction / (1 - self.train_fraction)
            self.validation_data: pd.DataFrame = available_data.groupby("category").sample(frac=adjusted_val_fraction, random_state=self.random_state)
            self.test_data: pd.DataFrame = available_data.drop(self.validation_data.index)
        else:
            self.train_data: pd.DataFrame = full_labelled_data.sample(frac=self.train_fraction, random_state=self.random_state)
            available_data: pd.DataFrame = full_labelled_data.drop(self.train_data.index)
            adjusted_val_fraction: float = self.validation_fraction / (1 - self.train_fraction)
            self.validation_data: pd.DataFrame = available_data.sample(frac=adjusted_val_fraction, random_state=self.random_state)
            self.test_data: pd.DataFrame = available_data.drop(self.validation_data.index)
        self._save_train_validation_test_sets()


    def _save_train_validation_test_sets(self):
        self.train_data = self.train_data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        self.validation_data = self.validation_data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        self.test_data = self.test_data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        self.train_data.to_csv(os.environ["PATH_LABELLED_TRAIN_DATA"], index=False, header=False)
        self.validation_data.to_csv(os.environ["PATH_LABELLED_VALIDATION_DATA"], index=False, header=False)
        self.test_data.to_csv(os.environ["PATH_LABELLED_TEST_DATA"], index=False, header=False)
