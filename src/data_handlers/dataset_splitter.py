import os
import pandas as pd


class DatasetSplitter:


    def __init__(self, stratified=True):
        self.stratified = stratified
        self.dataset_filepath = os.environ["PATH_LABELLED_DATA"]
        self.train_fraction = 0.8
        self.random_state = 235
        self.create_test_and_validation_sets()
        self.save_test_and_validation_sets()


    def create_test_and_validation_sets(self):
        full_labelled_data =  pd.read_csv(self.dataset_filepath, header=None, names=["query", "category"])
        if self.stratified:
            self.train_data = full_labelled_data.groupby("category").sample(frac=self.train_fraction, random_state=self.random_state)
            self.validation_data = full_labelled_data.drop(self.train_data.index)
        else:
            self.train_data = full_labelled_data.sample(frac=self.train_fraction, random_state=self.random_state)
            self.validation_data = full_labelled_data.drop(self.train_data.index)


    def save_test_and_validation_sets(self):
        self.train_data.to_csv(os.environ["PATH_TRAIN_DATA"], index=False, header=False)
        self.validation_data.to_csv(os.environ["PATH_VALIDATION_DATA"], index=False, header=False)
