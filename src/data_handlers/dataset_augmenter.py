import os
import logging


import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class DatasetAugmenter:


    def __init__(self, right_skewness_percentile: int = 25) -> None:
        self.right_skewness_percentile: int = right_skewness_percentile
        self.random_state: int = 235
        self.train_data: pd.DataFrame = None
        self.augmented_train_data: pd.DataFrame = None

    
    def augment_right_skewness(self) -> None:
        self._prepare_train_set_data()
        if self.train_data is not None:
            category_counts: pd.Series = self.train_data['category'].value_counts()
            skewness_percentile: np.float64 = np.percentile(category_counts, self.right_skewness_percentile)
            def upsample_category(group):
                category_count: int = len(group)    
                if category_count < skewness_percentile:
                    return group.sample(n=int(skewness_percentile), replace=True, random_state=235)
                else:
                    return group
            upsampled_df = self.train_data.groupby('category', group_keys=False).apply(upsample_category).reset_index(drop=True)
            self.augmented_train_data = upsampled_df
            self._save_augmented_train_data()
        else:
            logger.error("No train data to augment. Please first provide a labelled data set and (optionally) split into train, validations, and test.\nExiting...")


    def _save_augmented_train_data(self) -> None:
        self.augmented_train_data = self.augmented_train_data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        self.augmented_train_data.to_csv(os.environ["PATH_AUGMENTED_TRAIN_DATA"], index=False, header=False)
    

    def _prepare_train_set_data(self) -> None:
        train_data_filepath: str = os.environ["PATH_LABELLED_TRAIN_DATA"]
        self.train_data =  pd.read_csv(train_data_filepath, header=None, names=["query", "category"])
        

