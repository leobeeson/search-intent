import os
import pandas as pd
import numpy as np


class DatasetAugmenter:

    def __init__(self, train_data: pd.DataFrame, right_skewness_percentile: int = 25) -> None:
        self.train_data: pd.DataFrame = train_data
        self.right_skewness_percentile: int = right_skewness_percentile
        self.augmented_train_data =  self.augment_right_skewness()

    
    def augment_right_skewness(self) -> None:
        category_counts: pd.Series = self.train_data['category'].value_counts()
        skewness_percentile: np.float64 = np.percentile(category_counts, self.right_skewness_percentile)
        def upsample_category(group):
            category_count: int = len(group)    
            if category_count < skewness_percentile:
                return group.sample(n=int(skewness_percentile), replace=True, random_state=235)
            else:
                return group
        upsampled_df = self.train_data.groupby('category', group_keys=False).apply(upsample_category).reset_index(drop=True)
        return upsampled_df


    def save_augmented_train_data(self) -> None:
        self.augmented_train_data.to_csv(os.environ["PATH_AUGMENTED_TRAIN_DATA"], index=False, header=False)
    



