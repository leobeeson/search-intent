import os
import logging
import pandas as pd


from datasets import Dataset, Features, Value, ClassLabel, DatasetDict


logger = logging.getLogger(__name__)


class ArrowDatasetConverter:


    def __init__(self, use_augmented: bool = True) -> None:
        self.use_augmented: bool = use_augmented
        self.train_data: pd.DataFrame = None
        self.validation_data: pd.DataFrame = None
        self.test_data: pd.DataFrame = None
        self.number_distinct_classes: int = None
        self._load_data()


    def convert_data_to_arrow(self) -> None:
        features = Features({"text": Value("string"), "label": ClassLabel(num_classes=self.number_distinct_classes)})
        dataset_train: Dataset = Dataset.from_pandas(self.train_data, features=features)
        dataset_validation: Dataset = Dataset.from_pandas(self.validation_data, features=features)
        dataset_test: Dataset = Dataset.from_pandas(self.test_data, features=features)   
        dataset_dict = DatasetDict()
        dataset_dict["train"] = dataset_train
        dataset_dict["validation"] = dataset_validation
        dataset_dict["test"] = dataset_test
        dataset_dict.save_to_disk(os.environ["PATH_ARROW_CACHE"])
        logger.info("Successfully converted the labelled data to arrow format. Ready to train.")    


    def _load_data(self) -> None:
        try:
            if self.use_augmented:
                self.train_data = pd.read_csv(os.environ["PATH_AUGMENTED_TRAIN_DATA"], header=None, names=["text", "label"])
            else:
                self.train_data = pd.read_csv(os.environ["PATH_LABELLED_TRAIN_DATA"], header=None, names=["text", "label"])
            self.validation_data = pd.read_csv(os.environ["PATH_LABELLED_VALIDATION_DATA"], header=None, names=["text", "label"])
            self.test_data = pd.read_csv(os.environ["PATH_LABELLED_TEST_DATA"], header=None, names=["text", "label"])
            all_data = pd.concat([self.train_data, self.validation_data, self.test_data])
            self.number_distinct_classes = all_data["label"].nunique()
        except FileNotFoundError:
            logger.error("Could not find the labelled data files. Please run the data preparation pipeline first.")
            raise FileNotFoundError("Could not find the labelled data files. Please run the data preparation pipeline first.")


    def toggle_use_augmented(self) -> None:
        self.use_augmented = not self.use_augmented
