import os
import re
import torch
import logging
import time
import pandas as pd
import numpy as np


from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_from_disk, DatasetDict, Dataset, Features, Value
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
from torch.utils.data import DataLoader


from src.data_handlers.unlabelled_data_reader import UnlabelledDataReader


logger = logging.getLogger(__name__)


class Predictor:


    def __init__(self):
        self.model_path = os.environ['PATH_DISTILBERT_MODEL']
        logger.info("Setting up DistilBert model...")
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
        logger.info("Setting up DistilBert tokenizer...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Torch Device is: {self.device}")
        self.model.to(self.device)
        self.categories: list[int] = []
        self.probabilities: list[float] = []
        self.predictions: list[tuple] = []
        self.batch_size = 16
        self.evaluation: dict[str, float] = {}
        self.classification_report = None


    def load_unlabelled_data(self, filepath: str) -> None:
        logger.info(f"Loading data...")
        if filepath is None:
            self.data_filepath: str = os.environ["PATH_UNLABELLED_TEST_DATA"]
        else:
            self.data_filepath: str = filepath
        reader = UnlabelledDataReader(self.data_filepath)
        unlabelled_data: list[str] = reader.read_data()
        df = pd.DataFrame(unlabelled_data, columns=["text"])
        features = Features({"text": Value("string")})
        dataset: Dataset = Dataset.from_pandas(df, features=features)
        self.dataset = dataset


    def load_labelled_data(self, split: str = "test") -> None:
        logger.info(f"Loading labelled data...")
        self.data_filepath: str = os.environ['PATH_ARROW_CACHE']
        self.data_split: str = split
        dataset_dict: DatasetDict = load_from_disk(self.data_filepath)
        dataset: Dataset = dataset_dict[self.data_split]
        self.dataset = dataset


    def tokenize_data(self):
        self.dataset = self.dataset.map(lambda e: self.tokenizer(e["text"], truncation=True, padding="max_length"), batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "text"])


    def tokenize_holdout_data(self):
        self.dataset = self.dataset.map(lambda e: self.tokenizer(e['text'], truncation=True, padding='max_length'), batched=True)
        if self.dataset.num_rows == 0:
            logger.error("No data in holdout dataset. Did you load the data?\nStratified splitting can cause no rows being left for the test set if initial training set is too small.\nIf this is the case, split the data without stratification like this: python3 main.py split --no-stratify.")
            raise ValueError("No data in dataset. Did you load the data?")
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "text", "label"])


    def predict(self, batch_size=32):
        logger.info("Starting prediction...")
        self.model.eval()
        dataloader = DataLoader(self.dataset, batch_size=batch_size)
        for batch in tqdm(dataloader, desc="Predicting"):
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != "text"}
            with torch.no_grad():
                outputs = self.model(**inputs)
            categories = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            max_probabilities = np.max(probabilities, axis=1)

            self.categories.extend(categories)
            self.probabilities.extend(max_probabilities)

            batch_predictions = list(zip(batch["text"], categories, max_probabilities))
            self.predictions.extend(batch_predictions)
        logger.info("Saving predictions to file...")
        self._save_predictions()


    def evaluate(self, batch_size=32):
        logger.info("Starting evaluation...")
        self.model.eval()
        dataloader = DataLoader(self.dataset, batch_size=batch_size)
        true_labels = []
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {k: v.to(self.device) for k, v in batch.items() if k not in ["text", "label"]}
            true_labels.extend(batch["label"].numpy())
            with torch.no_grad():
                outputs = self.model(**inputs)
            categories = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            self.categories.extend(categories)

            batch_predictions = list(zip(batch["text"], categories))
            self.predictions.extend(batch_predictions)

        self.evaluation["accuracy"] = accuracy_score(true_labels, self.categories)
        self.evaluation["f1_macro"] = f1_score(true_labels, self.categories, average="macro", zero_division=0)
        self.evaluation["precision_macro"] = precision_score(true_labels, self.categories, average="macro", zero_division=0)
        self.evaluation["recall_macro"] = recall_score(true_labels, self.categories, average="macro", zero_division=0)
        self.evaluation["f1_weighted"] = f1_score(true_labels, self.categories, average="weighted", zero_division=0)
        self.evaluation["precision_weighted"] = precision_score(true_labels, self.categories, average="weighted", zero_division=0)
        self.evaluation["recall_weighted"] = recall_score(true_labels, self.categories, average="weighted", zero_division=0)
        self.classification_report = classification_report(true_labels, self.categories, zero_division=0)

        logger.info(f"""
        \nEvaluation results:
        Accuracy = {self.evaluation["accuracy"]:.2f}
        Macro F1 Score = {self.evaluation["f1_macro"]:.2f}
        Weighted F1 Score = {self.evaluation["f1_weighted"]:.2f}
        Macro Precision = {self.evaluation["precision_macro"]:.2f}
        Weighted Precision = {self.evaluation["precision_weighted"]:.2f}
        Macro Recall = {self.evaluation["recall_macro"]:.2f}
        Weighted Recall = {self.evaluation["recall_weighted"]:.2f}
        \n
        """)


    def get_predictions(self):
        return self.predictions


    def _save_predictions(self):
        filename: str = self._filename_to_save_string(self.data_filepath)
        filepath: str = os.path.join(os.environ["PATH_PREDICTIONS"], filename)
        logger.info(f"Saving predictions to {filepath}")
        try:
            with open(filepath, "w") as csv_file:
                for prediction in self.predictions:
                    csv_file.write(f"{prediction[0]},{prediction[1]},{prediction[2]}\n")
            logger.info(f"Predictions saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save predictions. Exception: {e}")
         
    
    def _filename_to_save_string(self, filepath):
        base_name = os.path.basename(filepath)
        name_without_extension, _ = os.path.splitext(base_name)
        save_string = re.sub(r'\W+', '_', name_without_extension)
        timestamp = int(time.time())
        save_string = f"predictions-{timestamp}-{save_string}.csv"
        return save_string
