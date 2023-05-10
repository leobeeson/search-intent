import argparse
import configparser
import os


from datetime import datetime
import pandas as pd


from src.app_controllers.trainer import Trainer
from src.app_controllers.predictor import Predictor
from src.data_handlers.dataset_augmenter import DatasetAugmenter
from src.data_handlers.labelled_data_reader import LabelledDataReader
from src.loggers.log_utils import setup_logger

logger = setup_logger("")


def load_config():
    logger.info("Reading configuration file...")
    config = configparser.ConfigParser()
    files_read = config.read("config.ini")
    if not files_read:
        logger.critical("Failed to read config.ini file")
        raise Exception("Failed to read config.ini file")
    for section in config.sections():
        for key in config[section]:
            os.environ[key.upper()] = config[section][key]
    logger.info("Configuration variables loaded to global environment.")


def main():
    load_config()
    parser = argparse.ArgumentParser(description="Train or predict using mypkg.")
    parser.add_argument("mode", choices=["train", "predict", "augment", "validate"], help="Mode to run the program in.")
    parser.add_argument("--filepath", help="Optional file path argument.")

    args = parser.parse_args()

    if args.mode == "train":
        trainer = Trainer()
        trainer.split_data()
        trainer.train()
    elif args.mode == "validate":
        predictor = Predictor(validate=True)
        predictor.predict()
    elif args.mode == "predict":
        predictor = Predictor()
        predictor.predict()
    elif args.mode == "augment":
        train_data_filepath = os.environ["PATH_TRAIN_DATA"]
        train_data: pd.DataFrame =  pd.read_csv(train_data_filepath, header=None, names=["query", "category"])
        dataset_augmenter = DatasetAugmenter(train_data)
        dataset_augmenter.save_augmented_train_data()


if __name__ == "__main__":
    main()