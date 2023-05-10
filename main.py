import argparse
import configparser
import os


from datetime import datetime
import pandas as pd


from src.app_controllers.trainer import Trainer
from src.app_controllers.predictor import Predictor
from src.data_handlers.dataset_augmenter import DatasetAugmenter
from src.app_controllers.data_handler import DataHandler
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
    parser.add_argument("mode", choices=["split", "train", "predict", "augment", "validate"], help="Mode to run the program in.")
    parser.add_argument("--filepath", help="Optional file path argument.")
    parser.add_argument("--no-stratify", action="store_true", help="Do not stratify the data during the split.")


    args = parser.parse_args()

    if args.mode == "split":
        data_handler = DataHandler(stratified=not args.no_stratify)
        data_handler.split_labelled_data()
    elif args.mode == "train":
        trainer = Trainer()
        trainer.split_data()
        trainer.train()
    elif args.mode == "validate":
        predictor = Predictor(validate=True)
        predictor.predict()
    elif args.mode == "predict":
        if args.filepath is None:
            logger.warning("No filepath provided for prediction.\nUsing stored test data for predictions.")    
            data_filepath: str = os.environ["PATH_LABELLED_TEST_DATA"]
            predictor = Predictor(data_filepath)
        else:
            predictor = Predictor(args.filepath)
        predictor.predict()
    elif args.mode == "augment":
        train_data_filepath = os.environ["PATH_LABELLED_TRAIN_DATA"]
        train_data: pd.DataFrame =  pd.read_csv(train_data_filepath, header=None, names=["query", "category"])
        dataset_augmenter = DatasetAugmenter(train_data)
        dataset_augmenter.save_augmented_train_data()


if __name__ == "__main__":
    main()