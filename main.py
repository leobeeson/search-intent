import argparse
import configparser
import os


from src.app_controllers.trainer import Trainer
from src.app_controllers.data_handler import DataHandler
from src.loggers.log_utils import setup_logger
from src.app_controllers.inferer import Inferer


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
    parser.add_argument("mode", choices=["split", "augment", "transform", "train", "optimize", "evaluate", "predict"], help="Mode to run the program in.")
    parser.add_argument("--filepath", help="Optional file path argument.")
    parser.add_argument("--no-stratify", action="store_true", help="Do not stratify the data during the split.")
    parser.add_argument("--percentile", type=int, default=25, choices=range(1, 101), help="Percentile to use in 'augment' mode. Must be between 1 and 100 (inclusive). Default is 25.")
    parser.add_argument("--model", type=str, default="distilbert", choices=["distilbert"], help="The name of the model to evaluate. Default is 'distilbert'.")

    args = parser.parse_args()

    if args.mode == "split":
        data_handler = DataHandler()
        if args.no_stratify:
            data_handler.toggle_stratified()
        data_handler.split_labelled_data()
    elif args.mode == "augment":
        data_handler = DataHandler()
        if args.percentile != 25:
            data_handler.set_augment_percentile(args.percentile)
        data_handler.augment_train_data()
    elif args.mode == "transform":
        data_handler = DataHandler()
        data_handler.convert_to_arrow()
    elif args.mode == "train":
        trainer = Trainer()
        trainer.build_arrow_dataset()
        trainer.train_transformer()
    elif args.mode == "optimize":
        trainer = Trainer()
        trainer.optimize_hyperparameters()
    elif args.mode == "evaluate":
        if args.model == "distilbert":
            inferer = Inferer()
            inferer.set_up_predictor()
            inferer.evaluate_holdout_data()
    elif args.mode == "predict":
        inferer = Inferer()
        inferer.set_up_predictor()
        if args.filepath is None:
            logger.warning("No filepath provided for prediction.\nUsing stored unlabelled test data for predictions.")    
            inferer.predict_unlabelled_data()
        else:
            inferer.predict_unlabelled_data(args.filepath)


if __name__ == "__main__":
    main()