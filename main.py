import argparse
import configparser
import os

from src.app_controllers.trainer import Trainer
from src.app_controllers.predictor import Predictor

def load_config():
    config = configparser.ConfigParser()
    files_read = config.read('config.ini')
    if not files_read:
        raise Exception("Failed to read config.ini file")
    for section in config.sections():
        for key in config[section]:
            os.environ[key.upper()] = config[section][key]


def main():
    load_config()
    parser = argparse.ArgumentParser(description="Train or predict using mypkg.")
    parser.add_argument("mode", choices=["train", "predict"], help="Mode to run the program in.")
    parser.add_argument("--filepath", help="Optional file path argument.")

    args = parser.parse_args()

    if args.mode == "train":
        trainer = Trainer()
        trainer.split_data()
        trainer.train()
    elif args.mode == "predict":
        predictor = Predictor()
        predictor.predict()

if __name__ == "__main__":
    main()