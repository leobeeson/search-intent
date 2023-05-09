import os
import configparser
import pytest

@pytest.fixture(scope="session", autouse=True)
def load_config_for_tests():
    config = configparser.ConfigParser()
    files_read = config.read("config.ini")
    if not files_read:
        raise Exception("Failed to read config.ini file")
    for section in config.sections():
        for key in config[section]:
            os.environ[key.upper()] = config[section][key]
