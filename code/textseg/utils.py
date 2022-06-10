import configparser
from config import ROOT_DIR, CONFIG_DIR

def read_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_DIR/"config.ini")
    return config

