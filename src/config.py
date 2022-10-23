import yaml
import logging

settings = dict()

log_levels = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}

with open("config.yaml", 'r') as stream:
    try:
        settings = yaml.safe_load(stream)
        settings['log_level'] = log_levels[settings['log_level']]
    except yaml.YAMLError as exc:
        print(exc)
