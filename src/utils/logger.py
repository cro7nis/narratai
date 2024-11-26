import logging
from loguru import logger
import sys
from configs import settings



def configure_logger(level,
                     format):
    logger.remove()  # remove initial loguru logger to add one with custom configs
    logger.configure(extra={"classname": "None"})
    logger.add(sys.stderr, format=format, filter=None, level=level, backtrace=True,
               diagnose=True)

    return logger


logger = configure_logger(**settings.logging)
