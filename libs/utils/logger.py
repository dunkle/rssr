# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.

import logging


def create_logger(log_file=None, log_level=logging.INFO):

    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=format_str, level=log_level)
    logger = logging.getLogger(__name__)

    if log_file is not None:
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logging.getLogger('').addHandler(file_handler)

    return logger
