"""
Neuraxle's Logging module
====================================
This module contains the Logging class, which is used to log information about the execution of a pipeline.
It is used by the classes inheriting from BaseStep to log information.
It is also modified by the AutoML class and its Trainer and various Trial repositories-related classes
to log info in various folders.

..
    Copyright 2019, Neuraxio Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""
import logging
import os
import sys

LOGGER_FORMAT = "[%(asctime)s][%(name)-12s][%(module)-8s][%(lineno)-4d][%(levelname)-8s]: %(message)s"
LOGGING_DATETIME_STR_FORMAT = '%Y-%m-%d_%H:%M:%S.%f'
if sys.version_info.major <= 3 and sys.version_info.minor <= 7:
    logging.basicConfig(format=LOGGER_FORMAT, datefmt=LOGGING_DATETIME_STR_FORMAT, level=logging.INFO)
else:
    logging.basicConfig(format=LOGGER_FORMAT, datefmt=LOGGING_DATETIME_STR_FORMAT, level=logging.INFO, force=True)


class Logger(object):
    """
    Logging class to log information about the execution of a pipeline in Neuraxle.
    The default logging config is set to INFO level with the following format:

    .. code-block:: python

        logging.basicConfig(format=LOGGER_FORMAT, datefmt=LOGGING_DATETIME_STR_FORMAT, level=logging.INFO, force=True)

    The logging config can be modified and complemented by passing the following parameters:
    - logger_name: The logger name.
    - logger_level: The logger level.
    - logger_file: The logger file - especially useful when using the AutoML class' Trials.
    - logger_format: The logger format.
    - logger_date_format: The logger date format.

    You can edit the root logger by calling this Logger class once with the desired configuration.
    The trial loggers and the loggers retrieved by the contexts will inherit this configuration.
    """

    def __init__(
        self, logger: logging.Logger = None, logger_name: str = "neuraxle", logger_level: int = None,
        logger_file: str = None, logger_format: str = None, logger_date_format: str = None
    ):
        """
        Initialize the logger.

        :param logger: The logger to use.
        :param logger_name: The logger name.
        :param logger_level: The logger level.
        :param logger_file: The logger file.
        :param logger_format: The logger format.
        :param logger_date_format: The logger date format.
        """

        self.logger = logger
        self.logger_name = logger_name
        self.logger_level = logger_level
        self.logger_file = logger_file
        self.logger_format = logger_format
        self.logger_date_format = logger_date_format

        if self.logger is None:
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(logger_level)
            if logger_file is not None:
                file_handler = logging.FileHandler(logger_file, logger_format, logger_date_format)
                self.logger.addHandler(file_handler)
        else:
            self._free_logger_file()
            self.logger.setLevel(logger_level)
            if logger_file is not None:
                file_handler = logging.FileHandler(logger_file, logger_format, logger_date_format)
                self.logger.addHandler(file_handler)

    def _free_logger_file(self):
        """
        Remove file handlers from logger to free file locks.
        """
        for h in self.logger.handlers:
            if isinstance(h, logging.FileHandler):
                self.logger.removeHandler(h)

    def _initialize_logger_with_file(self) -> logging.Logger:
        logger_folder = os.path.dirname(self.logger_file)
        os.makedirs(logger_folder, exist_ok=True)

        logfile_path = os.path.join(self.logger_file)
        logger_name = f"trial_{self.trial_number}"
        logger = logging.getLogger(logger_name)
        formatter = logging.Formatter(fmt=LOGGER_FORMAT, datefmt=LOGGING_DATETIME_STR_FORMAT)
        file_handler = logging.FileHandler(filename=logfile_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger


class ContextualLogger(Logger):
    """
    Logger for the Trial repository to log the trial to a file.
    """

    def __init__(
        context: 'ExecutionContext', logger: logging.Logger = None,
        logger_name: str = "neuraxle", logger_level: int = None, logger_file: str = None,
        logger_format: str = None, logger_date_format: str = None,
        trial_number: int = None
    ):
        """
        Initialize the logger.

        :param context: The execution context configured and attuned to the Trial.
        :param trial_number: The trial number.
        """
        logger_name: str = context.trial_number
        super().__init__(
            logger=logger, logger_name=logger_name, logger_level=logger_level, logger_file=logger_file,
            logger_format=logger_format, logger_date_format=logger_date_format
        )
        self.trial_number = trial_number
