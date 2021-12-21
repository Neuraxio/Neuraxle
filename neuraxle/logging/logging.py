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
import sys
from io import StringIO
from typing import Dict

LOGGER_FORMAT = "[%(asctime)s][%(levelname)-8s][%(name)-8s][%(module)-8s][%(lineno)-4d]: %(message)s"
LOGGER_FORMAT = "[%(asctime)s][%(levelname)-8s][%(name)-8s][%(module)-1s.py:%(lineno)-1d]: %(message)s"
LOGGING_DATETIME_STR_FORMAT = '%Y-%m-%d_%H:%M:%S.%f'
if sys.version_info.major <= 3 and sys.version_info.minor <= 7:
    logging.basicConfig(format=LOGGER_FORMAT, datefmt=LOGGING_DATETIME_STR_FORMAT, level=logging.INFO)
else:
    logging.basicConfig(format=LOGGER_FORMAT, datefmt=LOGGING_DATETIME_STR_FORMAT, level=logging.INFO, force=True)

NEURAXLE_ROOT_LOGGER_NAME = "neuraxle"
NEURAXLE_LOG_FORMATTER = logging.Formatter(fmt=LOGGER_FORMAT, datefmt=LOGGING_DATETIME_STR_FORMAT)
NEURAXLE_LOGGER_STRING_IO: Dict[str, StringIO] = {}
NEURAXLE_LOGGER_FILE_HANDLERS: Dict[str, logging.FileHandler] = {}


class _FilterSTDErr(logging.Filter):
    def filter(self, record):
        return record.levelno < logging.WARN


class NeuraxleLogger(logging.Logger):

    @staticmethod
    def from_identifier(identifier: str) -> 'NeuraxleLogger':
        """
        Returns a logger from an identifier.
        :param identifier: The identifier of the logger
        :return: The logger
        """
        logger: NeuraxleLogger = logging.getLogger(identifier)
        logger.setLevel(logging.DEBUG)

        logger.with_string_io(identifier)
        return logger

    def with_string_io(self, identifier: str) -> 'NeuraxleLogger':
        """
        Returns a logger from an identifier.
        :param identifier: The identifier of the logger
        :return: The logger
        """
        if identifier not in NEURAXLE_LOGGER_STRING_IO:
            string_io = StringIO()
            NEURAXLE_LOGGER_STRING_IO[identifier] = string_io
            string_io_handler = logging.StreamHandler(stream=string_io)
            string_io_handler.set_name("string_io_handler")
            string_io_handler.setFormatter(NEURAXLE_LOG_FORMATTER)
            self.addHandler(string_io_handler)
        return self

    def with_file_handler(self, file_path: str) -> 'NeuraxleLogger':
        """
        Returns a logger from an identifier.
        :param identifier: The identifier of the logger
        :return: The logger
        """
        self.without_file_handler()

        file_handler = logging.FileHandler(file_path)
        file_handler.set_name(f"file_handler://{file_path}")
        file_handler.setFormatter(NEURAXLE_LOG_FORMATTER)
        self.addHandler(file_handler)

        NEURAXLE_LOGGER_FILE_HANDLERS[self.name] = file_handler
        return self

    def without_file_handler(self) -> 'NeuraxleLogger':
        """
        Returns a logger from an identifier.
        :param identifier: The identifier of the logger
        :return: The logger
        """
        if self.name in NEURAXLE_LOGGER_FILE_HANDLERS:
            NEURAXLE_LOGGER_FILE_HANDLERS[self.name].close()
            self.removeHandler(NEURAXLE_LOGGER_FILE_HANDLERS[self.name])
            del NEURAXLE_LOGGER_FILE_HANDLERS[self.name]
        return self

    def with_std_handlers(self) -> 'NeuraxleLogger':
        error_handler = logging.StreamHandler(stream=sys.stderr)
        error_handler.set_name("error_handler")
        error_handler.setFormatter(NEURAXLE_LOG_FORMATTER)
        error_handler.setLevel(logging.WARN)
        self.addHandler(error_handler)

        info_handler = logging.StreamHandler(stream=sys.stdout)
        info_handler.set_name("info_handler")
        info_handler.setFormatter(NEURAXLE_LOG_FORMATTER)
        info_handler.setLevel(logging.DEBUG)
        err_filter = _FilterSTDErr()
        info_handler.addFilter(err_filter)
        self.addHandler(info_handler)
        return self

    def get_string_history(self) -> str:
        return NEURAXLE_LOGGER_STRING_IO[self.name].getvalue()


logging.setLoggerClass(NeuraxleLogger)

NEURAXLE_ROOT_LOGGER: NeuraxleLogger = NeuraxleLogger.from_identifier(
    NEURAXLE_ROOT_LOGGER_NAME
).with_std_handlers()

# error_handler = logging.StreamHandler(stream=sys.stderr)
# error_handler.set_name("error_handler")
# error_handler.setFormatter(NEURAXLE_LOG_FORMATTER)
# error_handler.setLevel(logging.WARN)
# NEURAXLE_ROOT_LOGGER.addHandler(error_handler)
#
# info_handler = logging.StreamHandler(stream=sys.stdout)
# info_handler.set_name("info_handler")
# info_handler.setFormatter(NEURAXLE_LOG_FORMATTER)
# info_handler.setLevel(logging.DEBUG)
# err_filter = _FilterSTDErr()
# info_handler.addFilter(err_filter)
# NEURAXLE_ROOT_LOGGER.addHandler(info_handler)
