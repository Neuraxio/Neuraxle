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
import re
import sys
from threading import Thread
from io import StringIO
from multiprocessing import Queue
from typing import IO, Dict, List, Optional

LOGGER_FORMAT = "⠀[%(asctime)s][%(levelname)-8s][%(name)-8s][%(module)-1s.py:%(lineno)-1d][%(processName)s:%(threadName)s]: %(message)s"
LOGGER_FORMAT_PREFIX_REPLACE_REGEXPR = r"⠀(\[.*?\]): ", r""
LOGGER_FORMAT_PREFIX_SEP_L = "["
LOGGER_FORMAT_PREFIX_SEP_R = "]"
LOGGING_DATETIME_STR_FORMAT = '%Y-%m-%d_%H:%M:%S.%f'

FORMATTER = logging.Formatter(fmt=LOGGER_FORMAT, datefmt=LOGGING_DATETIME_STR_FORMAT)

if sys.version_info.major <= 3 and sys.version_info.minor <= 7:
    logging.basicConfig(format=LOGGER_FORMAT, datefmt=LOGGING_DATETIME_STR_FORMAT, level=logging.INFO)
else:
    logging.basicConfig(format=LOGGER_FORMAT, datefmt=LOGGING_DATETIME_STR_FORMAT, level=logging.INFO, force=True)

NEURAXLE_LOGGER_NAME = "neuraxle"


LOGGER_STRING_IO: Dict[str, StringIO] = {}
LOGGER_FILE_HANDLERS: Dict[str, logging.FileHandler] = {}


class _FilterSTDErr(logging.Filter):
    def filter(self, record):
        return record.levelno < logging.WARN


class NeuraxleLogger(logging.Logger):

    @staticmethod
    def root() -> 'NeuraxleLogger':
        return NeuraxleLogger(NEURAXLE_LOGGER_NAME)

    @staticmethod
    def from_identifier(identifier: str) -> 'NeuraxleLogger':
        """
        Returns a logger from an identifier.
        :param identifier: The identifier of the logger
        :return: The logger
        """
        logger: NeuraxleLogger = logging.getLogger(identifier)

        logger.with_string_io(identifier)
        return logger

    def with_string_io(self, identifier: str) -> 'NeuraxleLogger':
        """
        Returns a logger from an identifier.
        :param identifier: The identifier of the logger
        :return: The logger
        """
        if identifier not in LOGGER_STRING_IO:
            string_io = StringIO()
            LOGGER_STRING_IO[identifier] = string_io
            self._add_stream_handler("string_io_handler", string_io)
        return self

    def with_file_handler(self, file_path: str) -> 'NeuraxleLogger':
        """
        Returns a logger from an identifier.
        :param identifier: The identifier of the logger
        :return: The logger
        """
        self.without_file_handler()

        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel('DEBUG')
        self._add_partial_handler(f"file_handler:{file_path}", file_handler, level=logging.DEBUG)

        LOGGER_FILE_HANDLERS[self.name] = file_handler
        return self

    def without_file_handler(self) -> 'NeuraxleLogger':
        """
        Returns a logger from an identifier.
        :param identifier: The identifier of the logger
        :return: The logger
        """
        if self.name in LOGGER_FILE_HANDLERS:
            LOGGER_FILE_HANDLERS[self.name].close()
            try:
                self.removeHandler(LOGGER_FILE_HANDLERS[self.name])
            except KeyError as ke:
                raise ke from ke  # Breakpoint here.
            del LOGGER_FILE_HANDLERS[self.name]
        return self

    def read_log_file(self) -> List[str]:
        if self.name not in LOGGER_FILE_HANDLERS:
            raise ValueError(
                f"No file handler for logger named `{self.name}`. Perhaps you forgot to call "
                f"`self.with_file_handler`, or removed the file handler?")
        with open(LOGGER_FILE_HANDLERS[self.name].baseFilename, 'r') as f:
            return f.readlines()

    def with_std_handlers(self) -> 'NeuraxleLogger':
        self._add_stream_handler("errr_handler", sys.stderr, logging.WARN)
        self._add_stream_handler("info_handler", sys.stdout, logging.DEBUG, _FilterSTDErr())
        return self

    def get_scoped_string_history(self) -> str:
        return LOGGER_STRING_IO[self.name].getvalue()

    def get_root_string_history(self) -> str:
        return LOGGER_STRING_IO[NEURAXLE_LOGGER_NAME].getvalue()

    def get_short_scoped_logs(self) -> List[str]:
        _logs = self.get_scoped_string_history()
        _logs = self.shorten_log_lines_prefixes(_logs)
        return _logs.split("\n")

    def get_short_root_logs(self) -> List[str]:
        _logs = self.get_root_string_history()
        _logs = self.shorten_log_lines_prefixes(_logs)
        return _logs.split("\n")

    def __iter__(self):
        """
        Short method to access :func:`get_short_scoped_logs` 's iterator.
        """
        return self.get_short_scoped_logs().__iter__()

    def __getitem__(self, item):
        """
        Short method to access :func:`get_short_scoped_logs` 's items.
        """
        return self.get_short_scoped_logs()[item]

    def print_root_string_history(self) -> None:
        print(f"{NEURAXLE_LOGGER_NAME} -> str:\n\n"
              f"{LOGGER_STRING_IO[NEURAXLE_LOGGER_NAME].getvalue()}")

    def _add_stream_handler(
        self,
        handler_name: str,
        stream: IO,
        level: Optional[int] = None,
        _filter: Optional[logging.Filter] = None
    ) -> 'NeuraxleLogger':
        handler = logging.StreamHandler(stream=stream)
        return self._add_partial_handler(handler_name, handler, level, _filter)

    def _add_partial_handler(
        self,
        handler_name: str,
        handler: logging.Handler,
        level: Optional[int] = None,
        _filter: Optional[logging.Filter] = None
    ) -> 'NeuraxleLogger':
        handler.setFormatter(FORMATTER)
        handler.set_name(f"{self.name}.{handler_name}")
        if level is not None:
            handler.setLevel(level)
        if _filter is not None:
            handler.addFilter(_filter)
        self.addHandler(handler)
        return self

    @staticmethod
    def shorten_log_lines_prefixes(logs):
        # This line is for retrocompability with old logs of 0.7.0.
        c = "⠀"
        logs = (c + logs.replace("\n", "\n" + c)).replace(c + c, c)
        logs = re.sub(LOGGER_FORMAT_PREFIX_REPLACE_REGEXPR[0], LOGGER_FORMAT_PREFIX_REPLACE_REGEXPR[1], logs)
        return logs.strip().rstrip(c).rstrip("\n")


logging.setLoggerClass(NeuraxleLogger)

NEURAXLE_ROOT_LOGGER: NeuraxleLogger = NeuraxleLogger.from_identifier(NEURAXLE_LOGGER_NAME)
NEURAXLE_ROOT_LOGGER.setLevel(logging.DEBUG)
NEURAXLE_ROOT_LOGGER.with_std_handlers()


class ParallelLoggingConsumerThread:
    """
    This class is used to receive logging messages sent from worker processes that
    are running in different PROCESSES (not threads).

    The logging will work when using threads. However, when using processes (multiprocessing.Process) :class:`multiprocessing.Process`
    , the present class
    is needed in the main process to receive the logging messages from the worker processes properly in the main console output.
    """

    def __init__(self, logging_queue: Queue = None):
        # type is, to me more precise: Queue[logging.LogRecord]
        self.logging_queue: Queue = logging_queue or Queue()
        self.logging_thread: Thread = None

    def start(self):
        """
        Start the thread to read the logs from the queue.
        """
        _logging_thread = Thread(
            target=self.logger_consumer_thread_func,
            args=(self.logging_queue,)
        )
        _logging_thread.daemon = True
        _logging_thread.start()
        self.logging_thread = _logging_thread

    @staticmethod
    def logger_consumer_thread_func(queue: Queue):
        """
        This function is destined to be run in a separate thread.
        Receiving a None signal in the queue means it's time to finish and to break the
        while True. This None signal is as sent from within the self. :func:`join` () method.
        The Queue normally consumes logging.LogRecord items.
        """
        while True:
            rec: logging.LogRecord = queue.get()

            if rec is None:
                break

            logger = logging.getLogger(rec.name)
            logger.handle(rec)

    def join(self, timeout: float = None):
        """
        Send the None signal for the thread to finish, and join on the thread to finish it.
        Note that once this method is called, the thread will finish and the threading.Thread
        object will be destroyed. Therefore, this method should be called only once and after
        the producers of parallel logging messages finished their job.

        Timeout is in seconds.
        """
        self.logging_queue.put(None)
        self.logging_thread.join(timeout=timeout)
        self.logging_queue.close()
        self.logging_queue.cancel_join_thread()
        self.logging_queue.join_thread()


def register_log_producer_for_main_logger_thread_to_consume(logging_queue: Queue):
    if logging_queue is not None:
        queue_handler = logging.handlers.QueueHandler(logging_queue)
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        root.addHandler(queue_handler)
