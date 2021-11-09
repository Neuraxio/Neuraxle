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
