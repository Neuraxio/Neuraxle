"""
Neuraxle's SQL Hyperparameter Repository Classes
=================================================
Data objects and related repositories used by AutoML, SQL version.

Classes are splitted like this for the AutoML:
- Projects
- Clients
- Rounds (runs)
- Trials
- TrialSplits
- MetricResults

..
    Copyright 2021, Neuraxio Inc.

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

import datetime
import hashlib
import json
import logging
import os
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from json.encoder import JSONEncoder
from logging import FileHandler, Logger
from typing import Any, Callable, Dict, Iterable, List, Tuple, Type

import numpy as np
from neuraxle.base import BaseStep, ExecutionContext, Flow
from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.space import HyperparameterSamples, RecursiveDict
from neuraxle.logging.logging import LOGGING_DATETIME_STR_FORMAT
from neuraxle.metaopt.data.vanilla import (AutoMLFlow, BaseDataclass,
                                           ClientDataclass,
                                           MetricResultsDataclass,
                                           ProjectDataclass, RecursiveDict,
                                           RoundDataclass, TrialDataclass,
                                           TrialSplitDataclass, TrialStatus)
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, DateTime, Float, Integer, String, Table, MetaData, ForeignKey
