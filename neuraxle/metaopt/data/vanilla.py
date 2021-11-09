"""
Neuraxle's Base Hyperparameter Repository Classes
=================================================
Data objects and related repositories used by AutoML.

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
import json
from collections import OrderedDict
from numbers import Number
from dataclasses import dataclass, field
from enum import Enum
from json.encoder import JSONEncoder
from typing import Any, Callable, Dict, Iterable, List, Tuple, Type

import numpy as np
from neuraxle.hyperparams.space import HyperparameterSamples, RecursiveDict


class TrialStatus(Enum):
    """
    Enum of the possible states of a trial.
    """
    PLANNED = 'planned'
    RUNNING = 'running'
    ABORTED = 'aborted'  # TODO: consider aborted.
    FAILED = 'failed'
    SUCCESS = 'success'


@dataclass
class BaseMetadata:
    # TODO: from json, to json?
    pass


class ProjectMetadata(BaseMetadata):
    name: str = ""
    clients: List['ClientMetadata'] = field(default_factory=list)


class ClientMetadata(BaseMetadata):
    rounds: List['RoundMetadata'] = field(default_factory=list)
    main_metric_name: str = None  # By default, the first metric is the main one.  # TODO: make it configurable.


class RoundMetadata(BaseMetadata):
    trials: List['TrialMetadata'] = field(default_factory=list)


class BaseTrialMetadata(BaseMetadata):
    """
    Base class for :class:`TrialMetadata` and :class:`TrialSplitMetadata`.
    """
    hyperparams: HyperparameterSamples
    start_time: datetime.datetime = None
    end_time: datetime.datetime = None
    log: str = None


class TrialMetadata(BaseTrialMetadata):
    """
    Trial object used by AutoML algorithm classes.
    """
    trial_number: int = 0
    status: TrialStatus = TrialStatus.PLANNED
    validation_splits: List['TrialSplit'] = field(default_factory=list)


class TrialSplitMetadata(BaseTrialMetadata):
    """
    TrialSplit object used by AutoML algorithm classes.
    """
    split_number: int = 0
    introspection_data: RecursiveDict[str, Number] = None
    metric_results: OrderedDict[str, 'MetricResultMetadata'] = field(default_factory=OrderedDict)


class MetricResultMetadata(BaseMetadata):
    """
    MetricResult object used by AutoML algorithm classes.
    """
    train_values: List[float] = field(default_factory=list)
    validation_values: List[float] = field(default_factory=list)
    higher_score_is_better: bool = True


metadata_classes = {
    ProjectMetadata.__name__: ProjectMetadata,
    ClientMetadata.__name__: ClientMetadata,
    RoundMetadata.__name__: RoundMetadata,
    TrialMetadata.__name__: TrialMetadata,
    TrialSplitMetadata.__name__: TrialSplitMetadata,
    RecursiveDict.__name__: RecursiveDict,
    MetricResultMetadata.__name__: MetricResultMetadata,
}


def object_decoder(obj):
    if '__type__' in obj and obj['__type__'] in metadata_classes:
        cls: Type = metadata_classes[obj['__type__']]
        kwargs = dict(obj)
        del kwargs['__type__']
        return cls(**kwargs)
    return obj


class MetadataJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, RecursiveDict):
            return obj.to_flat_dict()
        elif type(obj) in metadata_classes:
            return {'__type__': type(obj).__name__, **obj.asdict()}  # TODO: **self.default(obj) ?
        else:
            return JSONEncoder.default(self, obj)


def to_json(obj: str) -> str:
    return json.dumps(obj, cls=MetadataJSONEncoder)


def from_json(json: str) -> str:
    return json.loads(json, object_pairs_hook=OrderedDict, object_hook=object_decoder)
