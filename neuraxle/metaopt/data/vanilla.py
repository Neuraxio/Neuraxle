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

from abc import ABC
import datetime
import json
from collections import OrderedDict
from numbers import Number
from dataclasses import dataclass, field
from enum import Enum
from json.encoder import JSONEncoder
from typing import Any, Callable, Dict, Generic, Iterable, List, Tuple, Type, TypeVar

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
class ScopedLocation:
    """
    A location in the metadata tree.
    """
    project_name: str = None
    client_name: str = None
    round_number: int = None
    trial_number: int = None
    split_number: int = None
    metric_name: str = None


SubMetadataT = TypeVar('SubMetadataT', bound='BaseMetadata')


@dataclass
class BaseMetadata(Generic[SubMetadataT], ABC):
    # TODO: from json, to json?
    pass

    def __getitem__(self, loc: ScopedLocation) -> 'SubMetadataT':
        raise NotImplementedError("This is an abstract class. Use a concrete class.")


class RootMetadata(BaseMetadata['ProjectMetadata']):
    projects: OrderedDict[str, 'ProjectMetadata'] = field(default_factory=OrderedDict)

    def __getitem__(self, loc: ScopedLocation) -> 'ProjectMetadata':
        if loc.project_name is None:
            return self
            # TODO: do this behavior for all the other cases.
        return self.projects[loc.project_name]


class ProjectMetadata(BaseMetadata['ClientMetadata']):
    project_name: str = "default_project"
    clients: OrderedDict[str, 'ClientMetadata'] = field(default_factory=list)

    def __getitem__(self, loc: ScopedLocation):
        return self.clients[loc.client_name]


class ClientMetadata(BaseMetadata['RoundMetadata']):
    client_name: str = "default_client"
    rounds: List['RoundMetadata'] = field(default_factory=list)
    main_metric_name: str = None  # By default, the first metric is the main one.  # TODO: make it configurable.

    def __getitem__(self, loc: ScopedLocation):
        return self.rounds[loc.round_name]


class RoundMetadata(BaseMetadata['TrialMetadata']):
    round_number: int = 0
    trials: List['TrialMetadata'] = field(default_factory=list)

    def __getitem__(self, loc: ScopedLocation):
        return self.trials[loc.trial_name]


class BaseTrialMetadataMixin:
    """
    Base class for :class:`TrialMetadata` and :class:`TrialSplitMetadata`.
    """
    hyperparams: HyperparameterSamples
    start_time: datetime.datetime = None
    end_time: datetime.datetime = None
    log: str = None


class TrialMetadata(BaseTrialMetadataMixin, BaseMetadata['TrialSplitMetadata']):
    """
    Trial object used by AutoML algorithm classes.
    """
    trial_number: int = 0
    status: TrialStatus = TrialStatus.PLANNED
    validation_splits: List['TrialSplitMetadata'] = field(default_factory=list)

    def __getitem__(self, loc: ScopedLocation):
        return self.validation_splits[loc.split_name]


class TrialSplitMetadata(BaseTrialMetadataMixin, BaseMetadata['MetricResultMetadata']):
    """
    TrialSplit object used by AutoML algorithm classes.
    """
    split_number: int = 0
    introspection_data: RecursiveDict[str, Number] = None
    metric_results: OrderedDict[str, 'MetricResultMetadata'] = field(default_factory=OrderedDict)

    def __getitem__(self, loc: ScopedLocation):
        return self.metric_name[loc.metric_name]


class MetricResultMetadata(BaseMetadata['NoneType']):
    """
    MetricResult object used by AutoML algorithm classes.
    """
    metric_name: str = "main"
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
