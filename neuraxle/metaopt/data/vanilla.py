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

from abc import ABC, abstractclassmethod, abstractmethod
import datetime
import json
from collections import OrderedDict
from numbers import Number
from dataclasses import dataclass, field
from enum import Enum
from json.encoder import JSONEncoder
from typing import Any, Callable, Dict, Generic, Iterable, List, Tuple, Type, TypeVar, Union, Optional

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


SubDataclassT = TypeVar('SubMetadataT', bound=Optional['BaseMetadata'])
ScopedLocationAttr = Union[str, int]


@dataclass(order=True)
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

    # get the field value from BaseMetadata subclass:
    def __getitem__(self, key: Type[SubDataclassT]) -> ScopedLocationAttr:
        """
        Get sublocation attr from the provided :class:`BaseMetadata` type.
        """
        return {
            ProjectDataclass: self.project_name,
            ClientDataclass: self.client_name,
            RoundDataclass: self.round_number,
            TrialDataclass: self.trial_number,
            TrialSplitDataclass: self.split_number,
            MetricResultsDataclass: self.metric_name,
        }[key]

    def as_list(self) -> List[ScopedLocationAttr]:
        """
        Returns a list of the scoped location attributes.
        Item that has a value of None are not included in the list.

        :return: list of not none scoped location attributes
        """
        return [i for i in [
            self.project_name,
            self.client_name,
            self.round_number,
            self.trial_number,
            self.split_number,
            self.metric_name,
        ] if i is not None]


@dataclass(order=True)
class BaseDataclass(Generic[SubDataclassT], ABC):
    # TODO: from json, to json?

    def __getitem__(self, loc: ScopedLocation) -> 'SubDataclassT':
        located_attr: ScopedLocationAttr = loc[self.__class__]
        if located_attr is None:
            return self
        last_self_attr = self.get_sublocation()
        return last_self_attr[located_attr][loc]

    @abstractmethod
    def get_sublocation(self) -> Union[List[SubDataclassT], OrderedDict[str, SubDataclassT]]:
        raise NotImplementedError("Must be implemented in subclasses.")


class BaseTrialDataclassMixin:
    """
    Mixin class for :class:`TrialMetadata` and :class:`TrialSplitMetadata` that
    also must inherit from :class:`BaseMetadata`.
    """
    hyperparams: HyperparameterSamples
    start_time: datetime.datetime = None
    end_time: datetime.datetime = None
    log: str = None


class RootMetadata(BaseDataclass['ProjectDataclass']):
    projects: OrderedDict[str, 'ProjectDataclass'] = field(
        default_factory=lambda: OrderedDict({"default_project": ProjectDataclass()}))

    def get_sublocation(self) -> OrderedDict[str, 'ProjectDataclass']:
        return self.projects


class ProjectDataclass(BaseDataclass['ClientDataclass']):
    project_name: str = "default_project"
    clients: OrderedDict[str, 'ClientDataclass'] = field(
        default_factory=lambda: OrderedDict({"default_client": ClientDataclass()}))

    def get_sublocation(self) -> OrderedDict[str, 'ClientDataclass']:
        return self.clients


class ClientDataclass(BaseDataclass['RoundDataclass']):
    client_name: str = "default_client"
    rounds: List['RoundDataclass'] = field(default_factory=list)
    main_metric_name: str = None  # By default, the first metric is the main one.  # TODO: make it configurable.

    def get_sublocation(self) -> list['RoundDataclass']:
        return self.rounds


class RoundDataclass(BaseDataclass['TrialDataclass']):
    round_number: int = 0
    trials: List['TrialDataclass'] = field(default_factory=list)

    def get_sublocation(self) -> List['TrialDataclass']:
        return self.trials


class TrialDataclass(BaseTrialDataclassMixin, BaseDataclass['TrialSplitDataclass']):
    """
    Trial object used by AutoML algorithm classes.
    """
    trial_number: int = 0
    validation_splits: List['TrialSplitDataclass'] = field(default_factory=list)
    status: TrialStatus = TrialStatus.PLANNED

    def get_sublocation(self) -> List['TrialSplitDataclass']:
        return self.validation_splits


class TrialSplitDataclass(BaseTrialDataclassMixin, BaseDataclass['MetricResultsDataclass']):
    """
    TrialSplit object used by AutoML algorithm classes.
    """
    split_number: int = 0
    metric_results: OrderedDict[str, 'MetricResultsDataclass'] = field(default_factory=OrderedDict)
    introspection_data: RecursiveDict[str, Number] = None

    def get_sublocation(self) -> OrderedDict[str, 'MetricResultsDataclass']:
        return self.metric_results


class MetricResultsDataclass(BaseDataclass[float]):
    """
    MetricResult object used by AutoML algorithm classes.
    """
    metric_name: str = "main"
    validation_values: List[float] = field(default_factory=list)
    train_values: List[float] = field(default_factory=list)
    higher_score_is_better: bool = True

    def get_sublocation(self) -> list[float]:
        return self.validation_values


str_2_dataclass = {
    ProjectDataclass.__name__: ProjectDataclass,
    ClientDataclass.__name__: ClientDataclass,
    RoundDataclass.__name__: RoundDataclass,
    TrialDataclass.__name__: TrialDataclass,
    TrialSplitDataclass.__name__: TrialSplitDataclass,
    MetricResultsDataclass.__name__: MetricResultsDataclass,
    RecursiveDict.__name__: RecursiveDict,
}


def object_decoder(obj):
    if '__type__' in obj and obj['__type__'] in str_2_dataclass:
        cls: Type = str_2_dataclass[obj['__type__']]
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
        elif type(obj) in str_2_dataclass:
            return {'__type__': type(obj).__name__, **obj.asdict()}  # TODO: **self.default(obj) ?
        else:
            return JSONEncoder.default(self, obj)


def to_json(obj: str) -> str:
    return json.dumps(obj, cls=MetadataJSONEncoder)


def from_json(json: str) -> str:
    return json.loads(json, object_pairs_hook=OrderedDict, object_hook=object_decoder)
