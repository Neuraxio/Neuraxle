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
import copy
import datetime
import json
import logging
import os
from abc import ABC, abstractclassmethod, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from json.encoder import JSONEncoder
from numbers import Number
from typing import (Any, Callable, Dict, Generic, Iterable, List, Optional,
                    Sequence, Tuple, Type, TypeVar, Union)

import numpy as np
from neuraxle.base import BaseStep, ExecutionContext, Flow, TrialStatus
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace, RecursiveDict)
from neuraxle.logging.logging import LOGGING_DATETIME_STR_FORMAT
from neuraxle.metaopt.observable import _ObservableRepo, _ObserverOfRepo

SubDataclassT = TypeVar('SubDataclassT', bound=Optional['BaseDataclass'])
ScopedLocationAttr = Union[str, int]
Slice = Sequence


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
    def __getitem__(
        self, key: Union[Type[SubDataclassT], Slice[Type[SubDataclassT]]]
    ) -> Union[ScopedLocationAttr, 'ScopedLocation']:
        """
        Get sublocation attr from the provided :class:`BaseMetadata` type,
        or otherwise get a slice of the same type, sliced from a :class:`BaseMetadata` type range of attributes to keep.
        """
        if isinstance(key, SubDataclassT):
            return getattr(self, dataclass_2_attr[key])

        elif isinstance(key, slice):
            if key.stop is None or key.start is not None:
                raise ValueError("Slice stop must be specified and slice start must be None.")

            idx: SubDataclassT = dataclass_2_attr.keys().index(key.stop) + 1
            return ScopedLocation(
                *self.as_list()[:idx]
            )

        raise ValueError(f"Invalid key type {key.__class__.__name__} for key {key}.")

    def __setitem__(self, key: Type[SubDataclassT], value: ScopedLocationAttr):
        """
        Set sublocation attr from the provided :class:`BaseMetadata` type.
        """
        # Throw value error if the key's type is not yet to be defined:
        curr_attr_to_set_idx: int = len(self)
        key_idx: int = list(dataclass_2_attr.keys()).index(key)
        if curr_attr_to_set_idx != key_idx:
            raise ValueError(
                f"{key} is not yet to be defined. Currently, "
                f"{dataclass_2_attr.keys()[curr_attr_to_set_idx]} is to be set.")
        key_attr_name: str = dataclass_2_attr[key]
        setattr(self, key_attr_name, value)

    def pop(self) -> ScopedLocationAttr:
        """
        Returns the last `not None` scoped location attribute and remove it from self.
        """
        for attr_name in reversed(dataclass_2_attr.values()):
            attr_value = getattr(self, attr_name)
            if attr_value is not None:
                setattr(self, attr_name, None)
                return attr_value

    def __len__(self) -> int:
        """
        Returns the number of not none scoped location attributes.
        """
        return len(self.as_list())

    def as_list(self) -> List[ScopedLocationAttr]:
        """
        Returns a list of the scoped location attributes.
        Item that has a value of None are not included in the list.

        :return: list of not none scoped location attributes
        """
        _list: List[ScopedLocationAttr] = [
            getattr(self, attr_name)
            for attr_name in dataclass_2_attr.values()
        ]
        _list_ret = []
        for i in _list + [None]:
            if i is not None:
                _list_ret.append(i)
            else:
                break
        return _list_ret


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

    @abstractmethod
    def get_id(self) -> ScopedLocationAttr:
        """
        Return the id of the current object, that is also the same attr that
        the object is located in the metadata ScopedLocation.
        """
        if self.__class__ in dataclass_2_attr:
            attr_name: str = dataclass_2_attr[self.__class__]
            attr = getattr(self, attr_name)
            return attr
        return None


class BaseTrialDataclassMixin:
    """
    Mixin class for :class:`TrialMetadata` and :class:`TrialSplitMetadata` that
    also must inherit from :class:`BaseMetadata`.
    """
    hyperparams: HyperparameterSamples
    status: TrialStatus = TrialStatus.PLANNED
    created_time: datetime.datetime = field(default_factory=datetime.datetime.now)
    start_time: datetime.datetime = field(default_factory=datetime.datetime.now)
    end_time: datetime.datetime = None
    log: str = ""

    def end(self, status: TrialStatus, add_to_log: str = "") -> 'BaseTrialDataclassMixin':
        self.status = status
        self.end_time = datetime.datetime.now()
        self.log += add_to_log
        return self


class RootMetadata(BaseDataclass['ProjectDataclass']):
    projects: OrderedDict[str, 'ProjectDataclass'] = field(
        default_factory=lambda: OrderedDict({"default_project": ProjectDataclass()}))

    def get_sublocation(self) -> OrderedDict[str, 'ProjectDataclass']:
        return self.projects

    def get_id(self) -> ScopedLocationAttr:
        return None


class ProjectDataclass(BaseDataclass['ClientDataclass']):
    project_name: str = "default_project"
    clients: OrderedDict[str, 'ClientDataclass'] = field(
        default_factory=lambda: OrderedDict({"default_client": ClientDataclass()}))

    def get_sublocation(self) -> OrderedDict[str, 'ClientDataclass']:
        return self.clients


class ClientDataclass(BaseDataclass['RoundDataclass']):
    client_name: str = "default_client"
    rounds: List['RoundDataclass'] = field(default_factory=list)
    # By default, the first metric is the main one.  # TODO: make it configurable, or in round?
    main_metric_name: str = None

    def get_sublocation(self) -> list['RoundDataclass']:
        return self.rounds


class RoundDataclass(BaseDataclass['TrialDataclass']):
    round_number: int = 0
    trials: List['TrialDataclass'] = field(default_factory=list)

    def get_sublocation(self) -> List['TrialDataclass']:
        return self.trials


class TrialDataclass(BaseTrialDataclassMixin, BaseDataclass['TrialSplitDataclass']):
    """
    This class is a data structure most often used under :class:`AutoML` to store information about a trial.
    This information is itself managed by the :class:`HyperparameterRepository` class
    and the :class:`Trial` class within the AutoML.
    """
    trial_number: int = 0
    validation_splits: List['TrialSplitDataclass'] = field(default_factory=list)

    def get_sublocation(self) -> List['TrialSplitDataclass']:
        return self.validation_splits


class TrialSplitDataclass(BaseTrialDataclassMixin, BaseDataclass['MetricResultsDataclass']):
    """
    TrialSplit object used by AutoML algorithm classes.
    """
    split_number: int = 0
    metric_results: OrderedDict[str, 'MetricResultsDataclass'] = field(default_factory=OrderedDict)
    introspection_data: RecursiveDict[str, Number] = field(default_factory=RecursiveDict)

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


dataclass_2_attr: OrderedDict[BaseDataclass, str] = OrderedDict([
    (ProjectDataclass, "project_name"),
    (ClientDataclass, "client_name"),
    (RoundDataclass, "round_number"),
    (TrialDataclass, "trial_number"),
    (TrialSplitDataclass, "split_number"),
    (MetricResultsDataclass, "metric_name"),
])


str_2_dataclass: OrderedDict[str, BaseDataclass] = OrderedDict([
    (ProjectDataclass.__name__, ProjectDataclass),
    (ClientDataclass.__name__, ClientDataclass),
    (RoundDataclass.__name__, RoundDataclass),
    (TrialDataclass.__name__, TrialDataclass),
    (TrialSplitDataclass.__name__, TrialSplitDataclass),
    (MetricResultsDataclass.__name__, MetricResultsDataclass),
    (RecursiveDict.__name__, RecursiveDict),
])


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


class HyperparamsRepository(_ObservableRepo[Tuple['HyperparamsRepository', BaseDataclass]], ABC):
    """
    Hyperparams repository that saves hyperparams, and scores for every AutoML trial.
    Cache folder can be changed to do different round numbers.

    .. seealso::
        :class:`AutoML`,
        :class:`Trainer`,
        :class:`~neuraxle.metaopt.data.trial.Trial`,
        :class:`InMemoryHyperparamsRepository`,
        :class:`HyperparamsJSONRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    @abstractmethod
    def get(self, scope: ScopedLocation) -> SubDataclassT:
        """
        Get metadata from scope.

        The fetched metadata will be the one that is the last item
        that is not a None in the provided scope.

        :param scope: scope to get metadata from.
        :return: metadata from scope.
        """
        raise NotImplementedError("Use a concrete class. This is an abstract class.")

    def load_trials(self, status: 'TrialStatus' = None) -> 'RoundManager':
        """
        Load all hyperparameter trials with their corresponding score.
        Sorted by creation date.
        Filtered by probided status.

        :param status: status to filter trials.
        :return: Trials (hyperparams, scores)
        """
        # TODO: delete this method.
        pass

    def save_trial(self, trial: 'TrialManager'):
        """
        Save trial, and notify trial observers.

        :param trial: trial to save.
        :return:
        """
        self._save_trial(trial)
        self.notify_next(value=(self, trial))  # notify a tuple of (repo, trial) to observers

    def _save_trial(self, trial: 'TrialManager'):
        """
        save trial.

        :param trial: trial to save.
        :return:
        """
        # TODO: delete this method.
        pass

    def get_best_hyperparams(self, loc: ScopedLocation) -> TrialDataclass:
        """
        Get best hyperparams.

        :param status: status to filter trials.
        :return: best hyperparams.
        """
        round: RoundDataclass = self.get(ScopedLocation(loc.as_list()[:3]))
        return round.best_trial

    @abstractmethod
    def get_logger_path(self, scope: ScopedLocation) -> str:
        """
        Get logger path from scope.

        :param scope: scope to get logger path from.
        :return: logger path with given scope.
        """

        raise NotImplementedError("Use a concrete class. This is an abstract class.")

    def _save_model(self, trial: 'TrialManager', pipeline: BaseStep, context: ExecutionContext):
        hyperparams = self.hyperparams.to_flat_dict()
        # TODO: ???
        trial_hash = trial.get_trial_id(hyperparams)
        pipeline.set_name(trial_hash).save(context, full_dump=True)

    def load_model(self, trial: 'TrialManager', context: ExecutionContext) -> BaseStep:
        """
        Load model in the trial hash folder.
        """
        # TODO: glob?
        trial_hash: str = trial.get_trial_id(self.hyperparams.to_flat_dict)
        return ExecutionContext.load(
            context=context,
            pipeline_name=trial_hash,
        )


class VanillaHyperparamsRepository(HyperparamsRepository):
    """
    Hyperparams repository that saves data AutoML-related info.
    """

    def __init__(
        self,
        cache_folder: str
    ):
        """
        :param cache_folder: folder to store trials.
        :param hyperparams_repo_class: class to use to save hyperparams.
        :param hyperparams_repo_kwargs: kwargs to pass to hyperparams_repo_class.
        """
        super().__init__()
        self.cache_folder = cache_folder
        self.root: RootMetadata = RootMetadata()

    def get(self, scope: ScopedLocation) -> SubDataclassT:
        """
        """

        if scope.project_name is None:
            return None
        else:
            proj: ProjectDataclass = self.get_project(scope.project_name)

        if scope.client_name is None:
            return proj
        else:
            client: ClientDataclass = self.get_client(proj, scope.client_name)

        if scope.round_name is None:
            return client
        else:
            round: RoundDataclass = self.get_round(client, scope.round_name)

        if scope.trial_name is None:
            return round
        else:
            trial: TrialDataclass = self.get_trial(round, scope.trial_name)

        if scope.split_name is None:
            return trial
        else:
            split: TrialSplitDataclass = self.get_split(trial, scope.split_name)

        if scope.metric_name is None:
            return split
        else:
            metric: MetricResultsDataclass = self.get_metric(split, scope.metric_name)

        return metric

    def get_logger_path(self, scope: ScopedLocation) -> str:
        scoped_path: str = self.get_scoped_path(scope)
        return os.path.join(scoped_path, 'log.txt')

    def get_scoped_path(self, scope: ScopedLocation) -> str:
        return os.path.join(self.cache_folder, *scope.as_list())


class InMemoryHyperparamsRepository(HyperparamsRepository):
    """
    In memory hyperparams repository that can print information about trials.
    Useful for debugging.
    """

    def __init__(self, pre_made_trials: Optional['RoundManager'] = None):
        HyperparamsRepository.__init__(self)
        self.trials: RoundManager = pre_made_trials if pre_made_trials is not None else RoundManager()

    def load_trials(self, status: 'TrialStatus' = None) -> 'RoundManager':
        """
        Load all trials with the given status.

        :param status: trial status
        :return: list of trials
        """
        return self.trials.filter(status)

    def _save_trial(self, trial: 'TrialManager'):
        """
        Save trial.

        :param trial: trial to save
        :return:
        """
        self.trials.append(trial)


class BaseHyperparameterOptimizer(ABC):

    def __init__(self, main_metric_name: str = None):
        """
        :param main_metric_name: if None, pick first metric from the metrics callback.
        """
        self.main_metric_name = main_metric_name

    @abstractmethod
    def find_next_best_hyperparams(self, round_scope) -> HyperparameterSamples:
        """
        Find the next best hyperparams using previous trials.

        :param round_scope: round scope
        :return: next hyperparameter samples to train on
        """
        # TODO: revise arguments.
        raise NotImplementedError()


class AutoMLFlow(Flow):

    def __init__(
        self,
        repo: HyperparamsRepository,
        logger: logging.Logger = None,
        loc: ScopedLocation = None,
    ):
        super().__init__(logger=logger)
        self.repo: HyperparamsRepository = repo
        self.loc: ScopedLocation = loc or ScopedLocation()

        self.synchroneous()

    @staticmethod
    def from_flow(flow: Flow, repo: HyperparamsRepository) -> 'AutoMLFlow':
        f = AutoMLFlow(
            repo=repo,
            logger=flow.logger,  # TODO: loc?
            # loc=flow.loc,
        )
        f._lock = flow.synchroneous()  # TODO: lock to be in AutoMLContext instead.
        return f

    def with_new_loc(self, loc: ScopedLocation):
        """
        Create a new AutoMLFlow with a new ScopedLocation.

        :param loc: ScopedLocation
        :return: AutoMLFlow
        """
        return AutoMLFlow(self.repo, self.logger, loc)

    def copy(self):
        f = AutoMLFlow(
            repo=self.repo,
            logger=self.logger,
            loc=copy.copy(self.loc)
        )
        f._lock = self._lock
        return f

    def log_model(self, model: 'BaseStep'):
        # TODO: move to scoped actions below.
        raise NotImplementedError("")
        self.repo.save_model(model)
        return super().log_model(model)


class AutoMLContext(ExecutionContext):

    @staticmethod
    def from_context(context: ExecutionContext, repo: HyperparamsRepository) -> 'AutoMLContext':
        """
        Create a new AutoMLContext from an ExecutionContext.

        :param context: ExecutionContext
        """
        context = context.copy()
        flow = AutoMLFlow.from_flow(context.flow, repo)

        new_context = AutoMLContext(
            root=context.root,
            flow=AutoMLFlow.from_flow(flow, repo),
            execution_phase=context.execution_phase,
            execution_mode=context.execution_mode,
            stripped_saver=context.stripped_saver,
            parents=context.parents,
            services=context.services,
        )
        # TODO: repo in context or just in flow?
        new_context.register_service(HyperparamsRepository, repo)
        return new_context

    # TODO: @lock in repo.
    @property
    def lock(self):
        return self.flow._lock

    @property
    def repo(self) -> HyperparamsRepository:
        return self.get_service(HyperparamsRepository)

    @property
    def loc(self) -> ScopedLocation:
        return self.flow.loc

    def push_attr(self, name: Type[SubDataclassT], value: ScopedLocationAttr) -> 'AutoMLContext':
        """
        Push a new attribute into the ScopedLocation.

        :param name: attribute name
        :param value: attribute value
        :return: an AutoMLContext copy with the new loc attribute.
        """
        new_self: AutoMLContext = self.copy()
        new_self.flow.loc[name] = value
        return new_self
