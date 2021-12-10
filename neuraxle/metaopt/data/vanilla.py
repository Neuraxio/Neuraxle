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
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from json.encoder import JSONEncoder
from numbers import Number
from typing import (Any, Callable, Dict, Generic, Iterable, List, Optional,
                    Sequence, Tuple, Type, TypeVar, Union)

import numpy as np
from neuraxle.base import (BaseStep, ExecutionContext, Flow, TrialStatus,
                           synchroneous_flow_method)
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace, RecursiveDict)
from neuraxle.logging.logging import LOGGER_FORMAT, LOGGING_DATETIME_STR_FORMAT
from neuraxle.metaopt.observable import _ObservableRepo, _ObserverOfRepo

SubDataclassT = TypeVar('SubDataclassT', bound=Optional['BaseDataclass'])
ScopedLocationAttrInt = int
ScopedLocationAttrStr = str
ScopedLocationAttr = Union[str, int]
Slice = Sequence

DEFAULT_PROJECT: ScopedLocationAttrStr = "default_project"
DEFAULT_CLIENT: ScopedLocationAttrStr = "default_client"


@dataclass(order=True)
class ScopedLocation:
    """
    A location in the metadata tree.
    """
    project_name: ScopedLocationAttrStr = None
    client_name: ScopedLocationAttrStr = None
    round_number: ScopedLocationAttrInt = None
    trial_number: ScopedLocationAttrInt = None
    split_number: ScopedLocationAttrInt = None
    metric_name: ScopedLocationAttrStr = None

    # get the field value from BaseMetadata subclass:
    def __getitem__(
        self, key: Union[Type['BaseDataclass'], Slice[Type['BaseDataclass']]]
    ) -> Union[ScopedLocationAttr, 'ScopedLocation']:
        """
        Get sublocation attr from the provided :class:`BaseMetadata` type,
        or otherwise get a slice of the same type, sliced from a :class:`BaseMetadata` type range of attributes to keep.
        """
        if key is None:
            return None

        if isinstance(key, BaseDataclass):
            return getattr(self, key._id_attr_name)

        if key in dataclass_2_id_attr.keys():
            return getattr(self, dataclass_2_id_attr[key])

        if isinstance(key, slice):
            if key.stop is None or key.start is not None:
                raise ValueError("Slice stop must be specified and slice start must be None.")

            idx: BaseDataclass = dataclass_2_id_attr.keys().index(key.stop) + 1
            return ScopedLocation(
                *self.as_list()[:idx]
            )

        raise ValueError(f"Invalid key type {key.__class__.__name__} for key {key}.")

    def with_dc(self, dc: 'BaseDataclass') -> 'ScopedLocation':
        """
        Returns a new :class:`ScopedLocation` with the provided :class:`BaseDataclass` (dc) type's id added.
        """
        if isinstance(dc, RootDataclass):
            return ScopedLocation()
        self_copy = copy.deepcopy(self)
        self_copy[dc.__class__] = dc.get_id()
        return self_copy

    def with_id(self, _id: ScopedLocationAttr) -> 'ScopedLocation':
        """
        Returns a new :class:`ScopedLocation` with the provided id added.
        """
        return ScopedLocation(*(self.as_list() + [_id]))

    def __setitem__(self, key: Type['BaseDataclass'], value: ScopedLocationAttr):
        """
        Set sublocation attr from the provided :class:`BaseMetadata` type.
        """
        # Throw value error if the key's type is not yet to be defined:
        curr_attr_to_set_idx: int = len(self)
        key_idx: int = list(dataclass_2_id_attr.keys()).index(key)
        if curr_attr_to_set_idx != key_idx:
            raise ValueError(
                f"{key} is not yet to be defined. Currently, "
                f"{dataclass_2_id_attr.keys()[curr_attr_to_set_idx]} is the next to be set.")
        key_attr_name: str = dataclass_2_id_attr[key]
        setattr(self, key_attr_name, value)

    def pop(self) -> ScopedLocationAttr:
        """
        Returns the last `not None` scoped location attribute and remove it from self.
        """
        for attr_name in reversed(dataclass_2_id_attr.values()):
            attr_value = getattr(self, attr_name)
            if attr_value is not None:
                setattr(self, attr_name, None)
                return attr_value

    def popped(self) -> 'ScopedLocation':
        """
        Returns a new :class:`ScopedLocation` with the last `not None` scoped location attribute removed.
        """
        return ScopedLocation(*self.to_list()[:-1])

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
            for attr_name in dataclass_2_id_attr.values()
        ]
        _list_ret = []
        for i in _list + [None]:
            if i is not None:
                _list_ret.append(i)
            else:
                break
        return _list_ret

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(self.as_list())})"

    def __str__(self) -> str:
        return self.__repr__()


@dataclass(order=True)
class BaseDataclass(Generic[SubDataclassT], ABC):
    # TODO: from json, to json?

    @property
    def _id_attr_name(self) -> str:
        return dataclass_2_id_attr[self.__class__]

    @property
    def _sublocation_attr_name(self) -> str:
        return dataclass_2_subloc_attr[self.__class__]

    def get_id(self) -> ScopedLocationAttr:
        return getattr(self, self._id_attr_name)

    def set_id(self, _id: ScopedLocationAttr) -> None:
        setattr(self, self._id_attr_name, _id)

    def get_sublocation(self) -> Union[List[SubDataclassT], OrderedDict[str, SubDataclassT]]:
        return getattr(self, dataclass_2_subloc_attr[self.__class__])

    def set_sublocation(self, sublocation: Union[List[SubDataclassT], OrderedDict[str, SubDataclassT]]) -> None:
        setattr(self, dataclass_2_subloc_attr[self.__class__], sublocation)

    def __getitem__(self, loc: ScopedLocation) -> 'SubDataclassT':
        subdataklass: Type[SubDataclassT] = dataclass_2_subdataclass[self.__class__]
        located_sub_attr: ScopedLocationAttr = loc[subdataklass]
        if located_sub_attr is None:
            return self
        last_self_attr = self.get_sublocation()
        return last_self_attr[located_sub_attr][loc]

    def __len__(self) -> int:
        return len(self.get_sublocation())

    @abstractmethod
    def store(self, dc: SubDataclassT) -> ScopedLocationAttr:
        raise NotImplementedError("Must use mixins.")

    @abstractmethod
    def shallow(self) -> 'BaseDataclass':
        raise NotImplementedError("Must use mixins.")

    @abstractmethod
    def get_sublocation_values(self) -> List[SubDataclassT]:
        raise NotImplementedError("Must use mixins.")


@dataclass(order=True)
class DataclassHasOrderedDictMixin:
    def get_sublocation(self) -> OrderedDict[ScopedLocationAttrStr, SubDataclassT]:
        ret = super().get_sublocation()
        if not isinstance(ret, OrderedDict):
            raise ValueError(f"{ret} should be an OrderedDict. Inconsistant object.")
        return ret

    def set_sublocation(self, sublocation: OrderedDict[ScopedLocationAttrStr, SubDataclassT]) -> None:
        if not isinstance(sublocation, OrderedDict):
            raise ValueError(f"{sublocation} should be an OrderedDict. Cannot set.")
        setattr(self, self._sublocation_attr_name, sublocation)

    def get_sublocation_values(self) -> List[SubDataclassT]:
        return self.get_sublocation().values()

    def store(self, dc: SubDataclassT) -> ScopedLocationAttrStr:
        self.get_sublocation()[dc.get_id()] = dc
        return dc.get_id()

    def shallow(self) -> 'BaseDataclass':
        self_copy = copy.copy(self)
        new_sublocation = OrderedDict([(k, None) for k in self.get_sublocation().keys()])
        self_copy.set_sublocation(new_sublocation)
        # Deep copy only after trimming sublocations.
        return copy.deepcopy(self_copy)


@dataclass(order=True)
class DataclassHasListMixin:
    def get_sublocation(self) -> List[SubDataclassT]:
        return super().get_sublocation()

    def set_sublocation(self, sublocation: List[SubDataclassT]) -> None:
        setattr(self, self._sublocation_attr_name, sublocation)

    def get_sublocation_values(self) -> List[SubDataclassT]:
        return self.get_sublocation()

    def store(self, dc: SubDataclassT) -> ScopedLocationAttrInt:
        _id = dc.get_id()
        if _id == self.get_next_i():
            self.get_sublocation().append(dc)
        elif _id < self.get_next_i():
            self.get_sublocation()[dc.get_id()] = dc
        else:
            raise ValueError(f"{dc} has id {dc.get_id()} which is greater than the next id {self.get_next_i()}.")
        return _id

    def shallow(self) -> 'BaseDataclass':
        self_copy = copy.copy(self)
        new_sublocation = [None for _ in range(len(self))]
        self_copy.set_sublocation(new_sublocation)
        # Deep copy only after trimming sublocations.
        return copy.deepcopy(self_copy)

    def get_next_i(self) -> ScopedLocationAttrInt:
        return len(self.get_sublocation())


@dataclass(order=True)
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
        # TODO: use this method.
        self.status = status
        self.end_time = datetime.datetime.now()
        self.log += add_to_log
        return self


@dataclass(order=True)
class RootDataclass(DataclassHasOrderedDictMixin, BaseDataclass['ProjectDataclass']):
    projects: OrderedDict[str, 'ProjectDataclass'] = field(
        default_factory=lambda: OrderedDict({DEFAULT_PROJECT: ProjectDataclass()}))

    @property
    def _id_attr_name(self) -> str:
        return None

    @property
    def _sublocation_attr_name(self) -> str:
        return "projects"

    def get_id(self) -> ScopedLocationAttr:
        return None

    def get_sublocation(self) -> OrderedDict[str, SubDataclassT]:
        return self.projects


@dataclass(order=True)
class ProjectDataclass(DataclassHasOrderedDictMixin, BaseDataclass['ClientDataclass']):
    project_name: str = DEFAULT_PROJECT
    clients: OrderedDict[str, 'ClientDataclass'] = field(
        default_factory=lambda: OrderedDict({DEFAULT_CLIENT: ClientDataclass()}))

    def get_id(self) -> ScopedLocationAttrStr:
        return self.project_name


@dataclass(order=True)
class ClientDataclass(DataclassHasListMixin, BaseDataclass['RoundDataclass']):
    client_name: ScopedLocationAttrStr = DEFAULT_CLIENT
    rounds: List['RoundDataclass'] = field(default_factory=list)
    # By default, the first metric is the main one.  # TODO: make it configurable, or in round?
    main_metric_name: str = None

    def get_id(self) -> ScopedLocationAttrStr:
        return self.client_name


@dataclass(order=True)
class RoundDataclass(DataclassHasListMixin, BaseDataclass['TrialDataclass']):
    round_number: ScopedLocationAttrInt = 0
    trials: List['TrialDataclass'] = field(default_factory=list)

    def get_id(self) -> ScopedLocationAttrInt:
        return self.round_number


@dataclass(order=True)
class TrialDataclass(DataclassHasListMixin, BaseTrialDataclassMixin, BaseDataclass['TrialSplitDataclass']):
    """
    This class is a data structure most often used under :class:`AutoML` to store information about a trial.
    This information is itself managed by the :class:`HyperparameterRepository` class
    and the :class:`Trial` class within the AutoML.
    """
    trial_number: ScopedLocationAttrInt = 0
    validation_splits: List['TrialSplitDataclass'] = field(default_factory=list)

    def get_id(self) -> ScopedLocationAttrInt:
        return self.trial_number


@dataclass(order=True)
class TrialSplitDataclass(DataclassHasOrderedDictMixin, BaseTrialDataclassMixin, BaseDataclass['MetricResultsDataclass']):
    """
    TrialSplit object used by AutoML algorithm classes.
    """
    split_number: ScopedLocationAttrInt = 0
    metric_results: OrderedDict[str, 'MetricResultsDataclass'] = field(default_factory=OrderedDict)
    # introspection_data: RecursiveDict[str, Number] = field(default_factory=RecursiveDict)

    def get_id(self) -> ScopedLocationAttrInt:
        return self.split_number


@dataclass(order=True)
class MetricResultsDataclass(DataclassHasListMixin, BaseDataclass[float]):
    """
    MetricResult object used by AutoML algorithm classes.
    """
    metric_name: ScopedLocationAttrStr = "main"
    validation_values: List[float] = field(default_factory=list)  # one per epoch.
    train_values: List[float] = field(default_factory=list)
    higher_score_is_better: bool = True

    def get_id(self) -> ScopedLocationAttrStr:
        return self.metric_name


dataclass_2_id_attr: OrderedDict[BaseDataclass, str] = OrderedDict([
    (ProjectDataclass, "project_name"),
    (ClientDataclass, "client_name"),
    (RoundDataclass, "round_number"),
    (TrialDataclass, "trial_number"),
    (TrialSplitDataclass, "split_number"),
    (MetricResultsDataclass, "metric_name"),
])

dataclass_2_subloc_attr: OrderedDict[BaseDataclass, str] = OrderedDict([
    (RootDataclass, "projects"),
    (ProjectDataclass, "clients"),
    (ClientDataclass, "rounds"),
    (RoundDataclass, "trials"),
    (TrialDataclass, "validation_splits"),
    (TrialSplitDataclass, "metric_results"),
    (MetricResultsDataclass, "validation_values"),
])

dataclass_2_subdataclass: OrderedDict[BaseDataclass, SubDataclassT] = OrderedDict([
    (RootDataclass, ProjectDataclass),
    (ProjectDataclass, ClientDataclass),
    (ClientDataclass, RoundDataclass),
    (RoundDataclass, TrialDataclass),
    (TrialDataclass, TrialSplitDataclass),
    (TrialSplitDataclass, MetricResultsDataclass),
    (MetricResultsDataclass, None),
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
    def load(self, scope: ScopedLocation, deep=False) -> SubDataclassT:
        """
        Get metadata from scope.

        The fetched metadata will be the one that is the last item
        that is not a None in the provided scope.

        :param scope: scope to get metadata from.
        :return: metadata from scope.
        """
        raise NotImplementedError("Use a concrete class. This is an abstract class.")

    @abstractmethod
    def save(self, _dataclass: SubDataclassT, scope: ScopedLocation, deep=False) -> None:
        """
        Save metadata to scope.

        :param metadata: metadata to save.
        :param scope: scope to save metadata to.
        :param deep: if True, save metadata's sublocations recursively.
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
        round: RoundDataclass = self.load(ScopedLocation(loc.as_list()[:3]))
        return round.best_trial

    @abstractmethod
    def get_scoped_logger_path(self, scope: ScopedLocation) -> str:
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
        self.cache_folder = os.path.join(cache_folder, self.__class__.__name__)
        self.root: RootDataclass = RootDataclass()

    def load(self, scope: ScopedLocation, deep=False) -> SubDataclassT:
        ret: BaseDataclass = self.root[scope]
        if not deep:
            ret = ret.shallow()
        return copy.deepcopy(ret)

    def save(self, _dataclass: SubDataclassT, scope: ScopedLocation, deep=False) -> None:
        _dataclass: SubDataclassT = copy.deepcopy(_dataclass)
        _id_from_scope: ScopedLocationAttr = scope[_dataclass.__class__]
        scope = scope[:_dataclass.__class__]  # Sanitizing scope to dtype loc.

        if _id_from_scope is not None:
            if _id_from_scope != _dataclass.get_id():
                raise ValueError(
                    f"The scope `{scope}` with {_dataclass.__class__.__name__} id `{_id_from_scope}` does not match the provided dataclass id `{_dataclass.get_id()}` for `{_dataclass}`."
                )
            scope.pop()
            # else check if the scope is at least of the good class length:
        elif len(scope) != dataclass_2_id_attr.keys().index(_dataclass.__class__):
            raise ValueError(
                f"The scope `{scope}` is not of the good length for dataclass type `{_dataclass.__class__.__name__}`."
            )

        if not deep:
            # Reassign saved sublocation to new dataclass:
            prev_dc: SubDataclassT = self.get(scope.with_dc(_dataclass), deep=True)
            _dataclass.set_sublocation(prev_dc.get_sublocation())

        self.root[scope].store(_dataclass)

    def get_scoped_logger_path(self, scope: ScopedLocation) -> str:
        scoped_path: str = self.get_scoped_path(scope)
        return os.path.join(scoped_path, 'log.txt')

    def get_scoped_path(self, scope: ScopedLocation) -> str:
        return os.path.join(self.cache_folder, *scope.as_list())


class InMemoryHyperparamsRepository(HyperparamsRepository):
    """
    In memory hyperparams repository that can print information about trials.
    Useful for debugging.
    """

    def __init__(self, pre_made_trials: Optional['Round'] = None):
        HyperparamsRepository.__init__(self)
        self.trials: Round = pre_made_trials if pre_made_trials is not None else Round()

    def load_trials(self, status: 'TrialStatus' = None) -> 'Round':
        """
        Load all trials with the given status.

        :param status: trial status
        :return: list of trials
        """
        return self.trials.filter(status)

    def _save_trial(self, trial: 'Trial'):
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

    def get_main_metric_name(self) -> str:
        return self.main_metric_name

    @abstractmethod
    def find_next_best_hyperparams(self, round) -> HyperparameterSamples:
        """
        Find the next best hyperparams using previous trials, that is the
        whole :class:`neuraxle.metaopt.data.aggregate.Round`.

        :param round: a :class:`neuraxle.metaopt.data.aggregate.Round`
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

    def add_file_handler_to_logger(
        self,
        logging_file: str
    ) -> logging.Logger:
        os.makedirs(os.path.dirname(logging_file), exist_ok=True)
        formatter = logging.Formatter(
            fmt=LOGGER_FORMAT, datefmt=LOGGING_DATETIME_STR_FORMAT)
        file_handler = logging.FileHandler(filename=logging_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def free_logger_files(self):
        """
        Remove file handlers from logger to free file lock on Windows.
        """
        for h in self.logger.handlers:
            if isinstance(h, logging.FileHandler):
                self.logger.removeHandler(h)


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
        return self.flow.synchroneous()

    @property
    def loc(self) -> ScopedLocation:
        return self.flow.loc

    @property
    def repo(self) -> HyperparamsRepository:
        return self.get_service(HyperparamsRepository)

    def push_attr(self, subdataclass: SubDataclassT) -> 'AutoMLContext':
        """
        Push a new attribute into the ScopedLocation.

        :param name: attribute name
        :param value: attribute value
        :return: an AutoMLContext copy with the new loc attribute.
        """
        new_self: AutoMLContext = self.copy()
        new_self.flow.loc = new_self.flow.loc.with_dc(subdataclass)
        return new_self
