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
from dataclasses import asdict, dataclass, field, fields
from json.encoder import JSONEncoder
from typing import (Any, Callable, Dict, Generic, Iterable, List, Optional,
                    Sequence, Tuple, Type, TypeVar, Union)

import neuraxle.metaopt.data.aggregates as agg
import numpy as np
from neuraxle.base import (CX, BaseService, BaseStep, ContextLock, Flow,
                           TrialStatus)
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace, RecursiveDict)
from neuraxle.logging.logging import (LOGGER_FORMAT,
                                      LOGGING_DATETIME_STR_FORMAT,
                                      NeuraxleLogger)
from neuraxle.logging.warnings import RaiseDeprecatedClass
from neuraxle.metaopt.observable import _ObservableRepo, _ObserverOfRepo

SubDataclassT = TypeVar('SubDataclassT', bound=Optional['BaseDataclass'])
ScopedLocationAttrInt = int
ScopedLocationAttrStr = str
ScopedLocationAttr = Union[str, int]
Slice = Sequence

DEFAULT_PROJECT: ScopedLocationAttrStr = "default_project"
DEFAULT_CLIENT: ScopedLocationAttrStr = "default_client"


@dataclass(order=True)
class ScopedLocation(BaseService):
    """
    A location in the metadata tree.
    """
    project_name: ScopedLocationAttrStr = None
    client_name: ScopedLocationAttrStr = None
    round_number: ScopedLocationAttrInt = None
    trial_number: ScopedLocationAttrInt = None
    split_number: ScopedLocationAttrInt = None
    metric_name: ScopedLocationAttrStr = None

    def __post_init__(self):
        BaseService.__init__(self)

    # get the field value from BaseMetadata subclass:
    def __getitem__(
        self, key: Union[Type['BaseDataclass'], Slice[Type['BaseDataclass']]]
    ) -> Union[ScopedLocationAttr, 'ScopedLocation']:
        """
        Get sublocation attr from the provided :class:`BaseMetadata` type,
        or otherwise get a slice of the same type, sliced from a :class:`BaseMetadata` type range of attributes to keep.
        If the key is a slice, the end loc is kept as inclusive.
        """
        if key is None:
            return ScopedLocation()

        if isinstance(key, slice):
            if key.stop is None or key.start is not None:
                raise ValueError("Slice stop must be specified and slice start must be None.")

            idx: int = key.stop
            if not isinstance(idx, int):
                if idx == RootDataclass:
                    return ScopedLocation()
                idx: int = list(dataclass_2_id_attr.keys()).index(key.stop) + 1

            return ScopedLocation(
                *self.as_list()[:idx]
            )

        if key == RootDataclass:
            return None

        if isinstance(key, int):
            if key < 0:
                if len(self) == 0:
                    return None
                else:
                    return self.as_list()[key]
            key = list(dataclass_2_id_attr.keys())[key]

        if key in dataclass_2_id_attr.keys():
            return getattr(self, dataclass_2_id_attr[key])

        raise ValueError(f"Invalid key type {key.__class__.__name__} for key {key}.")

    def copy(self) -> 'ScopedLocation':
        """
        Returns a copy of the :class:`ScopedLocation`.
        """
        return ScopedLocation(*self.as_list())

    def with_dc(self, dc: 'BaseDataclass') -> 'ScopedLocation':
        """
        Returns a new :class:`ScopedLocation` with the provided :class:`BaseDataclass` (dc) type's id added.
        """
        if isinstance(dc, RootDataclass):
            return ScopedLocation()
        self_copy = self.copy()
        self_copy[dc.__class__] = dc.get_id()
        return self_copy

    def with_id(self, _id: ScopedLocationAttr) -> 'ScopedLocation':
        """
        Returns a longer :class:`ScopedLocation` with the provided id added at the end.
        """
        return ScopedLocation(*(self.as_list() + [_id]))

    def at_dc(self, dc: 'BaseDataclass') -> 'ScopedLocation':
        """
        Returns a trimmed :class:`ScopedLocation` with the provided :class:`BaseDataclass` (dc) type's id as the ScopedLocation's deepest attribute.
        """
        if isinstance(dc, RootDataclass):
            return ScopedLocation()
        return self[:dc.__class__]

    @staticmethod
    def default(
        round_number: Optional[ScopedLocationAttrInt] = None,
        trial_number: Optional[ScopedLocationAttrInt] = None,
        split_number: Optional[ScopedLocationAttrInt] = None,
        metric_name: Optional[ScopedLocationAttrStr] = None,
    ) -> 'ScopedLocation':
        """
        Returns the default :class:`ScopedLocation`. That is:

        .. code-block:: python
            ScopedLocation("default_project", "default_client", *kargs, **kwargs).

        """
        if not ((round_number is not None) >= (trial_number is not None) >= (split_number is not None) >= (metric_name is not None)):
            raise ValueError(
                "round_number, trial_number, split_number, and metric_name "
                "must be specified in order, one after the other.")

        _args = []
        if round_number is not None:
            _args.append(round_number)
            if trial_number is not None:
                _args.append(trial_number)
                if split_number is not None:
                    _args.append(split_number)
                    if metric_name is not None:
                        _args.append(metric_name)

        return ScopedLocation(DEFAULT_PROJECT, DEFAULT_CLIENT, *_args)

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
                f"{list(dataclass_2_id_attr.keys())[curr_attr_to_set_idx]} is the next to be set.")
        key_attr_name: str = dataclass_2_id_attr[key]
        setattr(self, key_attr_name, value)

    def peek(self) -> ScopedLocationAttr:
        """
        Pop without removing the last element: return the last non-None element.
        """
        return self.as_list()[-1]

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
        return ScopedLocation(*self.as_list()[:-1])

    def __len__(self) -> int:
        """
        Returns the number of not none scoped location attributes.
        """
        return len(self.as_list())

    def as_list(self, stringify: bool = False) -> List[ScopedLocationAttr]:
        """
        Returns a list of the scoped location attributes.
        Item that has a value of None are not included in the list.

        :param stringify: If True, the scoped location attributes are converted to strings.
        :return: list of not none scoped location attributes
        """
        _list: List[ScopedLocationAttr] = [
            getattr(self, attr_name)
            for attr_name in dataclass_2_id_attr.values()
        ]
        _list_ret = []
        for i in _list + [None]:
            if i is not None:
                if stringify:
                    i = str(i)
                _list_ret.append(i)
            else:
                break
        return _list_ret

    def new_dataclass_from_id(self):
        """
        Creates a new :class:`BaseDataclass` of the right type with
        just the provided ID filled.
        """
        dataklass = self.last_dc_type()

        return dataklass().set_id(self.as_list()[-1])

    def last_dc_type(self) -> Type['BaseDataclass']:
        dataklass: Type[BaseDataclass] = list(dataclass_2_id_attr.keys())[len(self) - 1]
        return dataklass

    def __eq__(self, other: 'ScopedLocation') -> bool:
        return self.as_list() == other.as_list()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.as_list())[1:-1]})"

    def __str__(self) -> str:
        return self.__repr__()


@dataclass(order=True)
class BaseDataclass(Generic[SubDataclassT], ABC):
    # TODO: from json, to json?

    def __post_init__(self):
        self._validate()

    def _validate(self):
        for k, v in zip(self.get_sublocation_keys(), self.get_sublocation_values()):
            if isinstance(v, dict) and '__type__' in v:
                assert v['__type__'] == dataclass_2_subdataclass[self.__class__].__name__
                v: SubDataclassT = BaseDataclass.from_dict(v)
                assert v.get_id() == k
                self.store(v)

        subdataklass: Type[SubDataclassT] = dataclass_2_subdataclass[self.__class__]
        if subdataklass is not None and not all(
            (isinstance(s, subdataklass) for s in self.get_sublocation_values() if s is not None)
        ):
            raise ValueError(f"{self.__class__.__name__} must have all sublocation values as {subdataklass.__name__}.")

    @staticmethod
    def from_dict(_dict: Dict[str, Any]) -> 'BaseDataclass':
        if '__type__' not in _dict:
            raise ValueError("Dict must have a __type__ key.")
        _dict: dict = copy.copy(_dict)
        _type_str: str = _dict.pop('__type__')
        _klass: type = str_2_dataclass[_type_str]

        return _klass(**_dict)

    def to_dict(self) -> Dict[str, Any]:
        _dict = OrderedDict()
        _dict['__type__'] = self.__class__.__name__
        for _field in fields(self):
            _attr = getattr(self, _field.name)
            if _attr is not None:
                if isinstance(_attr, BaseDataclass):
                    _attr = _attr.to_dict()
                _dict[_field.name] = _attr
        return _dict

    @property
    def _id_attr_name(self) -> str:
        return dataclass_2_id_attr[self.__class__]

    @property
    def _sublocation_attr_name(self) -> str:
        return dataclass_2_subloc_attr[self.__class__]

    def get_id(self) -> ScopedLocationAttr:
        return getattr(self, self._id_attr_name)

    def set_id(self, _id: ScopedLocationAttr) -> 'BaseDataclass':
        setattr(self, self._id_attr_name, _id)
        self._validate()
        return self

    def get_sublocation(self) -> Union[List[SubDataclassT], OrderedDict[str, SubDataclassT]]:
        return getattr(self, dataclass_2_subloc_attr[self.__class__])

    def set_sublocation(self, sublocation: Union[List[SubDataclassT], OrderedDict[str, SubDataclassT]]) -> 'BaseDataclass':
        setattr(self, dataclass_2_subloc_attr[self.__class__], sublocation)
        return self

    def __contains__(self, key: ScopedLocation) -> bool:
        # Too shallow:
        if len(key) < list(dataclass_2_subdataclass.keys()).index(self.__class__):
            raise ValueError(f"Key not deep enough for this dataclass of type {self.__class__.__name__}")

        # Terminal recursion condition:
        if key.at_dc(self) == key:
            return key.peek() == self.get_id()

        # Recursive call deeper otherwise:
        key_subattr: ScopedLocationAttr = key[self.subdataclass_type()]
        if key_subattr in self.get_sublocation_keys():
            return key in self.get_sublocation()[key_subattr]
        return False

    def __getitem__(self, loc: ScopedLocation) -> 'SubDataclassT':
        subdataklass: Type[SubDataclassT] = dataclass_2_subdataclass[self.__class__]
        if subdataklass is None:
            return self
        located_sub_attr: ScopedLocationAttr = loc[subdataklass]
        if located_sub_attr is None:
            return self
        if located_sub_attr not in self.get_sublocation_keys():
            raise KeyError(f"Item at loc={loc} is not in {self.shallow()}.")
        subloc: SubDataclassT = self.get_sublocation()[located_sub_attr]
        sublocs_sublocs: SubDataclassT = subloc[loc]
        return sublocs_sublocs

    def __len__(self) -> int:
        return len(self.get_sublocation())

    @abstractmethod
    def store(self, dc: SubDataclassT) -> ScopedLocationAttr:
        """
        Add a subdataclass to the sublocation, at its proper ID.
        """
        raise NotImplementedError("Must use mixins.")

    @abstractmethod
    def shallow(self) -> 'BaseDataclass':
        """
        Replaces the sublocation items with None.
        """
        raise NotImplementedError("Must use mixins.")

    @abstractmethod
    def empty(self) -> 'BaseDataclass':
        """
        Do empty the sublocation.
        """
        raise NotImplementedError("Must use mixins.")

    @abstractmethod
    def get_sublocation_values(self) -> List[SubDataclassT]:
        raise NotImplementedError("Must use mixins.")

    @abstractmethod
    def get_sublocation_keys(self) -> List[ScopedLocationAttr]:
        raise NotImplementedError("Must use mixins.")

    @classmethod
    def subdataclass_type(cls) -> Type[SubDataclassT]:
        return dataclass_2_subdataclass[cls]


@dataclass(order=True)
class DataclassHasOrderedDictMixin:

    def _validate(self):
        super()._validate()
        if not isinstance(self.get_sublocation(), OrderedDict):
            raise ValueError(f"{self.__class__.__name__} must have an OrderedDict as sublocation.")
        if not all(
            (isinstance(s, ScopedLocationAttrStr) for s in self.get_sublocation_keys())
        ):
            raise ValueError(f"{self.__class__.__name__} must have all sublocation keys as strings.")

    def get_sublocation(self) -> OrderedDict[ScopedLocationAttrStr, SubDataclassT]:
        ret = super().get_sublocation()
        return ret

    def set_sublocation(self, sublocation: OrderedDict[ScopedLocationAttrStr, SubDataclassT]) -> None:
        setattr(self, self._sublocation_attr_name, sublocation)
        self._validate()

    def get_sublocation_values(self) -> List[SubDataclassT]:
        return list(self.get_sublocation().values())

    def get_sublocation_keys(self) -> List[ScopedLocationAttrStr]:
        return list(self.get_sublocation().keys())

    def store(self, dc: SubDataclassT) -> ScopedLocationAttrStr:
        self.get_sublocation()[dc.get_id()] = dc
        self._validate()
        return dc.get_id()

    def shallow(self) -> 'BaseDataclass':
        self_copy = copy.copy(self)
        new_sublocation = OrderedDict([(k, None) for k in self.get_sublocation().keys()])
        self_copy.set_sublocation(new_sublocation)
        # Deep copy only after trimming sublocations.
        return copy.deepcopy(self_copy)

    def empty(self) -> 'BaseDataclass':
        self_copy = copy.copy(self)
        self_copy.set_sublocation(OrderedDict())
        return copy.deepcopy(self_copy)


@dataclass(order=True)
class DataclassHasListMixin:

    def _validate(self):
        super()._validate()
        if not isinstance(self.get_sublocation(), List):
            raise ValueError(f"{self.__class__.__name__} must have a List as sublocation.")

    def get_sublocation(self) -> List[SubDataclassT]:
        return super().get_sublocation()

    def set_sublocation(self, sublocation: List[SubDataclassT]) -> None:
        setattr(self, self._sublocation_attr_name, sublocation)
        self._validate()

    def get_sublocation_values(self) -> List[SubDataclassT]:
        return self.get_sublocation()

    def get_sublocation_keys(self) -> List[ScopedLocationAttrInt]:
        return list(range(len(self)))

    def store(self, dc: SubDataclassT) -> ScopedLocationAttrInt:
        _id = dc.get_id()
        if _id == self.get_next_i():
            self.get_sublocation().append(dc)
        elif _id < self.get_next_i():
            self.get_sublocation()[dc.get_id()] = dc
        else:
            raise ValueError(f"{dc} has id {dc.get_id()} which is greater than the next id {self.get_next_i()}.")
        self._validate()
        return _id

    def shallow(self) -> 'BaseDataclass':
        self_copy = copy.copy(self)
        new_sublocation = [None for _ in range(len(self))]
        self_copy.set_sublocation(new_sublocation)
        # Deep copy only after trimming sublocations.
        return copy.deepcopy(self_copy)

    def empty(self) -> 'BaseDataclass':
        self_copy = copy.copy(self)
        self_copy.set_sublocation([])
        return copy.deepcopy(self_copy)

    def get_next_i(self) -> ScopedLocationAttrInt:
        return len(self.get_sublocation())


@dataclass(order=True)
class BaseTrialDataclassMixin:
    """
    Mixin class for :class:`TrialMetadata` and :class:`TrialSplitMetadata` that
    also must inherit from :class:`BaseMetadata`.
    """
    hyperparams: HyperparameterSamples = field(default_factory=lambda: HyperparameterSamples())
    status: TrialStatus = TrialStatus.PLANNED
    created_time: datetime.datetime = field(default_factory=datetime.datetime.now)
    start_time: datetime.datetime = field(default_factory=datetime.datetime.now)
    end_time: datetime.datetime = None
    # error: str = None
    # error_traceback: str = None
    # logs: List[str] = field(default_factory=list)

    def _validate(self):
        super()._validate()
        if not isinstance(self.hyperparams, HyperparameterSamples):
            self.hyperparams = HyperparameterSamples(self.hyperparams)
        self._validate_date_attr("created_time")
        self._validate_date_attr("start_time")
        self._validate_date_attr("end_time")
        if not isinstance(self.status, TrialStatus):
            self.status = TrialStatus(self.status)

    def _validate_date_attr(self, attr_name) -> None:
        attr_value = getattr(self, attr_name)
        if attr_value is not None and not isinstance(attr_value, datetime.datetime):
            attr_value = datetime.datetime.strptime(attr_value, LOGGING_DATETIME_STR_FORMAT)
            setattr(self, attr_name, attr_value)

    def start(self) -> 'BaseTrialDataclassMixin':
        self.status = TrialStatus.RUNNING
        self.start_time = datetime.datetime.now()
        self._validate()
        return self

    def end(self, status: TrialStatus) -> 'BaseTrialDataclassMixin':
        self.status = status
        self.end_time = datetime.datetime.now()
        self._validate()
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


@dataclass(order=True)
class ClientDataclass(DataclassHasListMixin, BaseDataclass['RoundDataclass']):
    client_name: ScopedLocationAttrStr = DEFAULT_CLIENT
    rounds: List['RoundDataclass'] = field(default_factory=list)


@dataclass(order=True)
class RoundDataclass(DataclassHasListMixin, BaseDataclass['TrialDataclass']):
    round_number: ScopedLocationAttrInt = 0
    trials: List['TrialDataclass'] = field(default_factory=list)
    main_metric_name: str = None


@dataclass(order=True)
class TrialDataclass(DataclassHasListMixin, BaseTrialDataclassMixin, BaseDataclass['TrialSplitDataclass']):
    """
    This class is a data structure most often used under :class:`AutoML` to store information about a trial.
    This information is itself managed by the :class:`HyperparameterRepository` class
    and the :class:`Trial` class within the AutoML.
    """
    trial_number: ScopedLocationAttrInt = 0
    validation_splits: List['TrialSplitDataclass'] = field(default_factory=list)
    retrained_split: 'TrialSplitDataclass' = None

    def store(self, dc: SubDataclassT) -> ScopedLocationAttrInt:
        if dc.get_id() == RETRAIN_TRIAL_SPLIT_ID:
            self.retrained_split = dc
            return RETRAIN_TRIAL_SPLIT_ID
        else:
            return super().store(dc)

    def __contains__(self, key: ScopedLocation) -> bool:
        if key[self.__class__] != self.get_id():
            return False
        if key[self.subdataclass_type()] == RETRAIN_TRIAL_SPLIT_ID:
            if self.retrained_split is None:
                return False
            return key in self.retrained_split
        else:
            return super().__contains__(key)

    def __getitem__(self, loc: ScopedLocation) -> SubDataclassT:
        if loc[self.__class__] != self.get_id():
            raise KeyError(f"Item at loc={loc} is not in {self.shallow()}.")
        if loc[self.subdataclass_type()] == RETRAIN_TRIAL_SPLIT_ID:
            if self.retrained_split is None:
                raise KeyError(f"Item at loc={loc} is not in {self.shallow()}. No retrains are available.")
            return self.retrained_split[loc]
        else:
            return super().__getitem__(loc)


RETRAIN_TRIAL_SPLIT_ID = -1


@dataclass(order=True)
class TrialSplitDataclass(DataclassHasOrderedDictMixin, BaseTrialDataclassMixin, BaseDataclass['MetricResultsDataclass']):
    """
    TrialSplit object used by AutoML algorithm classes.
    """
    split_number: ScopedLocationAttrInt = 0
    metric_results: OrderedDict[str, 'MetricResultsDataclass'] = field(default_factory=OrderedDict)
    # introspection_data: RecursiveDict[str, Number] = field(default_factory=RecursiveDict)

    def is_retrain_split(self) -> bool:
        return self.split_number == -1


@dataclass(order=True)
class MetricResultsDataclass(DataclassHasListMixin, BaseDataclass[float]):
    """
    MetricResult object used by AutoML algorithm classes.
    """
    metric_name: ScopedLocationAttrStr = "main"
    validation_values: List[float] = field(default_factory=list)  # one per epoch.
    train_values: List[float] = field(default_factory=list)
    higher_score_is_better: bool = True


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
    (RootDataclass.__name__, RootDataclass),
    (ProjectDataclass.__name__, ProjectDataclass),
    (ClientDataclass.__name__, ClientDataclass),
    (RoundDataclass.__name__, RoundDataclass),
    (TrialDataclass.__name__, TrialDataclass),
    (TrialSplitDataclass.__name__, TrialSplitDataclass),
    (MetricResultsDataclass.__name__, MetricResultsDataclass),
    (RecursiveDict.__name__, RecursiveDict),
])


def as_named_odict(
    obj: Union[BaseDataclass, List[BaseDataclass]]
) -> OrderedDict[ScopedLocationAttrStr, BaseDataclass]:

    if isinstance(obj, BaseDataclass):
        obj = [obj]

    return OrderedDict([
        (i.get_id(), i) for i in obj
    ])


def object_decoder(obj):
    if '__type__' in obj and obj['__type__'] in str_2_dataclass:
        # cls: Type = str_2_dataclass[obj['__type__']]
        return BaseDataclass.from_dict(obj)
        # kwargs = dict(obj)
        # del kwargs['__type__']
        # return cls(**kwargs)
    return obj


class MetadataJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, TrialStatus):
            return obj.value
        if isinstance(obj, RecursiveDict):
            return obj.to_flat_dict()
        if isinstance(obj, datetime.datetime):
            return obj.strftime(LOGGING_DATETIME_STR_FORMAT)
        if isinstance(obj, BaseDataclass):
            return obj.to_dict()
        return JSONEncoder.encode(self, obj)


def to_json(obj: str) -> str:
    return json.dumps(obj, cls=MetadataJSONEncoder)


def from_json(_json: str) -> str:
    return json.loads(_json, object_pairs_hook=OrderedDict, object_hook=object_decoder)


class HyperparamsRepository(_ObservableRepo[Tuple['HyperparamsRepository', BaseDataclass]], BaseService):
    """
    Hyperparams repository that saves hyperparams, and scores for every AutoML trial.
    Cache folder can be changed to do different round numbers.

    .. seealso::
        :class:`AutoML`,
        :class:`Trainer`,
        :class:`~neuraxle.metaopt.data.trial.Trial`,
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
    def save(self, _dataclass: SubDataclassT, scope: ScopedLocation, deep=False) -> 'HyperparamsRepository':
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

    def _save_model(self, trial: 'TrialManager', pipeline: BaseStep, context: CX):
        hyperparams = self.hyperparams.to_flat_dict()
        # TODO: ???
        trial_hash = trial.get_trial_id(hyperparams)
        pipeline.set_name(trial_hash).save(context, full_dump=True)

    def load_model(self, trial: 'TrialManager', context: CX) -> BaseStep:
        """
        Load model in the trial hash folder.
        """
        # TODO: glob?
        trial_hash: str = trial.get_trial_id(self.hyperparams.to_flat_dict)
        return CX.load(
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
        BaseService.__init__(self)
        HyperparamsRepository.__init__(self)
        self.cache_folder = os.path.join(cache_folder, self.__class__.__name__)
        self.root: RootDataclass = RootDataclass()

    @staticmethod
    def from_root(root: RootDataclass, cache_folder: str) -> 'VanillaHyperparamsRepository':

        return VanillaHyperparamsRepository(
            cache_folder=cache_folder,
        ).save(root, ScopedLocation(), deep=True)

    def load(self, scope: ScopedLocation, deep=False) -> SubDataclassT:
        try:
            ret: BaseDataclass = self.root[scope]
            if not deep:
                ret = ret.shallow()
        except KeyError as _:
            ret: BaseDataclass = scope.new_dataclass_from_id()

        return copy.deepcopy(ret)

    def save(self, _dataclass: SubDataclassT, scope: ScopedLocation, deep=False) -> 'VanillaHyperparamsRepository':
        # Sanitizing
        _dataclass: SubDataclassT = copy.deepcopy(_dataclass)

        # Sanity checks: good type
        if not isinstance(_dataclass, BaseDataclass):
            raise ValueError(f"Was expecting a dataclass. Got `{_dataclass.__class__.__name__}`.")
        # Sanity checks: sufficient scope depth
        scope = scope[:_dataclass.__class__]  # Sanitizing scope to dtype loc.
        _id_from_scope: ScopedLocationAttr = scope[_dataclass.__class__]
        if _id_from_scope is not None:
            if _id_from_scope != _dataclass.get_id():
                raise ValueError(
                    f"The scope `{scope}` with {_dataclass.__class__.__name__} id `{_id_from_scope}` does not match the provided dataclass id `{_dataclass.get_id()}` for `{_dataclass}`."
                )
            scope.pop()
            # else check if the scope is at least of the good class length:
        elif len(scope) != list(dataclass_2_subloc_attr.keys()).index(_dataclass.__class__):
            raise ValueError(
                f"The scope `{scope}` is not of the good length for dataclass type `{_dataclass.__class__.__name__}`."
            )

        # Passthrough dc's sublocation when saving shallow:
        if not deep:
            if isinstance(_dataclass, RootDataclass):
                _dataclass.set_sublocation(self.root.get_sublocation())
            elif scope.with_dc(_dataclass) in self.root:
                prev_dc: SubDataclassT = self.root[scope.with_dc(_dataclass)]
                _dataclass.set_sublocation(prev_dc.get_sublocation())
            else:
                _dataclass = _dataclass.empty()

        # Finally storing.
        if isinstance(_dataclass, RootDataclass):
            self.root = _dataclass
        else:
            self.root[scope].store(_dataclass)
        return self

    def get_scoped_logger_path(self, scope: ScopedLocation) -> str:
        scoped_path: str = self.get_scoped_path(scope)
        return os.path.join(scoped_path, 'log.txt')

    def get_scoped_path(self, scope: ScopedLocation) -> str:
        _scope_attrs = scope.as_list(stringify=True)
        return os.path.join(self.cache_folder, *_scope_attrs)


class InMemoryHyperparamsRepository(RaiseDeprecatedClass, HyperparamsRepository):
    """
    In memory hyperparams repository that can print information about trials.
    Useful for debugging.
    """

    def __init__(self, *kargs, **kwargs):
        RaiseDeprecatedClass.__init__(
            self,
            replacement_class=VanillaHyperparamsRepository,
            since_version="0.7.0",
        )


class BaseHyperparameterOptimizer(ABC):

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


class AutoMLContext(CX):

    @property
    def logger_at_scoped_loc(self) -> NeuraxleLogger:
        return logging.getLogger(self.get_identifier(include_step_names=False))

    def add_scoped_logger_file_handler(self) -> NeuraxleLogger:
        """
        Add a file handler to the logger at the current scoped location to capture logs
        at this scope and below this scope.
        """
        logging_file = self.repo.get_scoped_logger_path(self.loc)
        os.makedirs(os.path.dirname(logging_file), exist_ok=True)
        self.logger_at_scoped_loc.with_file_handler(logging_file)

    def free_scoped_logger_file_handler(self):
        """
        Remove file handlers from logger to free file lock (especially on Windows).
        """
        self.logger_at_scoped_loc.without_file_handler()

    def read_scoped_logger_file(self) -> str:
        with open(self.repo.get_scoped_logger_path(self.loc), "r") as f:
            l: str = "".join(f.readlines())
        return l

    def copy(self):
        copy_kwargs = self._get_copy_kwargs()
        return AutoMLContext(**copy_kwargs)

    def _get_copy_kwargs(self):
        return super()._get_copy_kwargs()

    @staticmethod
    def from_context(
        context: CX = None,
        repo: HyperparamsRepository = None,
        loc: ScopedLocation = None
    ) -> 'AutoMLContext':
        """
        Create a new AutoMLContext from an ExecutionContext.

        :param context: ExecutionContext
        """
        new_context: AutoMLContext = AutoMLContext.copy(
            context or AutoMLContext()
        )
        if not new_context.has_service(HyperparamsRepository):
            new_context.register_service(
                HyperparamsRepository,
                repo or VanillaHyperparamsRepository(new_context.get_path())
            )
        if not new_context.has_service(ScopedLocation):
            new_context.register_service(
                ScopedLocation,
                loc or ScopedLocation()
            )
        if not new_context.has_service(ContextLock):
            new_context.register_service(
                ContextLock,
                ContextLock()
            )
        return new_context

    @property
    def lock(self):
        return self.synchroneous()

    @property
    def loc(self) -> ScopedLocation:
        return self.get_service(ScopedLocation)

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
        return self.with_loc(self.loc.with_dc(subdataclass))

    def pop_attr(self) -> 'AutoMLContext':
        """
        Pop an attribute from the ScopedLocation.

        :return: an AutoMLContext copy with the new popped loc attribute.
        """
        return self.with_loc(self.loc.popped())

    def with_loc(self, loc: ScopedLocation) -> 'AutoMLContext':
        """
        Replace the ScopedLocation by the one provided.

        :param loc: ScopedLocation
        :return: an AutoMLContext copy with the new loc attribute.
        """
        new_self: AutoMLContext = self.copy()
        new_self.register_service(ScopedLocation, loc)
        return new_self

    def load_dc(self, deep=True) -> BaseDataclass:
        """
        Load the current dc from the repo.
        """
        return self.repo.load(self.loc, deep)

    def load_agg(self, deep=True) -> 'agg.BaseAggregate':
        """
        Load the current agg from the repo.
        """
        return agg.BaseAggregate.from_context(self, is_deep=deep)
