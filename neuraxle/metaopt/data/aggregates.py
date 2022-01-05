"""
Neuraxle's AutoML Scope Manager Classes
====================================


Scope manager objects used by AutoML algorithm classes.

Although the name "manager" for a class is disliked by the Neuraxle community,
it is used as these are Pythonic context managers, as described in
`Python's contextlib documentation <https://docs.python.org/3/library/contextlib.html>`__.

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
import gc
import hashlib
import json
import logging
import os
import traceback
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from json.encoder import JSONEncoder
from logging import FileHandler, Logger
from typing import (Any, Callable, Dict, Generic, Iterable, List, Optional,
                    Tuple, Type, TypeVar, Union)

import numpy as np
from neuraxle.base import (BaseService, BaseStep, ExecutionContext, Flow,
                           TrialStatus, _CouldHaveContext, _HasChildrenMixin,
                           synchroneous_flow_method)
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace, RecursiveDict)
from neuraxle.metaopt.data.vanilla import (DEFAULT_CLIENT, DEFAULT_PROJECT,
                                           AutoMLContext, BaseDataclass,
                                           BaseHyperparameterOptimizer,
                                           ClientDataclass,
                                           HyperparamsRepository,
                                           InMemoryHyperparamsRepository,
                                           MetricResultsDataclass,
                                           ProjectDataclass, RecursiveDict,
                                           RootDataclass, RoundDataclass,
                                           ScopedLocation, ScopedLocationAttr,
                                           SubDataclassT, TrialDataclass,
                                           TrialSplitDataclass,
                                           VanillaHyperparamsRepository,
                                           dataclass_2_id_attr)
from neuraxle.steps.flow import ReversiblePreprocessingWrapper

SubAggregateT = TypeVar('SubAggregateT', bound=Optional['BaseAggregate'])


def _with_method_as_context_manager(func: Callable[['BaseAggregate'], SubAggregateT]):
    """

    .. note::
        This is a method to be used as a context manager.
        This will sync items with the repos.
        Example:

    .. code-block:: python
        with obj.func() as managed_context:
            # obj.__enter__() is called
            managed_context.do_something()
        # obj.__exit__() is called

    .. seealso::
        :class:`neuraxle.metaopt.data.aggregates.BaseAggregate`
        :func:`neuraxle.metaopt.data.aggregates.BaseAggregate.__enter__`
        :func:`neuraxle.metaopt.data.aggregates.BaseAggregate.__exit__`
        :class:`neuraxle.metaopt.auto_ml.Trainer`
        :func:`neuraxle.metaopt.auto_ml.Trainer.train`

    """
    # Also add the docstring of `_with_method_as_context_manager' to the docstring of the `func`:
    func.__doc__ = (func.__doc__ or "") + _with_method_as_context_manager.__doc__
    return func


class BaseAggregate(_CouldHaveContext, BaseService, Generic[SubAggregateT, SubDataclassT]):
    """
    Base class for aggregated objects using the repo and the dataclasses to manipulate them.
    """

    def __init__(self, _dataclass: SubDataclassT, context: AutoMLContext, is_deep=False):
        BaseService.__init__(self, name=f"{self.__class__.__name__}_{_dataclass.get_id()}")
        _CouldHaveContext.__init__(self)
        self._dataclass: SubDataclassT = _dataclass
        self.context: AutoMLContext = context.push_attr(_dataclass)
        self.loc: ScopedLocation = self.context.loc.copy()
        self.is_deep = is_deep

        self.service_assertions = [Flow, HyperparamsRepository]
        self._invariant()

    def _invariant(self):
        _type: Type[SubDataclassT] = self.dataclass
        self._assert(isinstance(self._dataclass, _type),
                     f"self._dataclass should be of type {_type.__name__} but is of type "
                     f"{self._dataclass.__class__.__name__}", self.context)
        self._assert(isinstance(self.context, AutoMLContext),
                     f"self.context should be of type AutoMLContext but is of type "
                     f"{self.context.__class__.__name__}", self.context)
        self._assert(isinstance(self.loc, ScopedLocation),
                     f"self.loc should be of type ScopedLocation but is of type "
                     f"{self.loc.__class__.__name__}", self.context)
        self._assert(self.loc.as_list() == self.context.loc.as_list(),
                     f"{self.loc} should be equal to {self.context.loc}", self.context)
        self._assert(self.loc[-1] == self._dataclass.get_id(),
                     f"{self.loc}'s last attr should be equal to self._dataclass.get_id() "
                     f"and the id is {self._dataclass.get_id()}", self.context)
        self._assert_at_lifecycle(self.context)

    def without_context(self) -> 'BaseAggregate':
        """
        Return a copy of this aggregate without the context.
        Useful for initializing a temporary aggregate,
        such as a filtered or reduced aggregate without all its subaggregates
        to disallow saving the reduced aggregate.
        """
        self_copy = copy.copy(self)
        self_copy.context = None
        return self_copy

    @property
    def flow(self) -> Flow:
        return self.context.flow

    @property
    def repo(self) -> HyperparamsRepository:
        return self.context.repo

    @property
    def subaggregate(self) -> Type[SubAggregateT]:
        return aggregate_2_subaggregate[self.__class__]

    @property
    def dataclass(self) -> Type[BaseDataclass]:
        return aggregate_2_dataclass[self.__class__]

    @property
    def subdataclass(self) -> Type[SubDataclassT]:
        return aggregate_2_dataclass[aggregate_2_subaggregate[self.__class__]]

    def refresh(self, deep: bool = True):
        with self.context.lock:
            self._dataclass = self.repo.load(self.loc, deep=deep)
        self.is_deep = deep
        self._invariant()

    def save(self, deep: bool = True):
        if deep and deep != self.is_deep:
            raise ValueError(
                f"Cannot save {str(self)} with deep=True when self "
                f"is not already deep. You might want to use self.refresh(deep=True) at "
                f"some point to refresh self before saving deeply then.")

        self._invariant()
        with self.context.lock:
            self.repo.save(self._dataclass, self.loc, deep=deep)

    def save_subaggregate(self, subagg: SubAggregateT, deep=False):
        self._dataclass.store(subagg._dataclass)
        with self.context.lock:
            self.save(deep=False)
            subagg.save(deep=deep)

    def __enter__(self) -> SubAggregateT:
        # self.context.free_scoped_logger_handler_file()
        self._invariant()
        self._managed_resource._invariant()

        self._managed_resource.context.add_scoped_logger_file_handler()

        return self._managed_resource

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._managed_resource.context.free_scoped_logger_file_handler()
        # self.context.add_scoped_logger_file_handler()

        handled_err: bool = self._release_managed_subresource(self._managed_resource, exc_val)
        return handled_err

    @_with_method_as_context_manager
    def managed_subresource(self, *args, **kwds) -> SubAggregateT:
        self._managed_subresource(*args, **kwds)
        return self

    def _managed_subresource(self, *args, **kwds) -> SubAggregateT:
        self._invariant()
        self._managed_resource: SubAggregateT = self._acquire_managed_subresource(
            *args, **kwds)
        return self

    @abstractmethod
    def _acquire_managed_subresource(self, *args, **kwds) -> SubAggregateT:
        """
        Acquire a new subaggregate that is managed such that it is deep saved at the
        beginning.
        """
        # Create subaggregate:
        with self.context.lock:
            self.refresh(self.is_deep)
            subdataclass: SubDataclassT = self.repo.load(*args, **kwds)
            subagg: SubAggregateT = self.subaggregate(subdataclass, self.context)
            self.save_subaggregate(subagg, deep=False)
            return subagg

    def _release_managed_subresource(self, resource: SubAggregateT, e: Exception = None) -> Optional[Exception]:
        """
        Release a subaggregate that was acquired with managed_subresource. The subaggregate
        normally has already saved itself. We may update things again here if needed.

        Exceptions may be handled here. If handled, return True, if not, then return False.
        """
        with self.context.lock:
            self.refresh(self.is_deep)
            self.save(False)
        return False

    def __len__(self) -> int:
        return len(self._dataclass.get_sublocation())

    def __iter__(self) -> Iterable[SubAggregateT]:
        for subdataclass in self._dataclass.get_sublocation_values():
            if subdataclass is not None:
                yield self.subaggregate(subdataclass, self.context, is_deep=self.is_deep)

    def __getitem__(self, item: int) -> 'Trial':
        """
        Get trial at the given index.

        :param item: trial index
        :return:
        """
        return self.subaggregate(self._dataclass.get_sublocation()[item], self.context, self.is_deep)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        if self.is_deep:
            prefix = "<deep>"
            subobjs = str([str(t) for t in self.__iter__()])
        else:
            prefix = "<shallow>"
            subobjs = f"{self._dataclass.get_sublocation()}>"
        return (
            f"{self.__class__.__name__}{prefix}("
            f"id={self._dataclass.get_id()}, {subobjs}"
            f")"
        )

    def __eq__(self, other: object) -> bool:
        return self._dataclass == other._dataclass


class Root(BaseAggregate['Project', RootDataclass]):

    def save(self, deep: bool = False):
        if deep:
            for p in self.projects:
                p.save(deep=p.is_deep)

    @property
    def projects(self) -> List['Project']:
        return list(self)

    @staticmethod
    def from_repo(context: ExecutionContext, repo: HyperparamsRepository) -> 'Root':
        _dataclass: RootDataclass = repo.load(ScopedLocation())
        automl_context: AutoMLContext = AutoMLContext.from_context(context, repo)
        return Root(_dataclass, automl_context)

    @staticmethod
    def vanilla(context: ExecutionContext = None) -> 'Root':
        if context is None:
            context = ExecutionContext()
        vanilla_repo: HyperparamsRepository = VanillaHyperparamsRepository(
            os.path.join(context.get_path(), "hyperparams"))
        return Root.from_repo(context, vanilla_repo)

    @_with_method_as_context_manager
    def get_project(self, name: str) -> 'Root':
        # TODO: new project should be created if it does not exist?
        self._managed_subresource(project_name=name)
        return self

    @_with_method_as_context_manager
    def default_project(self) -> 'Root':
        self._managed_subresource(project_name=DEFAULT_PROJECT)
        return self

    def _acquire_managed_subresource(self, project_name: str) -> 'Project':
        project_loc: ScopedLocation = self.loc.with_id(project_name)
        return super()._acquire_managed_subresource(project_loc)


class Project(BaseAggregate['Client', ProjectDataclass]):

    @_with_method_as_context_manager
    def get_client(self, name: str) -> 'Project':
        # TODO: new client should be created if it does not exist?
        self._managed_subresource(client_name=name)
        return self

    @_with_method_as_context_manager
    def default_client(self) -> 'Project':
        self._managed_subresource(client_name=DEFAULT_CLIENT)
        return self

    def _acquire_managed_subresource(self, client_name: str) -> 'Client':
        client_loc: ScopedLocation = self.loc.with_id(client_name)
        return super()._acquire_managed_subresource(client_loc)


class Client(BaseAggregate['Round', ClientDataclass]):

    @_with_method_as_context_manager
    def new_round(self) -> 'Client':
        self._managed_subresource(new_round=True)
        return self

    @_with_method_as_context_manager
    def resume_last_round(self) -> 'Client':
        self._managed_subresource(new_round=False)
        return self

    def _acquire_managed_subresource(self, new_round: bool = True) -> 'Round':
        with self.context.lock:
            self.refresh(self.is_deep)

            # Get new round loc:
            round_id: int = self._dataclass.get_next_i()
            if not new_round:
                # Try get last round:
                round_id = max(0, round_id - 1)
            round_loc: ScopedLocation = self.loc.with_id(round_id)

            # Get round to return:
            _round_dataclass: RoundDataclass = self.repo.load(round_loc, deep=True)
            subagg: Round = Round(_round_dataclass, self.context, is_deep=True)

            if new_round or round_loc == 0:
                self.save_subaggregate(subagg, deep=True)
            return subagg


class Round(BaseAggregate['Trial', RoundDataclass]):

    def with_optimizer(
        self, hp_optimizer: BaseHyperparameterOptimizer, hps: HyperparameterSpace
    ) -> 'Round':
        self.hp_optimizer: BaseHyperparameterOptimizer = hp_optimizer
        self.hps: HyperparameterSpace = hps
        return self

    @_with_method_as_context_manager
    def new_rvs_trial(self) -> 'Round':
        self._managed_subresource(new_trial=True)
        return self

    @_with_method_as_context_manager
    def last_trial(self) -> 'Round':
        self._managed_subresource(new_trial=False)
        return self

    @property
    def _trials(self) -> List['Trial']:
        return [Trial(t, self.context) for t in self._dataclass.get_sublocation()]

    def _acquire_managed_subresource(self, new_trial=True) -> 'Trial':
        with self.context.lock:
            self.refresh(self.is_deep)
            # self.context.add_scoped_logger_file_handler()

            # Get new trial loc:
            trial_id: int = self._dataclass.get_next_i()
            if not new_trial:
                # Try get last trial
                trial_id = max(0, trial_id - 1)
            trial_loc = self.loc.with_id(trial_id)

            # Get trial to return:
            _trial_dataclass: TrialDataclass = self.repo.load(trial_loc, deep=True)
            new_hps: HyperparameterSamples = self.hp_optimizer.find_next_best_hyperparams(self)
            _trial_dataclass.hyperparams = new_hps
            self.flow.log_planned(new_hps)
            subagg: Trial = Trial(_trial_dataclass, self.context, is_deep=True)

            if new_trial or trial_loc == 0:
                self.save_subaggregate(subagg, deep=True)
            return subagg

    def _release_managed_subresource(self, resource: 'Trial', e: Exception = None) -> Optional[Exception]:
        """
        Release a subaggregate that was acquired with managed_subresource. The subaggregate
        normally has already saved itself. We may update things again here if needed.

        Exceptions may be handled here. If handled, return True, if not, then return False.
        """
        with self.context.lock:
            self.refresh(True)

            if e is None:
                resource.set_success()
            else:
                resource.set_failed(e)

            self.save_subaggregate(resource, deep=resource.is_deep)

            self.flow.log('Finished round hp search!')
            self.flow.log_best_hps(
                self.hp_optimizer.get_main_metric_name(),
                self.get_best_hyperparams()
            )

            self.save(False)
        return False

    def get_best_hyperparams(self, main_metric_name: str) -> HyperparameterSamples:
        """
        Get best hyperparams from all trials.

        :return:
        """
        if not self.is_deep:
            self.refresh(True)

        if len(self) == 0:
            return HyperparameterSamples(main_metric_name)
        return self.get_best_trial(main_metric_name).get_hyperparams()

    def get_best_trial(self, main_metric_name: str) -> Optional['Trial']:
        """
        :return: trial with best score from all trials
        """
        best_score, best_trial = None, None

        if len(self) == 0:
            raise Exception('Could not get best trial because there were no successful trial.')

        for trial in self._trials:
            trial_score = trial.get_avg_validation_score(main_metric_name)

            _has_better_score = best_score is None or (
                trial_score is not None and (
                    self.is_higher_score_better(main_metric_name) == (trial_score > best_score)
                )
            )

            if _has_better_score:
                best_score = trial_score
                best_trial = trial

        return best_trial

    def is_higher_score_better(self, metric_name: str = None) -> bool:
        """
        Return true if higher score is better. If metric_name is None, the optimizer's
        metric is taken.

        :return
        """
        if len(self) == 0:
            return False

        if metric_name is None:
            metric_name = self.hp_optimizer.get_main_metric_name()

        is_higher_score_better_attr = "is_higher_score_better_attr_" + metric_name
        if not hasattr(self, is_higher_score_better_attr):
            setattr(
                self,
                is_higher_score_better_attr,
                self._trials[-1].is_higher_score_better(metric_name)
            )
        return getattr(self, is_higher_score_better_attr)

    def append(self, trial: 'Trial'):
        """
        Add a new trial. Will also save the trial shallowly.

        :param trial: new trial
        :return:
        """
        self.save_subaggregate(trial, deep=False)

    def filter(self, status: TrialStatus) -> 'Round':
        """
        Get all the trials with the given trial status.

        :param status: trial status
        :return:
        """
        _round_copy: Round = self.copy().without_context()
        _trials: List[TrialDataclass] = [
            sdc
            for sdc in self._dataclass.trials
            if sdc.get_status() == status
        ]
        _round_copy._dataclass.trials = _trials

        return _round_copy

    def copy(self) -> 'Round':
        return Round(copy.deepcopy(self._dataclass), self.context.copy())

    def get_number_of_split(self):
        if len(self) > 0:
            return len(self[0]._validation_splits)
        return 0

    def get_metric_names(self) -> List[str]:
        if len(self) > 0:
            return list(self[-1]._dataclass.validation_splits[-1].metric_results.keys())
        return []


class Trial(BaseAggregate['TrialSplit', TrialDataclass]):
    """
    This class is a sub-contextualization of the :class:`HyperparameterRepository`
    class for holding a trial and manipulating it within its context.
    A Trial contains the results for each validation split.
    Each trial split contains both the training set results, and the validation set results.

    .. seealso::
        :class:`Trial`,
        :class:`TrialSplit`,
        :class:`HyperparamsRepository`,
        :class:`AutoML`,
        :class:`ExecutionContext`
    """

    @_with_method_as_context_manager
    def new_validation_split(self, continue_loop_on_error: bool) -> 'Trial':
        self._managed_subresource(continue_loop_on_error=continue_loop_on_error)
        return self

    @property
    def _validation_splits(self) -> List['TrialSplit']:
        if not self.is_deep:
            self.refresh(True)
        return [TrialSplit(s, self.context, is_deep=self.is_deep)
                for s in self._dataclass.validation_splits]

    def _acquire_managed_subresource(self, continue_loop_on_error: bool) -> 'TrialSplit':

        self.error_types_to_raise = (
            SystemError, SystemExit, EOFError, KeyboardInterrupt
        ) if continue_loop_on_error else (Exception,)

        with self.context.lock:
            self.refresh(self.is_deep)
            # self.context.add_scoped_logger_file_handler()

            # Get new split loc:
            split_id: int = self._dataclass.get_next_i()
            split_id = max(0, split_id)
            if split_id == 0:
                self.flow.log_start()
            split_loc = self.loc.with_id(split_id)

            # Get split to return:
            _split_dataclass: TrialSplitDataclass = self.repo.load(split_loc)
            _split_dataclass.hyperparams = self.get_hyperparams()
            subagg: TrialSplit = TrialSplit(_split_dataclass, self.context, is_deep=True)
            # TODO: logger loc and file for context.push_attr?

            self.save_subaggregate(subagg, deep=True)
            return subagg

    def _release_managed_subresource(self, resource: 'TrialSplit', e: Exception = None) -> Optional[Exception]:
        gc.collect()
        handled_error = True

        with self.context.lock:
            self.refresh(self.is_deep)
            # self.context.free_scoped_logger_handler_file()

            if e is None:
                resource.set_success()
            else:
                resource.set_failed(e)

                if any((isinstance(e, c) for c in self.error_types_to_raise)):
                    handled_error = False
                    self.set_failed(e)

            self.save_subaggregate(resource, deep=True)

        return handled_error

    def is_success(self):
        """
        Set trial status to success.
        """
        return self._dataclass.status == TrialStatus.SUCCESS

    def get_avg_validation_score(self, metric_name: str) -> float:
        """
        Returns the average score for all validation splits's
        best validation score for the specified scoring metric.

        :return: validation score
        """
        if self.is_success():
            scores = [
                val_split[metric_name].get_best_validation_score()
                for val_split in self._validation_splits
                if val_split.is_success() and metric_name in val_split.get_metric_names()
            ]
            return sum(scores) / len(scores)

    def get_avg_n_epoch_to_best_validation_score(self, metric_name: str) -> float:
        # TODO: use in flow.log_results:
        n_epochs = [
            val_split.get_n_epochs_to_best_validation_score(metric_name)
            for val_split in self._validation_splits if val_split.is_success()
        ]

        n_epochs = sum(n_epochs) / len(n_epochs)

        return n_epochs

    def get_hyperparams(self) -> RecursiveDict:
        """
        Return hyperparams dict.

        :return: hyperparams dict
        """
        return self._dataclass.hyperparams

    def set_success(self) -> 'Trial':
        """
        Set trial status to success.

        :return: self
        """
        self._dataclass.end(TrialStatus.SUCCESS)
        self.flow.log_success()
        return self

    def update_final_trial_status(self):
        """
        Set trial status to success.
        """
        success = True
        for validation_split in self._validation_splits:
            if not validation_split.is_success():
                success = False

        self.status = TrialStatus.SUCCESS if success else TrialStatus.FAILED

        self.save_trial()  # TODO?

    def set_failed(self, error: Exception) -> 'Trial':
        """
        Set failed trial with exception.

        :param error: catched exception
        :return: self
        """
        self._dataclass.end(TrialStatus.FAILED)
        self.flow.log_failure(exception=error)
        return self

    def get_trial_id(self, hp_dict: Dict):
        """
        Hash hyperparams with blake2s to create a trial hash.

        :param hp_dict: hyperparams dict
        :return:
        """
        current_hyperparameters_hash = hashlib.blake2s(str.encode(str(hp_dict))).hexdigest()
        return f"{self._dataclass.trial_number}_{current_hyperparameters_hash}"

    def is_higher_score_better(self, metric_name: str) -> bool:
        """
        Return if the metric is higher score is better.

        :param metric_name: metric name
        :return: bool
        """
        return self._dataclass.validation_splits[-1].metric_results[metric_name].higher_score_is_better


class TrialSplit(BaseAggregate['MetricResults', TrialSplitDataclass]):
    """
    One split of a trial.

    .. seealso::
        :class:`AutoML`,
        :class:`HyperparamsRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`DataContainer`
    """

    def __init__(self, _dataclass: SubDataclassT, context: AutoMLContext, is_deep=False):
        super().__init__(_dataclass, context, is_deep=is_deep)
        self.epoch: int = 0
        self.n_epochs: int = None

    def _invariant(self):
        self._assert(self.is_deep,
                     f"self.is_deep should always be set to True for "
                     f"{self.__class__.__name__}", self.context)
        super()._invariant()

    def with_n_epochs(self, n_epochs: int) -> 'TrialSplit':
        self.n_epochs: int = n_epochs
        return self

    def next_epoch(self) -> int:
        """
        Increment epoch. Returns the new epoch id.
        Epochs are 1-indexed like lengths: first epoch is 1, second is 2, etc.
        """
        if self.n_epochs is None:
            raise ValueError("self.n_epochs is not set. Please call self.with_n_epochs(n_epochs) first.")
        if self.epoch == 0:
            self.start()
        self.epoch: int = self.epoch + 1
        self.flow.log_epoch(self.epoch, self.n_epochs)
        return self.epoch

    def start(self) -> 'TrialSplit':
        """
        Start the trial split.
        """
        self.flow.log_start()
        self._dataclass.start()
        return self

    @property
    def metric_results(self) -> Dict[str, 'MetricResults']:
        return {
            metric_name: MetricResults(mr, self.context)  # .at_epoch(self.epoch, self.n_epochs)
            for metric_name, mr in self._dataclass.metric_results.items()
        }

    def metric_result(self, metric_name: str) -> 'MetricResults':
        """
        Get a metric result not managed with a "with" statement.
        """
        mr: MetricResultsDataclass = self._dataclass.get_sublocation()[metric_name]
        return MetricResults(mr, self.context)

    @_with_method_as_context_manager
    def managed_metric(self, metric_name: str, higher_score_is_better: bool) -> 'TrialSplit':
        """
        To be used as a with statement to get the managed metric.
        """
        self._managed_subresource(metric_name=metric_name, higher_score_is_better=higher_score_is_better)
        return self

    def _acquire_managed_subresource(self, metric_name: str, higher_score_is_better: bool) -> 'MetricResults':
        with self.context.lock:
            self.refresh(True)

            subdataclass: MetricResultsDataclass = self._create_or_get_metric_results(
                metric_name, higher_score_is_better)

            subagg: MetricResults = self.subaggregate(subdataclass, self.context, is_deep=True)
            return subagg

    def _create_or_get_metric_results(self, name, higher_score_is_better):
        if name not in self._dataclass.metric_results:
            self._dataclass.metric_results[name] = MetricResultsDataclass(
                metric_name=name,
                validation_values=[],
                train_values=[],
                higher_score_is_better=higher_score_is_better,
            )
        return self._dataclass.metric_results[name]

    def _release_managed_subresource(self, resource: 'MetricResults', e: Exception = None) -> Optional[Exception]:
        # TODO: func return type??

        handled_error = True
        with self.context.lock:
            if e is not None:
                handled_error = False
                self.context.flow.log_error(e)
            self.save_subaggregate(resource, deep=True)
        return handled_error

    def train_context(self) -> 'AutoMLContext':
        return self.context.train()

    def validation_context(self) -> 'AutoMLContext':
        return self.context.validation()

    def get_metric_names(self) -> List[str]:
        return list(self._dataclass.metric_results.keys())

    def set_success(self) -> 'TrialSplit':
        """
        Set trial status to success.

        :return: self
        """
        self._dataclass.end(status=TrialStatus.SUCCESS)
        self.flow.log_success()
        return self

    def is_success(self):
        """
        Set trial status to success.
        """
        return self._dataclass.status == TrialStatus.SUCCESS

    def set_failed(self, error: Exception) -> 'TrialSplit':
        """
        Set failed trial with exception.

        :param error: catched exception
        :return: self
        """
        if isinstance(error, SystemExit) or isinstance(error, KeyboardInterrupt):
            self._dataclass.end(TrialStatus.ABORTED)
            self.flow.log_aborted(error)
        else:
            self._dataclass.end(TrialStatus.FAILED)
            self.flow.log_failure(error)
        return self


class MetricResults(BaseAggregate[None, MetricResultsDataclass]):

    def _invariant(self):
        self._assert(self.is_deep,
                     f"self.is_deep should always be set to True for "
                     f"{self.__class__.__name__}", self.context)
        super()._invariant()

    def _acquire_managed_subresource(self, *args, **kwds) -> SubAggregateT:
        # TODO: epoch class and epoch location?
        raise NotImplementedError("MetricResults has no subresource to manage as a terminal resource.")

    @property
    def metric_name(self) -> str:
        return self._dataclass.metric_name

    def add_train_result(self, score: float):
        """
        Add a train metric result.

        :param name: name of the metric
        :param score: the value to be logged
        :param higher_score_is_better: wheter or not a higher score is better for this metric
        """
        self._dataclass.train_values.append(score)
        self.flow.log_train_metric(self.metric_name, score)
        self.save(True)

    def add_valid_result(self, score: float):
        """
        Add a validation metric result.

        :param name: name of the metric
        :param score: the value to be logged
        :param higher_score_is_better: wheter or not a higher score is better for this metric
        """
        self._dataclass.validation_values.append(score)
        self.flow.log_valid_metric(self.metric_name, score)
        self.save(True)

    def get_train_scores(self) -> List[float]:
        return self._dataclass.train_values

    def get_valid_scores(self) -> List[float]:
        """
        Return the validation scores for the given scoring metric.
        """
        return self._dataclass.validation_values

    def get_final_validation_score(self) -> float:
        """
        Return the latest validation score for the given scoring metric.
        """
        return self.get_valid_scores()[-1]

    def get_best_validation_score(self) -> float:
        """
        Return the best validation score for the given scoring metric.
        """
        if self.is_higher_score_better():
            f = np.max
        else:
            f = np.min

        return f(self.get_valid_scores())

    def get_n_epochs_to_best_validation_score(self) -> int:
        """
        Return the number of epochs
        """
        if self.is_higher_score_better():
            f = np.argmax
        else:
            f = np.argmin

        return f(self.get_valid_scores())

    def is_higher_score_better(self) -> bool:
        """
        Return True if higher scores are better for the main metric.

        :return:
        """
        return self._dataclass.higher_score_is_better

    def is_new_best_score(self) -> bool:
        """
        Return True if the latest validation score is the new best score.

        :return:
        """
        if self.get_best_validation_score() in self.get_valid_scores()[:-1]:
            return False
        return True

    def __iter__(self) -> Iterable[SubAggregateT]:
        """
        Loop over validation values.
        """
        return self._dataclass.validation_values


aggregate_2_subaggregate: OrderedDict[BaseAggregate, BaseAggregate] = OrderedDict([
    (Root, Project),
    (Project, Client),
    (Client, Round),
    (Round, Trial),
    (Trial, TrialSplit),
    (TrialSplit, MetricResults),
    (MetricResults, MetricResults),
])

aggregate_2_dataclass: OrderedDict[BaseAggregate, BaseDataclass] = OrderedDict([
    (Root, RootDataclass),
    (Client, ClientDataclass),
    (Round, RoundDataclass),
    (Project, ProjectDataclass),
    (Trial, TrialDataclass),
    (TrialSplit, TrialSplitDataclass),
    (MetricResults, MetricResultsDataclass),
])
