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
import gc
import hashlib
import os
from abc import abstractmethod
from collections import OrderedDict
from types import TracebackType
from typing import (Any, Callable, ContextManager, Dict, Generic, Iterable,
                    List, Optional, Tuple, Type, TypeVar, Union)

import numpy as np
from neuraxle.base import BaseService
from neuraxle.base import ExecutionContext as CX
from neuraxle.base import Flow, TrialStatus, _CouldHaveContext
from neuraxle.hyperparams.space import (FlatDict, HyperparameterSamples,
                                        HyperparameterSpace, RecursiveDict)
from neuraxle.metaopt.data.vanilla import (DEFAULT_CLIENT, DEFAULT_PROJECT,
                                           RETRAIN_TRIAL_SPLIT_ID,
                                           AutoMLContext, BaseDataclass,
                                           BaseHyperparameterOptimizer,
                                           ClientDataclass,
                                           HyperparamsRepository,
                                           MetricResultsDataclass,
                                           ProjectDataclass, RootDataclass,
                                           RoundDataclass, ScopedLocation,
                                           ScopedLocationAttr,
                                           ScopedLocationAttrInt,
                                           SubDataclassT, TrialDataclass,
                                           TrialSplitDataclass,
                                           VanillaHyperparamsRepository,
                                           dataclass_2_id_attr)

SubAggregateT = TypeVar('SubAggregateT', bound=Optional['BaseAggregate'])
ParentAggregateT = TypeVar('ParentAggregateT', bound=Optional['BaseAggregate'])


def _with_method_as_context_manager(
    func: Callable[['BaseAggregate'], 'BaseAggregate']
) -> Callable[['BaseAggregate'], ContextManager['SubAggregateT']]:
    """
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


class BaseAggregate(_CouldHaveContext, BaseService, ContextManager[SubAggregateT], Generic[ParentAggregateT, SubAggregateT, SubDataclassT]):
    """
    Base class for aggregated objects using the repo and the dataclasses to manipulate them.
    """

    def __init__(self, _dataclass: SubDataclassT, context: AutoMLContext, is_deep=False, parent: ParentAggregateT = None):
        BaseService.__init__(self, name=f"{self.__class__.__name__}_{_dataclass.get_id()}")
        _CouldHaveContext.__init__(self)
        self._dataclass: SubDataclassT = _dataclass
        # TODO: pre-push context to allow for dc auto-loading and easier parent auto-loading.
        self.context: AutoMLContext = context.push_attr(_dataclass)
        self.loc: ScopedLocation = self.context.loc.copy()
        self.is_deep = is_deep
        self._parent: ParentAggregateT = parent

        self.service_assertions = [Flow, HyperparamsRepository]
        self._invariant()

    @staticmethod
    def from_context(context: AutoMLContext, is_deep=True) -> 'BaseAggregate':
        """
        Return an aggregate from a context.
        Requirement: Have already pre-push the attr of dataclass into
        the context before calling this.
        """
        _dataclass = context.load_dc(deep=True)
        aggregate_class = dataclass_2_aggregate[_dataclass.__class__]
        return aggregate_class(_dataclass, context.pop_attr(), is_deep=is_deep)

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
    def parent(self) -> ParentAggregateT:
        if self._parent is None:
            self._parent = self.from_context(self.context.pop_attr(), is_deep=False)
            self._assert(isinstance(self._parent, self.parentaggregate),
                         f"parent should be of type {self.parentaggregate.__name__} but is of type {self._parent.__class__.__name__}", self.context)
        return self._parent

    @property
    def parentaggregate(self) -> Type[ParentAggregateT]:
        return {v: k for k, v in aggregate_2_subaggregate.items()}[self.__class__]

    @property
    def dataclass(self) -> Type[BaseDataclass]:
        return aggregate_2_dataclass[self.__class__]

    @property
    def subdataclass(self) -> Type[SubDataclassT]:
        return aggregate_2_dataclass[aggregate_2_subaggregate[self.__class__]]

    def sanitize_metric_name(self, metric_name: str = None):
        """
        If the argument metric is None, the optimizer's metric is taken. If the optimizer's metric is None, the parent metric is taken.
        """
        if metric_name is not None:
            return metric_name
        elif hasattr(self._dataclass, "main_metric_name") and self._dataclass.main_metric_name is not None:
            return self._dataclass.main_metric_name
        return metric_name or self.parent.sanitize_metric_name(metric_name)

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

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> Optional[bool]:
        self._managed_resource.context.free_scoped_logger_file_handler()
        # self.context.add_scoped_logger_file_handler()

        handled_err: bool = self._release_managed_subresource(self._managed_resource, exc_val)
        return handled_err

    @_with_method_as_context_manager
    def managed_subresource(self, *args, **kwds) -> SubAggregateT:
        self._managed_subresource(*args, **kwds)
        return self

    def _managed_subresource(self, *args, **kwds) -> ContextManager[SubAggregateT]:
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

    def _release_managed_subresource(self, resource: SubAggregateT, e: Exception = None) -> bool:
        """
        Release a subaggregate that was acquired with managed_subresource. The subaggregate
        normally has already saved itself. We may update things again here if needed.

        Exceptions may be handled here. If handled, return True, if not, then return False.
        """
        with self.context.lock:
            self.refresh(self.is_deep)
            self.save(False)
        handled_error = e is None
        return handled_error

    def __len__(self) -> int:
        return len(self._dataclass.get_sublocation())

    def __iter__(self) -> Iterable[SubAggregateT]:
        for subdataclass in self._dataclass.get_sublocation_values():
            if subdataclass is not None:
                yield self.subaggregate(subdataclass, self.context, is_deep=self.is_deep, parent=self)

    def __getitem__(self, item: int) -> 'Trial':
        """
        Get trial at the given index.

        :param item: trial index
        :return:
        """
        return self.subaggregate(self._dataclass.get_sublocation()[item], self.context, self.is_deep, self)

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


class Root(BaseAggregate[None, 'Project', RootDataclass]):

    def save(self, deep: bool = False):
        if deep:
            for p in self.projects:
                p.save(deep=p.is_deep)

    @property
    def projects(self) -> List['Project']:
        return list(self)

    @staticmethod
    def from_repo(context: CX, repo: HyperparamsRepository) -> 'Root':
        _dataclass: RootDataclass = repo.load(ScopedLocation())
        automl_context: AutoMLContext = AutoMLContext.from_context(context, repo)
        return Root(_dataclass, automl_context)

    @staticmethod
    def vanilla(context: CX = None) -> 'Root':
        if context is None:
            context = CX()
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


class Project(BaseAggregate[None, 'Client', ProjectDataclass]):

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


class Client(BaseAggregate[Project, 'Round', ClientDataclass]):

    @_with_method_as_context_manager
    def new_round(self, main_metric_name: str) -> 'Client':
        self._managed_subresource(new_round=True, main_metric_name=main_metric_name)
        return self

    @_with_method_as_context_manager
    def resume_last_round(self) -> 'Client':
        self._managed_subresource(new_round=False)
        return self

    @_with_method_as_context_manager
    def optim_round(self, new_round: bool, main_metric_name: str) -> 'Client':
        self._managed_subresource(new_round=new_round, main_metric_name=main_metric_name)
        return self

    def _acquire_managed_subresource(self, new_round: bool = True, main_metric_name: str = None) -> 'Round':
        with self.context.lock:
            self.refresh(self.is_deep)

            # Get new round loc:
            round_id: int = self._dataclass.get_next_i()
            if round_id == 0:
                new_round = True
            round_loc: ScopedLocation = self.loc.with_id(round_id)

            # Get round to return:
            _round_dataclass: RoundDataclass = self.repo.load(round_loc, deep=True)
            if main_metric_name is not None:
                _round_dataclass.main_metric_name = main_metric_name

            subagg: Round = Round(_round_dataclass, self.context, is_deep=True)
            if new_round:
                self.save_subaggregate(subagg, deep=True)
            return subagg


class Round(BaseAggregate[Client, 'Trial', RoundDataclass]):

    def with_optimizer(
        self, hp_optimizer: BaseHyperparameterOptimizer, hp_space: HyperparameterSpace
    ) -> 'Round':
        self.hp_optimizer: BaseHyperparameterOptimizer = hp_optimizer
        self.hp_space: HyperparameterSpace = hp_space
        return self

    @property
    def main_metric_name(self) -> str:
        return self._dataclass.main_metric_name

    @_with_method_as_context_manager
    def new_rvs_trial(self, continue_on_error: bool = False) -> 'Round':
        self._managed_subresource(new_trial=True, continue_on_error=continue_on_error)
        return self

    @_with_method_as_context_manager
    def last_trial(self, continue_on_error: bool = False) -> 'Round':
        self._managed_subresource(new_trial=False, continue_on_error=continue_on_error)
        return self

    @_with_method_as_context_manager
    def refitting_best_trial(self) -> 'Round':
        self._managed_subresource(new_trial=None, continue_on_error=False)
        return self

    @property
    def _trials(self) -> List['Trial']:
        if not self.is_deep:
            self.refresh(True)
        return [Trial(t, self.context) for t in self._dataclass.get_sublocation()]

    @property
    def get_id(self) -> int:
        return self._dataclass.get_id()

    def _acquire_managed_subresource(self, new_trial=True, continue_on_error: bool = False) -> 'Trial':
        """
        Get a trial.

        :param new_trial: If True, will create a new trial. If false, will load the last trial. If None, will load the best trial.
        :param continue_on_error: If True, will continue to the next trial if the current trial fails.
                                  Otherwise, will let the exception be raised for the failure (won't catch).
        """
        with self.context.lock:
            self.refresh(self.is_deep)
            # self.context.add_scoped_logger_file_handler()

            # Get new trial loc:
            trial_id: int = self._dataclass.get_next_i()
            if new_trial is None:
                trial_id: int = self.get_best_trial()._dataclass.get_id()
            elif not new_trial:
                # Try get last trial
                trial_id = max(0, trial_id - 1)
            trial_loc = self.loc.with_id(trial_id)

            # Get trial to return:
            _trial_dataclass: TrialDataclass = self.repo.load(trial_loc, deep=True)
            new_hps: HyperparameterSamples = self.hp_optimizer.find_next_best_hyperparams(self)
            _trial_dataclass.hyperparams = new_hps
            self.flow.log_planned(new_hps)
            subagg: Trial = Trial(_trial_dataclass, self.context, is_deep=True)

            if continue_on_error:
                subagg.continue_loop_on_error()

            if new_trial or trial_id == 0:
                self.save_subaggregate(subagg, deep=True)
            return subagg

    def _release_managed_subresource(self, resource: 'Trial', e: Exception = None) -> bool:
        """
        Release a subaggregate that was acquired with managed_subresource. The subaggregate
        normally has already saved itself. We may update things again here if needed.

        Exceptions may be handled here. If handled, return True, if not, then return False.
        """
        resource: Trial = resource  # typing
        handled_exception = False
        with self.context.lock:
            self.refresh(True)

            is_all_success: bool = resource.are_all_splits_successful()
            is_all_failure: bool = resource.are_all_splits_failures()

            if e is None:
                handled_exception = True
                if is_all_success:
                    resource.set_success()
                elif is_all_failure:
                    e = RuntimeError("All trial splits failed for this trial.")
                    resource.set_failed(e)
            else:
                resource.set_failed(e)

            self.save_subaggregate(resource, deep=resource.is_deep)

            if not is_all_failure:
                main_metric_name = self.main_metric_name
                self.flow.log('Finished round hp search!')
                self.flow.log_best_hps(
                    main_metric_name,
                    self.get_best_hyperparams(main_metric_name)
                )
            else:
                self.flow.log_failure(e)

            self.save(False)
        return handled_exception

    def get_best_hyperparams(self, metric_name: str = None) -> HyperparameterSamples:
        """
        Get best hyperparams from all trials.

        :return:
        """
        if not self.is_deep:
            self.refresh(True)

        metric_name = self.sanitize_metric_name(metric_name)

        if len(self) == 0:
            return HyperparameterSamples()
        return self.get_best_trial(metric_name).get_hyperparams()

    def get_best_trial(self, metric_name: str = None) -> Optional['Trial']:
        """
        :return: trial with best score from all trials
        """
        best_score, best_trial = None, None
        metric_name = self.sanitize_metric_name(metric_name)

        if len(self) == 0:
            raise Exception('Could not get best trial because there were no successful trial.')

        for trial in self._trials:
            trial_score = trial.get_avg_validation_score(metric_name)

            _has_better_score = best_score is None or (
                trial_score is not None and trial.is_success() and (
                    self.is_higher_score_better(metric_name) == (trial_score > best_score)
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

        metric_name = self.sanitize_metric_name(metric_name)

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
        return Round(copy.deepcopy(self._dataclass), self.context.copy(), self._parent.copy())

    def get_number_of_split(self):
        if len(self) > 0:
            return len(self[0]._validation_splits)
        return 0

    def get_metric_names(self) -> List[str]:
        if len(self) > 0:
            return list(self[-1]._dataclass.validation_splits[-1].metric_results.keys())
        return []

    def summary(
        self, metric_name: str = None
    ) -> List[Tuple[float, ScopedLocationAttrInt, FlatDict]]:
        """
        Get a summary of the round. Best score is first.
        Values in the triplet tuples are: (score, trial_number, hyperparams)

        :param metric_name:
        :return:
        """
        if not self.is_deep:
            self.refresh(True)
        metric_name = self.sanitize_metric_name(metric_name)

        results: List[float, FlatDict] = list()
        for trial in self._trials:
            score = trial.get_avg_validation_score(metric_name)
            trial_number = trial._dataclass.trial_number
            hp = trial.get_hyperparams()
            results.append((score, trial_number, hp))

        is_reverse: bool = self.is_higher_score_better(metric_name)
        results = list(sorted(results, reverse=is_reverse))
        summary = OrderedDict(results)
        return summary


class Trial(BaseAggregate[Round, 'TrialSplit', TrialDataclass]):
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

    def continue_loop_on_error(self) -> 'Trial':
        self.continue_loop_on_error: bool = True
        return self

    @property
    def _validation_splits(self) -> List['TrialSplit']:
        if not self.is_deep:
            self.refresh(True)
        return [TrialSplit(s, self.context, is_deep=self.is_deep)
                for s in self._dataclass.validation_splits]

    @property
    def main_metric_name(self) -> str:
        return self.parent.main_metric_name

    @_with_method_as_context_manager
    def new_validation_split(self, continue_loop_on_error: bool = False) -> 'Trial':
        continue_loop_on_error = continue_loop_on_error or (
            hasattr(self, "continue_loop_on_error") and self.continue_loop_on_error)

        self._managed_subresource(continue_loop_on_error=continue_loop_on_error)
        return self

    def retrain_split(self) -> 'Trial':
        self._managed_subresource(continue_loop_on_error=False, retrain_split=True)
        return self

    def _acquire_managed_subresource(self, continue_loop_on_error: bool, retrain_split=False) -> 'TrialSplit':

        self.error_types_to_raise = (
            SystemError, SystemExit, EOFError, KeyboardInterrupt
        ) if continue_loop_on_error else (Exception,)

        with self.context.lock:
            self.refresh(self.is_deep)
            # self.context.add_scoped_logger_file_handler()

            # Get new split loc:
            if retrain_split:
                split_id = RETRAIN_TRIAL_SPLIT_ID
            else:
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

    def _release_managed_subresource(self, resource: 'TrialSplit', e: Exception = None) -> bool:
        gc.collect()
        handled_error = False

        with self.context.lock:
            self.refresh(self.is_deep)
            # self.context.free_scoped_logger_handler_file()

            if e is None:
                resource.set_success()
                handled_error = True
            else:
                resource.set_failed(e)

                if any((isinstance(e, c) for c in self.error_types_to_raise)):
                    self.set_failed(e)
                    handled_error = False
                else:
                    handled_error = True

            if resource._dataclass.split_number == RETRAIN_TRIAL_SPLIT_ID:
                self._dataclass.retrained_split = resource._dataclass
                resource.save(deep=True)
                self.save(deep=False)
            else:
                self.save_subaggregate(resource, deep=True)

        return handled_error

    def is_success(self):
        """
        Checks if the trial is successful from its dataclass record.
        """
        return self._dataclass.status == TrialStatus.SUCCESS

    def are_all_splits_successful(self) -> bool:
        """
        Return true if all splits are successful.
        """
        return all(i.is_success() for i in self._validation_splits)

    def are_all_splits_failures(self) -> bool:
        """
        Return true if all splits are failed.
        """
        return all(not i.is_success() for i in self._validation_splits)

    def get_avg_validation_score(self, metric_name: str = None) -> Optional[float]:
        """
        Returns the average score for all validation splits's
        best validation score for the specified scoring metric.

        :return: validation score
        """
        if self.is_success():
            metric_name = self.sanitize_metric_name(metric_name)

            scores = [
                val_split[metric_name].get_best_validation_score()
                for val_split in self._validation_splits
                if val_split.is_success() and metric_name in val_split.get_metric_names()
            ]
            return sum(scores) / len(scores) if len(scores) > 0 else None

    def get_avg_n_epoch_to_best_validation_score(self, metric_name: str = None) -> float:
        # TODO: use in flow.log_results:
        metric_name = self.sanitize_metric_name(metric_name)

        n_epochs = [
            val_split.get_n_epochs_to_best_validation_score(metric_name)
            for val_split in self._validation_splits if val_split.is_success()
        ]

        n_epochs = sum(n_epochs) / len(n_epochs) if len(n_epochs) > 0 else None
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

    def is_higher_score_better(self, metric_name: str = None) -> bool:
        """
        Return if the metric is higher score is better.

        :param metric_name: metric name
        :return: bool
        """
        metric_name = self.sanitize_metric_name(metric_name)
        return self._dataclass.validation_splits[-1].metric_results[metric_name].higher_score_is_better


class TrialSplit(BaseAggregate[Trial, 'MetricResults', TrialSplitDataclass]):
    """
    One split of a trial. This is where a model is trained and evaluated on a specific dataset split.
    """

    def __init__(self, _dataclass: SubDataclassT, context: AutoMLContext, is_deep=False, parent: Trial = None):
        if parent is not None:
            _dataclass.hyperparams = parent.get_hyperparams()
        super().__init__(_dataclass, context, is_deep=is_deep, parent=parent)
        self.epoch: int = 0
        self.n_epochs: int = None

    def _invariant(self):
        self._assert(self.is_deep,
                     f"self.is_deep should always be set to True for "
                     f"{self.__class__.__name__}", self.context)
        super()._invariant()

    @property
    def main_metric_name(self) -> str:
        return self.parent.main_metric_name

    def get_hyperparams(self) -> RecursiveDict:
        """
        Return hyperparams dict.

        :return: hyperparams dict
        """
        return self._dataclass.hyperparams

    def with_n_epochs(self, n_epochs: int) -> 'TrialSplit':
        self.n_epochs: int = n_epochs
        return self

    def next_epoch(self) -> int:
        """
        Increment epoch. Returns the new epoch id.
        Epochs are 1-indexed like lengths: first epoch is 1, second is 2, etc.
        """
        if self.n_epochs is None:
            raise ValueError(
                "self.n_epochs is not set. Please call self.with_n_epochs(n_epochs) first on your TrialSplit.")
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

    def metric_result(self, metric_name: str = None) -> 'MetricResults':
        """
        Get a metric result not managed with a "with" statement.
        """
        metric_name = self.sanitize_metric_name(metric_name)
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

            subagg: MetricResults = self.subaggregate(subdataclass, self.context, is_deep=True, parent=self)
            return subagg

    def _create_or_get_metric_results(self, name, higher_score_is_better):
        name = self.sanitize_metric_name(name)
        if name not in self._dataclass.metric_results:
            self._dataclass.metric_results[name] = MetricResultsDataclass(
                metric_name=name,
                validation_values=[],
                train_values=[],
                higher_score_is_better=higher_score_is_better,
            )
        return self._dataclass.metric_results[name]

    def _release_managed_subresource(self, resource: 'MetricResults', e: Exception = None) -> bool:
        handled_error = False
        with self.context.lock:
            if e is not None:
                handled_error = False
                self.context.flow.log_error(e)
            else:
                handled_error = True
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


class MetricResults(BaseAggregate[TrialSplit, None, MetricResultsDataclass]):

    def _invariant(self):
        self._assert(self.is_deep,
                     f"self.is_deep should always be set to True for "
                     f"{self.__class__.__name__}", self.context)
        super()._invariant()

    @property
    def metric_name(self) -> str:
        return self._dataclass.metric_name

    def _acquire_managed_subresource(self, *args, **kwds) -> SubAggregateT:
        # TODO: epoch class and epoch location?
        raise NotImplementedError("MetricResults has no subresource to manage as a terminal resource.")

    def add_train_result(self, score: float):
        """
        Add a train metric result.

        :param name: name of the metric. If None, use the main metric name.
        :param score: the value to be logged
        :param higher_score_is_better: wheter or not a higher score is better for this metric
        """
        self._dataclass.train_values.append(score)
        self.flow.log_train_metric(self.metric_name, score)
        self.save(True)

    def add_valid_result(self, score: float):
        """
        Add a validation metric result.

        :param name: name of the metric. If None, use the main metric name.
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


aggregate_2_subaggregate: OrderedDict[Type[BaseAggregate], Type[BaseAggregate]] = OrderedDict([
    (type(None), Root),
    (Root, Project),
    (Project, Client),
    (Client, Round),
    (Round, Trial),
    (Trial, TrialSplit),
    (TrialSplit, MetricResults),
    (MetricResults, type(None)),
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
dataclass_2_aggregate: OrderedDict[BaseDataclass, BaseAggregate] = {
    dc: agg
    for agg, dc in aggregate_2_dataclass.items()
}
