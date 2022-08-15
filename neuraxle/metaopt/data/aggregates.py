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
    Copyright 2022, Neuraxio Inc.

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
import os
import typing
from abc import abstractmethod
from collections import OrderedDict
from types import TracebackType
from typing import Callable, ContextManager, Dict, Generic, Iterable, List, Optional, Type, TypeVar

import numpy as np
from neuraxle.base import BaseService, Flow, TrialStatus, _CouldHaveContext
from neuraxle.hyperparams.space import FlatDict, HyperparameterSamples, HyperparameterSpace
from neuraxle.metaopt.context import AutoMLContext
from neuraxle.metaopt.data.reporting import (BaseReport, ClientReport, MetricResultsReport, ProjectReport, RootReport,
                                             RoundReport, SubReportT, TrialReport, TrialSplitReport, dataclass_2_report)
from neuraxle.metaopt.data.vanilla import (DEFAULT_CLIENT, DEFAULT_PROJECT, RETRAIN_TRIAL_SPLIT_ID, BaseDataclass,
                                           ClientDataclass, MetricResultsDataclass, ProjectDataclass, RootDataclass,
                                           RoundDataclass, ScopedLocation, ScopedLocationAttr, ScopedLocationAttrInt,
                                           SubDataclassT, TrialDataclass, TrialSplitDataclass, dataclass_2_id_attr)
from neuraxle.metaopt.optimizer import BaseHyperparameterOptimizer
from neuraxle.metaopt.repositories.repo import (HyperparamsRepository, SynchronizedHyperparamsRepositoryWrapper,
                                                VanillaHyperparamsRepository)

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


class BaseAggregate(BaseReport, _CouldHaveContext, BaseService, ContextManager[SubAggregateT], Generic[ParentAggregateT, SubAggregateT, SubReportT, SubDataclassT]):
    """
    Base class for aggregated objects using the repo and the dataclasses to manipulate them.
    An aggregate is also a report, and can be used as a context manager for managing their children dataclasses,
    acting like a tape logging the information to the AutoML repositories.

    For more information, read this `article by Martin Fowler on DDD Aggregates <https://martinfowler.com/bliki/DDD_Aggregate.html>`_.

    .. seealso::
        :class:`neuraxle.metaopt.data.vanilla.ScopedLocation`
        :class:`neuraxle.metaopt.data.vanilla.BaseDataclass`
        :class:`neuraxle.metaopt.data.vanilla.HyperparamsRepository`
        :class:`neuraxle.metaopt.data.vanilla.AutoMLContext`
        :class:`neuraxle.metaopt.data.reporting.BaseReport`
    """

    def __init__(self, _dataclass: SubDataclassT, context: AutoMLContext, is_deep=False, parent: ParentAggregateT = None):
        BaseService.__init__(self, name=f"{self.__class__.__name__}_{_dataclass.get_id()}")
        _CouldHaveContext.__init__(self)
        dataclass_2_report[_dataclass.__class__].__init__(self, _dataclass)
        self._dataclass: SubDataclassT = _dataclass
        self._spare: SubDataclassT = copy.copy(_dataclass).shallow()
        # TODO: pre-push context to allow for dc auto-loading and easier parent auto-loading?
        self.context: AutoMLContext = context.push_attr(_dataclass).add_scoped_logger_file_handler()
        self.loc: ScopedLocation = self.context.loc._copy()
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
        with context.repo.lock:
            _dataclass = context.load_dc(deep=True)

        if isinstance(_dataclass, (RootDataclass, ProjectDataclass)) and len(_dataclass) == 0:
            raise ValueError("Len 0 while it should be longer:" + str(_dataclass))
        aggregate_class = dataclass_2_aggregate[_dataclass.__class__]
        return aggregate_class(_dataclass, context.pop_attr(), is_deep=is_deep)

    @classmethod
    def dummy(cls: Type['BaseAggregate'], context: AutoMLContext = None) -> 'BaseAggregate':
        """
        Create a dummy object of the desired type for testing purposes.
        The parent subtree will be created in a temporary repository,
        or in a real repository if the context is provided.
        """
        context = context or AutoMLContext.from_context(context)

        # create dataclass tree:
        dataclasses: List[BaseDataclass] = dataclass_2_id_attr.keys()
        root_dc: RootDataclass = RootDataclass()
        target_dc_class: Type[BaseDataclass] = aggregate_2_dataclass[cls]
        prev_dc: BaseDataclass = root_dc
        for _dataclass in dataclasses:
            # loop for every type of dataclass and create them until we find the one we want:
            dc: BaseDataclass = _dataclass()
            prev_dc.store(dc)

            prev_dc = dc
            context = context.push_attr(dc)
            if isinstance(dc, target_dc_class):
                break

        # create the aggregate:
        with context.repo.lock:
            context.repo.save(root_dc, ScopedLocation(), deep=True)
            return cls.from_context(context, is_deep=True)

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
    def repo(self) -> SynchronizedHyperparamsRepositoryWrapper:
        return self.context.repo

    def subaggregate(self, _dataclass: SubDataclassT, context: AutoMLContext, is_deep=False, parent: ParentAggregateT = None) -> SubAggregateT:
        return aggregate_2_subaggregate[self.__class__](_dataclass, context, is_deep, parent)

    @property
    def parent(self) -> ParentAggregateT:
        if self._parent is None and not isinstance(self, Root):
            self._parent = self.from_context(self.context.pop_attr(), is_deep=False)
            self._assert(isinstance(self._parent, self.parentaggregate),
                         f"parent should be of type {self.parentaggregate.__name__} but is of type {self._parent.__class__.__name__}", self.context)
        return self._parent

    @property
    def parentaggregate(self) -> Type[ParentAggregateT]:
        return {v: k for k, v in aggregate_2_subaggregate.items()}[self.__class__]

    @property
    def report(self) -> SubReportT:
        return aggregate_2_report[self.__class__](self._dataclass)

    @property
    def dataclass(self) -> Type[BaseDataclass]:
        return aggregate_2_dataclass[self.__class__]

    @property
    def subdataclass(self) -> Type[SubDataclassT]:
        return aggregate_2_dataclass[aggregate_2_subaggregate[self.__class__]]

    def refresh(self, deep: bool = True):
        with self.repo.lock:
            _new_dc: SubDataclassT = self.repo.load(self.loc, deep=deep)
            if len(_new_dc) < len(self._dataclass):
                _new_dc: SubDataclassT = self.repo.load(self.loc, deep=deep)
                raise ValueError("Loaded dataclass can't have shorter sublocation than it already did.")

            def _fail_on_unsaved_changes(_self_now: SubDataclassT, _self_before: SubDataclassT, _self_new_loaded: SubDataclassT):
                _has_self_changed = _self_now != _self_before
                if _has_self_changed:
                    _is_new_dc_different_from_now = _self_now != _self_new_loaded
                    if _is_new_dc_different_from_now:

                        raise RuntimeError(f"{_self_now.__class__.__name__} with id `{_self_now.get_id()}` "
                                           f"has unsaved changes while it tried to refresh itself. "
                                           f"Before changes: {_self_before}. "
                                           f"After changes, that is not saved: {_self_now}. "
                                           f"DC that we tried to load: {_self_new_loaded}. ")
            _fail_on_unsaved_changes(
                self._dataclass.shallow(),
                self._spare.shallow(),
                _new_dc.shallow()
            )

            self._dataclass = _new_dc

        self.is_deep = deep
        self._invariant()

    def save(self, deep: bool = True) -> 'BaseAggregate':
        if deep and deep != self.is_deep:
            raise ValueError(
                f"Cannot save {str(self)} with deep=True when self "
                f"is not already deep. You might want to use self.refresh(deep=True) at "
                f"some point to refresh self before saving deeply then.")

        self._invariant()
        self._spare = copy.copy(self._dataclass).shallow()

        with self.repo.lock:
            self.repo.save(self._dataclass, self.loc, deep=deep)
        return self

    def save_subaggregate(self, subagg: SubAggregateT, deep=False) -> 'BaseAggregate':
        self._dataclass.store(subagg._dataclass)

        with self.repo.lock:
            self.save(deep=False)
            subagg.save(deep=deep)

            _spare_reloaded = self.repo.load(self.loc, deep=deep)
            try:
                assert _spare_reloaded.shallow() == self._spare, ("reloaded spare different than spare", _spare_reloaded.shallow(), self._spare)
                assert _spare_reloaded.shallow() == self._dataclass.shallow(
                ), ("reloaded spare different than self", self._dataclass.shallow(), self._spare)
            except Exception as e:
                _spare_reloaded = self.repo.load(self.loc, deep=deep)
                raise e from e

        return self

    def __enter__(self) -> SubAggregateT:
        # self.context.free_scoped_logger_handler_file()
        self._invariant()
        self._managed_resource._invariant()
        return self._managed_resource

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> Optional[bool]:
        with self.repo.lock:  # TODO: locking twice, probably not needed.
            handled_err: bool = self._release_managed_subresource(self._managed_resource, exc_val)
            self._managed_resource.context.free_scoped_logger_file_handler()
        return handled_err

    @_with_method_as_context_manager
    def managed_subresource(self, *args, **kwds) -> SubAggregateT:
        self._managed_subresource(*args, **kwds)
        return self

    def _managed_subresource(self, *args, **kwds) -> ContextManager[SubAggregateT]:
        self._invariant()
        with self.repo.lock:
            self.refresh(self.is_deep)
            self._managed_resource: SubAggregateT = self._acquire_managed_subresource(
                *args, **kwds)
            self.save_subaggregate(self._managed_resource, deep=False)
        return self

    @abstractmethod
    def _acquire_managed_subresource(self, *args, **kwds) -> SubAggregateT:
        """
        Acquire a new subaggregate that is managed such that it is deep saved at the
        beginning.
        Generally:
            1. Refresh self,
            2. load subagg to fetch or create it,
            3. save loaded subagg in case it changed, and return it.
        """
        # Create subaggregate:
        subdataclass: SubDataclassT = self.repo.load(*args, **kwds)
        subagg: SubAggregateT = self.subaggregate(subdataclass, self.context, is_deep=False, parent=self)
        return subagg

    def _release_managed_subresource(self, resource: SubAggregateT, e: Exception = None) -> bool:
        """
        Release a subaggregate that was acquired with managed_subresource. The subaggregate
        normally has already saved itself. We may update things again here if needed.

        Exceptions may be handled here. If handled, return True, if not, then return False.
        """
        with self.repo.lock:
            self.refresh(self.is_deep)
            self.save(False)  # TODO: is this bad?

        handled_error = e is None
        return handled_error

    def __len__(self) -> int:
        return len(self._dataclass.get_sublocation())

    def __iter__(self) -> Iterable[SubAggregateT]:
        if not self.is_deep:
            self.refresh(True)
        for subdataclass in self._dataclass.get_sublocation_values():
            if subdataclass is not None:
                yield self.subaggregate(subdataclass, self.context, is_deep=True, parent=self)

    def __getitem__(self, item: int) -> 'Trial':
        """
        Get trial at the given index.

        :param item: trial index
        :return:
        """
        return self.subaggregate(self._dataclass.get_sublocation()[item], self.context, is_deep=True, parent=self)

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


class Root(RootReport, BaseAggregate[None, 'Project', RootReport, RootDataclass]):

    def save(self, deep: bool = False):
        if deep:
            for p in self.projects:
                p.save(deep=p.is_deep)

    @property
    def projects(self) -> List['Project']:
        return list(self)

    @staticmethod
    def from_repo(context: AutoMLContext, repo: HyperparamsRepository) -> 'Root':
        _dataclass: RootDataclass = repo.load(ScopedLocation())
        automl_context: AutoMLContext = AutoMLContext.from_context(context, repo)
        return Root(_dataclass, automl_context)

    @staticmethod
    def vanilla(context: AutoMLContext = None) -> 'Root':
        if context is None:
            context = AutoMLContext.from_context()
        vanilla_repo: HyperparamsRepository = VanillaHyperparamsRepository(
            os.path.join(context.get_path(), "hyperparams"))
        return Root.from_repo(context, vanilla_repo)

    @_with_method_as_context_manager
    def get_project(self, name: str) -> 'Root':
        self._managed_subresource(project_name=name)
        return self

    @_with_method_as_context_manager
    def default_project(self) -> 'Root':
        self._managed_subresource(project_name=DEFAULT_PROJECT)
        return self

    def _acquire_managed_subresource(self, project_name: str) -> 'Project':
        project_loc: ScopedLocation = self.loc.with_id(project_name)
        return super()._acquire_managed_subresource(project_loc)


class Project(ProjectReport, BaseAggregate[None, 'Client', ProjectReport, ProjectDataclass]):

    @_with_method_as_context_manager
    def get_client(self, name: str) -> 'Project':
        self._managed_subresource(client_name=name)
        return self

    @_with_method_as_context_manager
    def default_client(self) -> 'Project':
        self._managed_subresource(client_name=DEFAULT_CLIENT)
        return self

    def _acquire_managed_subresource(self, client_name: str) -> 'Client':
        client_loc: ScopedLocation = self.loc.with_id(client_name)
        return super()._acquire_managed_subresource(client_loc)


class Client(ClientReport, BaseAggregate[Project, 'Round', ClientReport, ClientDataclass]):

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
        # Get new round loc:
        round_id: int = self._dataclass.get_next_i()
        if not new_round:
            round_id = max(0, round_id - 1)
        if round_id == 0:
            new_round = True
        round_loc: ScopedLocation = self.loc.with_id(round_id)

        # Get round to save it and return:
        _round_dataclass: RoundDataclass = self.repo.load(round_loc, deep=True)
        if main_metric_name is not None:
            _round_dataclass.main_metric_name = main_metric_name

        subagg: Round = Round(_round_dataclass, self.context, is_deep=True)
        return subagg


class Round(RoundReport, BaseAggregate[Client, 'Trial', RoundReport, RoundDataclass]):

    def with_optimizer(
        self,
        hp_optimizer: BaseHyperparameterOptimizer,
        hp_space: HyperparameterSpace
    ) -> 'Round':
        self.hp_optimizer: BaseHyperparameterOptimizer = hp_optimizer
        self.hp_space: HyperparameterSpace = hp_space
        return self

    def with_metric(self, metric_name: str) -> 'Round':
        self._dataclass.main_metric_name = metric_name
        return self.save(False)

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
        self.refresh(deep=True)
        self._managed_subresource(new_trial=None, continue_on_error=False)
        return self

    @property
    def _trials(self) -> List['Trial']:
        if not self.is_deep:
            self.refresh(True)
        return [Trial(t, self.context) for t in self._dataclass.get_sublocation()]

    @property
    def round_number(self) -> int:
        return self._dataclass.round_number

    def _acquire_managed_subresource(self, new_trial: Optional[bool] = True, continue_on_error: bool = False) -> 'Trial':
        """
        Get a trial. If new_trial is None, refit the best trial.

        :param new_trial: If True, will create a new trial. If false, will load the last trial. If None, will load the best trial.
        :param continue_on_error: If True, will continue to the next trial if the current trial fails.
                                  Otherwise, will let the exception be raised for the failure (won't catch).
        """
        if not self.is_deep:
            raise RuntimeError("Round must be deep to get a trial.")

        # Get new trial loc:
        trial_id: int = self._dataclass.get_next_i()
        if new_trial is None:
            trial_id: int = self.get_best_trial_id()
        elif not new_trial:
            # Try get last trial
            trial_id = max(0, trial_id - 1)
        if trial_id > len(self._dataclass) + 1:
            raise ValueError(
                f"Can't create a trial with id {trial_id} when Round contains only {len(self._dataclass)} trials.")
        trial_loc = self.loc.with_id(trial_id)

        # Get trial to return, log it, and save it if new:
        _trial_dataclass: TrialDataclass = self.repo.load(trial_loc, deep=True)
        if new_trial is True:
            # TODO: pass self.report as arg instead in the find_next_best_hyperparams method, but for this we need space stored in report as well:
            new_hps: HyperparameterSamples = self.hp_optimizer.find_next_best_hyperparams(self.report, self.hp_space)
            # assert new_hps.to_flat_dict() not in self.report.get_all_hyperparams(as_flat=True, use_wildcards=False)  # TODO: TMP.
            _trial_dataclass.hyperparams = new_hps
            self.flow.log_planned(trial_id, _trial_dataclass.hyperparams)
        elif new_trial is False:
            self.flow.log_continued(trial_id)
        else:
            self.flow.log_retraining(trial_id, _trial_dataclass.hyperparams)

        subagg: Trial = Trial(_trial_dataclass, self.context.new_trial(), is_deep=True)
        if continue_on_error:
            subagg.continue_loop_on_error()
        return subagg

    def _release_managed_subresource(self, resource: 'Trial', e: Exception = None) -> bool:
        """
        Release a subaggregate that was acquired with managed_subresource. The subaggregate
        normally has already saved itself. We may update things again here if needed.

        Exceptions may be handled here. If handled, return True, if not, then return False.
        """
        resource: Trial = resource  # typing
        handled_exception = False
        with self.repo.lock:
            self.refresh(True)

            is_all_success: bool = resource.are_all_splits_successful()
            is_all_failure: bool = resource.are_all_splits_failures()

            if e is None:
                handled_exception = True
                if is_all_success:
                    resource._set_success()
                elif is_all_failure:
                    e = RuntimeError("All trial splits failed for this trial.")
                    resource._set_failed(e)
            else:
                resource._set_failed(e)

            self.save_subaggregate(resource, deep=resource.is_deep)

            if not is_all_failure and len(self) > 0:
                main_metric_name = self.main_metric_name
                self.flow.log('Finished round hp search!')
                try:
                    _best_trial: TrialReport = self.get_best_trial(main_metric_name)
                except Exception as err:
                    raise err from err
                self.flow.log_best_hps(
                    main_metric_name,
                    _best_trial.get_hyperparams(),
                    _best_trial.get_avg_validation_score(main_metric_name),
                    _best_trial.get_avg_n_epoch_to_best_validation_score(main_metric_name)
                )
            else:
                self.flow.log_failure(e or ValueError(
                    f"The current Round #{self.get_id()} of length {len(self.report)} seems to contains only failed trials."))

            self.save(False)
        return handled_exception

    def append(self, trial: 'Trial'):
        """
        Add a new trial. Will also save the trial shallowly.

        :param trial: new trial
        :return:
        """
        self.save_subaggregate(trial, deep=False)

    @property
    def main_metric_name(self) -> str:
        return self._dataclass.main_metric_name


class Trial(TrialReport, BaseAggregate[Round, 'TrialSplit', TrialReport, TrialDataclass]):
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
        self._continue_loop_on_error: bool = True
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
            hasattr(self, "_continue_loop_on_error") and self._continue_loop_on_error)

        self._managed_subresource(continue_loop_on_error=continue_loop_on_error)
        return self

    @_with_method_as_context_manager
    def retrain_split(self) -> 'Trial':
        self._managed_subresource(continue_loop_on_error=False, retrain_split=True)
        return self

    def _acquire_managed_subresource(self, continue_loop_on_error: bool, retrain_split=False) -> 'TrialSplit':
        self.error_types_to_raise = (
            SystemError, SystemExit, EOFError, KeyboardInterrupt
        ) if continue_loop_on_error else (Exception,)

        # Get new split loc:
        if retrain_split:
            split_id = RETRAIN_TRIAL_SPLIT_ID
        else:
            split_id: int = self._dataclass.get_next_i()
            split_id = max(0, split_id)
            if split_id == 0:
                self.flow.log_start()
        split_loc = self.loc.with_id(split_id)

        # Get split to save and return it:
        _split_dataclass: TrialSplitDataclass = self.repo.load(split_loc)
        _split_dataclass.hyperparams = self.get_hyperparams()

        subagg: TrialSplit = TrialSplit(_split_dataclass, self.context.new_trial_split(), is_deep=True)
        return subagg

    def _release_managed_subresource(self, resource: 'TrialSplit', e: Exception = None) -> bool:
        gc.collect()
        handled_error = False

        with self.repo.lock:
            self.refresh(self.is_deep)
            # self.context.free_scoped_logger_handler_file()

            if e is None:
                resource._set_success()
                handled_error = True
            else:
                resource._set_failed(e)

                if any((isinstance(e, c) for c in self.error_types_to_raise)):
                    self._set_failed(e)
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

    def _set_success(self) -> 'Trial':
        """
        Set trial status to success. Must save after to ensure consistency.
        """
        self._dataclass.end(TrialStatus.SUCCESS)

        metric_name = self.parent.main_metric_name
        avg_best_val_score = self.get_avg_validation_score(metric_name)
        avg_n_epochs_to_val_score = self.get_avg_n_epoch_to_best_validation_score(metric_name)
        self.flow.log_success(avg_best_val_score, avg_n_epochs_to_val_score, metric_name)
        return self

    def _set_failed(self, error: Exception) -> 'Trial':
        """
        Set failed trial with exception. Must save after to ensure consistency.

        :param error: catched exception
        :return: self
        """
        self._dataclass.end(TrialStatus.FAILED)
        self.flow.log_failure(exception=error)
        return self


class TrialSplit(TrialSplitReport, BaseAggregate[Trial, 'MetricResults', TrialSplitReport, TrialSplitDataclass]):
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
        with self.repo.lock:
            self.flow.log_start()
            self._dataclass.start()
            self.save(False)
        return self

    @property
    def metric_results(self) -> Dict[str, 'MetricResults']:
        return {
            metric_name: MetricResults(mr, self.context)  # .at_epoch(self.epoch, self.n_epochs)
            for metric_name, mr in self._dataclass.metric_results.items()
        }

    def metric_result(self, metric_name: str = None) -> 'MetricResults':
        """
        Get a metric result that is not managed with a "with" statement. Access it read-only.
        """
        metric_name = metric_name or self.parent.parent.main_metric_name
        mr: MetricResultsDataclass = self._dataclass.get_sublocation()[metric_name]
        return MetricResults(mr, self.context, is_deep=True)

    @_with_method_as_context_manager
    def managed_metric(self, metric_name: str, higher_score_is_better: bool) -> 'TrialSplit':
        """
        To be used as a with statement to get the managed metric.
        """
        self._managed_subresource(metric_name=metric_name, higher_score_is_better=higher_score_is_better)
        return self

    def _acquire_managed_subresource(self, metric_name: str, higher_score_is_better: bool) -> 'MetricResults':
        subdataclass: MetricResultsDataclass = self._create_or_get_metric_results(
            metric_name, higher_score_is_better)

        subagg: MetricResults = self.subaggregate(subdataclass, self.context, is_deep=True, parent=self)
        return subagg

    def _create_or_get_metric_results(self, metric_name, higher_score_is_better):
        if metric_name not in self._dataclass.metric_results:
            self._dataclass.metric_results[metric_name] = MetricResultsDataclass(
                metric_name=metric_name,
                validation_values=[],
                train_values=[],
                higher_score_is_better=higher_score_is_better,
            )
        return self._dataclass.metric_results[metric_name]

    def _release_managed_subresource(self, resource: 'MetricResults', e: Exception = None) -> bool:
        handled_error = False
        with self.repo.lock:
            if e is not None:
                handled_error = False
                self.flow.log_error(e)
            else:
                handled_error = True
            self.save_subaggregate(resource, deep=True)
        return handled_error

    def _set_success(self) -> 'TrialSplit':
        """
        Set trial status to success. Must save after to ensure consistency.

        :return: self
        """
        self._dataclass.end(status=TrialStatus.SUCCESS)

        self_metrics = self.get_metric_names()

        metric_name = self.parent.main_metric_name
        if metric_name not in self_metrics:
            if len(self_metrics) == 0:
                self.flow.log_warning(
                    f"TrialSplit {self._dataclass.get_id()} has no metrics. Please add a metric before setting to success.")
                metric_name = None
            else:
                metric_name = self_metrics[0]

        best_val_score = self.metric_result(
            metric_name).get_best_validation_score() if metric_name is not None else None
        n_epochs_to_val_score = self.metric_result(
            metric_name).get_n_epochs_to_best_validation_score() if metric_name is not None else None

        self.flow.log_success(best_val_score, n_epochs_to_val_score, metric_name)
        return self

    def _set_failed(self, error: Exception) -> 'TrialSplit':
        """
        Set failed trial with exception. Must save after to ensure consistency.

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

    def train_context(self) -> 'AutoMLContext':
        return self.context.train()

    def validation_context(self) -> 'AutoMLContext':
        return self.context.validation()


class MetricResults(MetricResultsReport, BaseAggregate[TrialSplit, None, MetricResultsReport, MetricResultsDataclass]):

    def _invariant(self):
        self._assert(self.is_deep,
                     f"self.is_deep should always be set to True for "
                     f"{self.__class__.__name__}", self.context)
        super()._invariant()

    @property
    def metric_name(self) -> str:
        return self._dataclass.metric_name

    def _acquire_managed_subresource(self, *args, **kwds) -> SubAggregateT:
        raise NotImplementedError("MetricResults has no subresource to manage as a terminal resource.")

    def add_train_result(self, score: float):
        """
        Add a train metric result.

        :param name: name of the metric. If None, use the main metric name.
        :param score: the value to be logged
        :param higher_score_is_better: wheter or not a higher score is better for this metric
        """
        with self.repo.lock:
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
        with self.repo.lock:
            self._dataclass.validation_values.append(score)
            self.flow.log_valid_metric(self.metric_name, score)
            self.save(True)

    def __iter__(self) -> Iterable[SubAggregateT]:
        """
        Loop over validation values.
        """
        return self._dataclass.validation_values


aggregate_2_subaggregate: typing.OrderedDict[Type[BaseAggregate], Type[BaseAggregate]] = OrderedDict([
    (type(None), Root),
    (Root, Project),
    (Project, Client),
    (Client, Round),
    (Round, Trial),
    (Trial, TrialSplit),
    (TrialSplit, MetricResults),
    (MetricResults, type(None)),
])

aggregate_2_report: typing.OrderedDict[Type[BaseAggregate], Type[BaseReport]] = OrderedDict([
    (Root, RootReport),
    (Project, ProjectReport),
    (Client, ClientReport),
    (Round, RoundReport),
    (Trial, TrialReport),
    (TrialSplit, TrialSplitReport),
    (MetricResults, MetricResultsReport),
])

aggregate_2_dataclass: typing.OrderedDict[BaseAggregate, BaseDataclass] = OrderedDict([
    (Root, RootDataclass),
    (Client, ClientDataclass),
    (Round, RoundDataclass),
    (Project, ProjectDataclass),
    (Trial, TrialDataclass),
    (TrialSplit, TrialSplitDataclass),
    (MetricResults, MetricResultsDataclass),
])
dataclass_2_aggregate: typing.OrderedDict[BaseDataclass, BaseAggregate] = {
    dc: agg
    for agg, dc in aggregate_2_dataclass.items()
}
