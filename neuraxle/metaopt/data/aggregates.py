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

import gc
import copy
import datetime
import hashlib
import json
import logging
import os
import traceback
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from json.encoder import JSONEncoder
from logging import FileHandler, Logger
from typing import (Any, Callable, Dict, Generic, Iterable, List, Optional,
                    Tuple, Type, TypeVar, Union)

import numpy as np
from neuraxle.base import (_HasChildrenMixin, BaseStep, ExecutionContext, Flow,
                           synchroneous_flow_method, TrialStatus)
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace, RecursiveDict)
from neuraxle.metaopt.data.vanilla import (DEFAULT_CLIENT, DEFAULT_PROJECT, AutoMLContext, AutoMLFlow,
                                           BaseDataclass,
                                           BaseHyperparameterOptimizer,
                                           ClientDataclass,
                                           HyperparamsRepository,
                                           InMemoryHyperparamsRepository,
                                           MetricResultsDataclass,
                                           ProjectDataclass, RecursiveDict,
                                           RootDataclass, RoundDataclass,
                                           ScopedLocation, ScopedLocationAttr,
                                           SubDataclassT, TrialDataclass,
                                           TrialSplitDataclass, VanillaHyperparamsRepository)

SubAggregateT = TypeVar('SubAggregateT', bound=Optional['BaseAggregate'])
# TODO: there are probably errors where the subtype is used where the base type should be used instead. Check this typing error. Might be bad with constructors. Will be checked in unit tests?


class BaseAggregate(Generic[SubAggregateT, SubDataclassT]):
    """
    Base class for aggregated objects using the repo and the dataclasses to manipulate them.
    """

    def __init__(self, _dataclass: SubDataclassT, context: AutoMLContext, is_deep=False):
        self._dataclass: SubDataclassT = _dataclass
        self.context: AutoMLContext = context.push_attr(_dataclass)
        self.loc: ScopedLocation = copy.copy(context.loc)
        self.is_deep = is_deep

    def without_context(self) -> 'BaseAggregate':
        """
        Return a copy of this aggregate without the context. Useful for initializing a temporary aggregate, such as a filtered or reduced aggregate without all its subaggregates
        to disallow saving the reduced aggregate.
        """
        self_copy = copy.copy(self)
        self_copy.context = None
        return self_copy

    @property
    def flow(self) -> AutoMLFlow:
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
        return aggregate_2_dataclass[self.__class__]

    def refresh(self, deep: bool = True):
        with self.context.lock:
            self._dataclass = self.repo.load(self.loc, deep=deep)
        self.is_deep = deep

    def save(self, deep: bool = True):
        if deep and deep != self.is_deep:
            raise ValueError(
                f"Cannot save {str(self)} with self.is_deep=False when self "
                f"is not already deep. You might want to use self.refresh(deep=True) at "
                f"some point to refresh self before saving deeply then.")

        with self.context.lock:
            self.repo.save(self._dataclass, self.loc, deep=deep)

    def save_subaggregate(self, subagg: SubAggregateT, deep=False):
        self._dataclass.store(subagg._dataclass)
        with self.context.lock:
            self.save(deep=False)
            subagg.save(deep=deep)

    def __enter__(self) -> SubAggregateT:
        return self._managed_resource

    def __exit__(self, exc_type, exc_val, exc_tb):
        handled_err: bool = self._release_managed_subresource(self._managed_resource, exc_val)
        return handled_err

    # @contextmanager
    def managed_subresource(self, *args, **kwds) -> SubAggregateT:
        self._managed_subresource(*args, **kwds)
        return self

    def _managed_subresource(self, *args, **kwds) -> SubAggregateT:
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
            self.refresh(False)
            subdataclass: SubDataclassT = self.repo.load(*args, **kwds)
            _subklass: Type[SubAggregateT] = self.subaggregate(subdataclass, self.context)
            subagg: SubAggregateT = _subklass(subdataclass, self.context)
            self.save_subaggregate(subagg, deep=True)
            return subagg

    def _release_managed_subresource(self, resource: SubAggregateT, e: Exception = None) -> Optional[Exception]:
        """
        Release a subaggregate that was acquired with managed_subresource. The subaggregate
        normally has already saved itself. We may update things again here if needed.

        Exceptions may be handled here. If handled, return True, if not, then return False.
        """
        with self.context.lock:
            self.refresh(False)
            self.save(False)
        return False

    def __len__(self) -> int:
        return len(self._dataclass.get_sublocation())

    def __iter__(self) -> Iterable[SubAggregateT]:
        return iter((
            self.subaggregate(subdataclass, self.context)
            for subdataclass in self._dataclass.get_sublocation()
        ))

    def __getitem__(self, item: int) -> 'Trial':
        """
        Get trial at the given index.

        :param item: trial index
        :return:
        """
        return self.subaggregate()(self._dataclass.get_sublocation()[item], self.context)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str([str(t) for t in self.__iter__()])})"


class Root(BaseAggregate['Project', RootDataclass]):

    def save(self, deep: bool = False):
        if deep:
            for p in self.projects:
                p.save(deep=True)

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

    # @contextmanager
    def get_project(self, name: str) -> 'Root':
        # TODO: new project should be created if it does not exist?
        self._managed_subresource(project_name=name)
        return self

    # @contextmanager
    def default_project(self) -> 'Root':
        self._managed_subresource(project_name=DEFAULT_PROJECT)
        return self

    def _acquire_managed_subresource(self, project_name: str) -> 'Project':
        project_loc: ScopedLocation = self.loc.with_id(project_name)
        super()._acquire_managed_subresource(project_loc)


class Project(BaseAggregate['Client', ProjectDataclass]):

    # @contextmanager
    def get_client(self, name: str) -> 'Project':
        # TODO: new client should be created if it does not exist?
        self._managed_subresource(client_name=name)
        return self

    # @contextmanager
    def default_client(self) -> 'Project':
        self._managed_subresource(client_name=DEFAULT_CLIENT)
        return self

    def _acquire_managed_subresource(self, client_name: str) -> 'Client':
        client_loc: ScopedLocation = self.loc.with_id(client_name)
        super()._acquire_managed_subresource(client_loc)


class Client(BaseAggregate['Round', ClientDataclass]):

    def new_round(self, hps: HyperparameterSpace) -> 'Client':
        self._managed_subresource(hps=hps, new_round=True)
        return self

    def resume_last_round(self, hps: HyperparameterSpace) -> 'Client':
        self._managed_subresource(hps=hps, new_round=False)
        return self

    def _acquire_managed_subresource(self, new_round=True) -> 'Round':
        with self.context.lock:
            self.refresh(False)

            # Get new round loc:
            round_id: int = self._dataclass.get_next_i()
            if not new_round:
                # Try get last round:
                round_id = max(0, round_id - 1)
            round_loc = self.loc.with_id(round_id)

            # Get round to return:
            _round_dataclass: RoundDataclass = self.repo.load(round_loc)
            subagg: Round = Round(_round_dataclass, self.context)

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

    # @contextmanager
    def new_rvs_trial(self) -> 'Round':
        self._managed_subresource(new_trial=True)
        return self

    # @contextmanager
    def last_trial(self) -> 'Round':
        self._managed_subresource(new_trial=False)
        return self

    @property
    def _trials(self) -> List['Trial']:
        return [Trial(t, self.context) for t in self._dataclass.get_sublocation()]

    def _acquire_managed_subresource(self, new_trial=True) -> 'Trial':
        with self.context.lock:
            self.refresh(False)
            self.flow.add_file_handler_to_logger(
                self.repo.get_scoped_logger_path(self.loc))

            # Get new trial loc:
            trial_id: int = self._dataclass.get_next_i()
            if not new_trial:
                # Try get last trial
                trial_id = max(0, trial_id - 1)
            trial_loc = self.loc.with_id(trial_id)

            # Get trial to return:
            _trial_dataclass: TrialDataclass = self.repo.load(trial_loc)
            new_hps: HyperparameterSamples = self.hp_optimizer.find_next_best_hyperparams(self)
            _trial_dataclass.hyperparams(new_hps)
            self.flow.log_planned(new_hps)
            subagg: Trial = Trial(_trial_dataclass, self.context)

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
            self.refresh(False)

            if e is not None:
                self.flow.log_error(e)

            self.flow.log_best_hps(
                self.hp_optimizer.get_main_metric_name(),
                self.get_best_hyperparams()
            )
            self.flow.log('Finished round hp search!')

            self.flow.free_logger_files()

            self.save(False)
        return False

    def get_best_hyperparams(self) -> HyperparameterSamples:
        """
        Get best hyperparams from all trials.

        :return:
        """
        self.refresh(True)  # TODO: is this too long to perform?

        if len(self) == 0:
            return HyperparameterSamples()

        return self.get_best_trial().get_hyperparams()

    def get_best_trial(self) -> Optional['Trial']:
        """
        :return: trial with best score from all trials
        """
        best_score, best_trial = None, None

        if len(self) == 0:
            raise Exception('Could not get best trial because there were no successful trial.')

        for trial in self._trials:
            trial_score = trial.get_avg_validation_score()

            if best_score is None or self.is_higher_score_better() == (trial_score > best_score):
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
            return self[-1]._dataclass.validation_splits[-1].metric_results.keys()
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

    # @contextmanager
    def new_split(self, continue_loop_on_error: bool) -> 'Trial':
        self._acquire_managed_subresource(
            continue_loop_on_error=continue_loop_on_error)
        return self

    @property
    def _validation_splits(self) -> List['TrialSplit']:
        # TODO: if is not deep: self.refresh(True)?
        return [TrialSplit(s, self.context) for s in self._dataclass.validation_splits]

    def _acquire_managed_subresource(self, continue_loop_on_error: bool) -> 'TrialSplit':

        self.error_types_to_raise = (
            SystemError, SystemExit, EOFError, KeyboardInterrupt
        ) if continue_loop_on_error else (Exception,)

        with self.context.lock:
            self.refresh(False)

            # TODO: active record design pattern such that the dataobjects are the ones containing the repo and saving themselves / finding themselves. Clean Code P.101.
            #     Active record should probably have a weakref of the repo since it doesn<t want to save the repo into itself.
            #     Active record should have its own location integrated into itself!!! When creating a subdataclass, pass the parent into ctor to have a weakref of the parent as well. Pickling to remove refs.
            #     Active record<s loc should not be known to the callers.

            #     TODO: FIND A WAY TO SAVE THE REPO INTO THE DATACLASS. And hide it from the rest. Where does the flow goes as well?

            # Get new split loc:
            split_id: int = self._dataclass.get_next_i()
            split_id = max(0, split_id - 1)
            if split_id == 0:
                self.flow.log_start()
            split_loc = self.loc.with_id(split_id)

            # Get split to return:
            _split_dataclass: TrialSplitDataclass = self.repo.load(split_loc)
            subagg: TrialSplit = TrialSplit(_split_dataclass, self.context)
            # TODO: logger loc and file for context.push_attr?

            self.save_subaggregate(subagg, deep=True)
            return subagg

    def _release_managed_subresource(self, resource: 'TrialSplit', e: Exception = None) -> Optional[Exception]:
        gc.collect()
        handled_error = True

        with self.context.lock:
            self.refresh(False)
            self.save_subaggregate(resource, deep=True)

            if e is None:
                self.flow.log_success()
            else:
                if isinstance(e, SystemExit):
                    self.flow.log_aborted()
                else:
                    self.flow.log_failure(e)

                if any((isinstance(e, c) for c in self.error_types_to_raise)):
                    handled_error = False

            self.save_subaggregate(resource)
        return handled_error

    def get_avg_validation_score(self, metric_name: str) -> float:
        """
        Returns the average score for all validation splits's
        best validation score for the specified scoring metric.

        :return: validation score
        """
        scores = [
            val_split.get_best_validation_score(metric_name)
            for val_split in self._validation_splits
            if val_split.is_success()
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
        self._dataclass.status = TrialStatus.SUCCESS
        self.save_trial()  # TODO?

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
        self.status = TrialStatus.FAILED
        self.error = str(error)
        self.error_traceback = traceback.format_exc()

        self.save_trial()

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

    def __init__(self, _dataclass: SubDataclassT, context: AutoMLContext):
        super().__init__(_dataclass, context, is_deep=True)
        self.epoch: int = 0
        self.n_epochs: int = None

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
        # TODO: lock?
        if self.epoch == 0:
            self.flow.log_start()
        self.epoch: int = self._dataclass.get_next_i()
        self.flow.log_epoch(self.epoch, self.n_epochs)
        return self.epoch

    def _acquire_managed_subresource(self, epoch: int, n_epochs: int) -> 'MetricResults':
        """
        Epoch is zero-indexed.
        """
        # Should use self.metric_result instead:
        raise NotImplementedError("Please use self.metric_results instead.")

    @property
    def metric_results(self) -> Dict[str, 'MetricResults']:
        return {
            metric_name: MetricResults(mr, self.context).at_epoch(self.epoch, self.n_epochs)
            for metric_name, mr in self._dataclass.metric_results.items()
        }

    def train_context(self) -> 'AutoMLContext':
        return self.context.train()

    def validation_context(self) -> 'AutoMLContext':
        return self.context.validation()

    def get_metric_names(self) -> List[str]:
        return list(self._dataclass.metric_results.keys())

    def add_metric_results_train(self, name: str, score: float, higher_score_is_better: bool, log_metric: bool = False):
        """
        Add a train metric result in the metric results dictionary.

        :param name: name of the metric
        :param score: score
        :param higher_score_is_better: if higher score is better or not for this metric
        :return:
        """
        self._create_metric_results_if_not_yet_done(name, higher_score_is_better)
        self._dataclass.metric_results[name].train_values.append(score)

        if log_metric:
            # TODO: log metric??
            self.trial.logger.info('{} train: {}'.format(name, score))

    def add_metric_results_validation(
        self, name: str, score: float, higher_score_is_better: bool, log_metric: bool = False
    ):
        """
        Add a validation metric result in the metric results dictionary.

        # TODO: Metric
        # Dataclass argument?

        :param name: name of the metric
        :param score: score
        :param higher_score_is_better: if higher score is better or not for this metric
        :return:
        """
        self._create_metric_results_if_not_yet_done(name, higher_score_is_better)
        self._dataclass.metric_results[name].validation_values.append(score)

        if log_metric:
            self.trial.logger.info('{} validation: {}'.format(name, score))

    def _create_metric_results_if_not_yet_done(self, name, higher_score_is_better):
        if name not in self._dataclass.metric_results:
            self._dataclass.metric_results[name] = MetricResultsDataclass(
                metrix_name=name,
                validation_values=[],
                train_values=[],
                higher_score_is_better=higher_score_is_better,
            )

    def get_train_scores(self, metric_name) -> List[float]:
        return self._dataclass.metric_results[metric_name].train_values

    def get_val_scores(self, metric_name: str) -> List[float]:
        """
        Return the validation scores for the main scoring metric.
        """
        return self._dataclass.metric_results[metric_name].validation_values

    def get_final_validation_score(self, metric_name: str) -> float:
        """
        Return the latest validation score for the main scoring metric.
        """
        return self.get_val_scores(metric_name)[-1]

    def get_best_validation_score(self, metric_name: str) -> float:
        """
        Return the best validation score for the main scoring metric.
        """
        if self.is_higher_score_better(metric_name):
            f = np.max
        else:
            f = np.min

        return f(self.get_val_scores(metric_name))

    def get_n_epochs_to_best_validation_score(self, metric_name: str) -> int:
        """
        Return the number of epochs
        """
        if self.is_higher_score_better(metric_name):
            f = np.argmax
        else:
            f = np.argmin

        return f(self.get_val_scores(metric_name))

    def is_higher_score_better(self, metric_name: str) -> bool:
        """
        Return True if higher scores are better for the main metric.

        :return:
        """
        return self._dataclass.metric_results[metric_name].higher_score_is_better

    def is_new_best_score(self, metric_name: str) -> bool:
        """
        Return True if the latest validation score is the new best score.

        :return:
        """
        if self.get_best_validation_score(metric_name) in self.get_val_scores()[:-1]:
            return False
        return True

    def set_success(self) -> 'TrialSplit':
        """
        Set trial status to success.

        :return: self
        """
        self._dataclass.status = TrialStatus.SUCCESS
        self.save_parent_trial()
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
        self._dataclass.status = TrialStatus.FAILED
        self.error = str(error)
        self.error_traceback = traceback.format_exc()
        self.save_parent_trial()
        return self

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "Trial.from_json({})".format(str(self.to_json()))


class MetricResults(BaseAggregate[None, MetricResultsDataclass]):
    pass
    # TODO: logging at each epoch.
    # TODO: epoch class and epoch location.
    # TODO: this will be used for the Epoch to save the MetricResults with the _acquire_managed_subresource.

    def at_epoch(self, epoch: int, n_epochs: int) -> 'MetricResults':
        """
        Return the metric results for a specific epoch.
        """
        self.epoch: int = epoch
        self.n_epochs: int = n_epochs
        return self

    _ = """
        def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: traceback):
            self.flow.log_error(exc_val)
            raise NotImplementedError("TODO: log MetricResultMetadata?")
    """


aggregate_2_subaggregate = OrderedDict([
    (Root, Client),
    (Client, Project),
    (Project, Trial),
    (Trial, TrialSplit),
    (TrialSplit, MetricResults),
    (MetricResults, MetricResults),
])

aggregate_2_dataclass: OrderedDict[BaseDataclass, str] = OrderedDict([
    (Root, RootDataclass),
    (Client, ClientDataclass),
    (Project, ProjectDataclass),
    (Trial, TrialDataclass),
    (TrialSplit, TrialSplitDataclass),
    (MetricResults, MetricResultsDataclass),
])
