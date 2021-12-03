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
import hashlib
import json
import logging
import os
import traceback
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from json.encoder import JSONEncoder
from logging import FileHandler, Logger
from typing import (Any, Callable, Dict, Generic, Iterable, List, Optional,
                    Tuple, Type, TypeVar, Union)

import numpy as np
from neuraxle.base import (BaseStep, ExecutionContext, Flow, TrialStatus,
                           synchroneous_flow_method)
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace, RecursiveDict)
from neuraxle.metaopt.data.vanilla import (AutoMLContext, AutoMLFlow,
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
                                           TrialSplitDataclass)

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
        _subklass: Type[SubAggregateT] = self.__orig_class__.__args__[0]
        return _subklass

    @property
    def subdataclass(self) -> Type[SubDataclassT]:
        _subdataklass: Type[SubAggregateT] = self.__orig_class__.__args__[1]
        return _subdataklass

    def refresh(self, deep: bool = True):
        with self.context.lock():
            self._dataclass = self.repo.get(self.context.location, deep=deep)
        self.is_deep = deep

    def save(self, deep: bool = True):
        if deep and deep != self.is_deep:
            raise ValueError(
                f"Cannot save {str(self)} with self.is_deep=False when self "
                f"is not already deep. You might want to use self.refresh(deep=True) at "
                f"some point to refresh self before saving deeply then.")

        with self.context.lock():
            self.repo.save(self.loc, self._dataclass, deep=deep)

    def save_subaggregate(self, subagg: SubAggregateT, deep=False):
        self._dataclass.store(subagg._dataclass)
        with self.context.lock():
            self.save(deep=False)
            subagg.save(deep=deep)

    @contextmanager
    def managed_subresource(self, *args, **kwds) -> SubAggregateT:
        resource = None
        e: Exception = None
        try:
            resource = self._acquire_managed_subresource(*args, **kwds)
            yield resource
        except Exception as exc:
            e = exc
        finally:
            return self._release_sub_resource(resource, e)

    @abstractmethod
    def _acquire_managed_subresource(self, *args, **kwds) -> SubAggregateT:
        """
        Acquire a new subaggregate that is managed such that it is deep saved at the
        beginning.
        """
        # Create subaggregate:
        with self.context.lock:
            self.refresh(False)
            subdataclass: SubDataclassT = self.repo.get(*args, **kwds)
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

    @staticmethod
    def from_repo(context: AutoMLContext) -> 'Root':
        _dataclass: RootDataclass = context.repo.get(ScopedLocation())
        return Root(_dataclass, context)

    def _acquire_managed_subresource(self, project_name: str) -> 'Project':
        super()._acquire_managed_subresource(project_name)


class Project(BaseAggregate['Client', ProjectDataclass]):

    def _acquire_managed_subresource(self, client_name: str) -> 'Project':
        client_loc = self.loc.with_id(client_name)
        super()._acquire_managed_subresource(client_loc)


class Client(BaseAggregate['Round', ClientDataclass]):

    def _acquire_managed_subresource(self, hps: HyperparameterSpace, new_round=True) -> 'Project':
        with self.context.lock:
            self.refresh(False)

            # Get new round loc:
            round_id: int = self._dataclass.get_next_i()
            if not new_round:
                # Try get last round:
                round_id = max(0, round_id - 1)
            round_loc = self.loc.with_id(round_id)

            # Get round to return:
            _round_dataclass: RoundDataclass = self.repo.get(round_loc)
            subagg: Round = Round(_round_dataclass, self.context).with_optimizer(hps)

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

    @property
    def trials(self) -> List['Trial']:
        return [Trial(t, self.context) for t in self._dataclass.get_sublocation()]

    def _acquire_managed_subresource(self, new_trial=True) -> 'Round':
        with self.context.lock:
            self.refresh(False)
            self.context.flow.add_file_handler_to_logger(
                self.repo.get_logger_path(self.loc))

            # Get new trial loc:
            trial_id: int = self._dataclass.get_next_i()
            if not new_trial:
                # Try get last trial
                trial_id = max(0, trial_id - 1)
            trial_loc = self.loc.with_id(trial_id)

            # Get trial to return:
            _trial_dataclass: TrialDataclass = self.repo.get(trial_loc)
            new_hps: HyperparameterSamples = self.hp_optimizer.find_next_best_hyperparams(self)
            _trial_dataclass.hyperparams(new_hps)
            self.context.flow.log_planned(new_hps)
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

        for trial in self.trials:
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
                self.trials[-1].is_higher_score_better(metric_name)
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
        _round: Round = self.copy().without_context()
        _trials: List[TrialDataclass] = [
            sdc
            for sdc in self._dataclass.trials
            if sdc.get_status() == status
        ]
        _round._dataclass.trials = _trials

        return _round

    def copy(self) -> 'Round':
        return Round(copy.deepcopy(self._dataclass), self.context.copy())

    def get_number_of_split(self):
        if len(self) > 0:
            return len(self[0].validation_splits)
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

    def _acquire_managed_subresource(
        self, continue_loop_on_error: bool, delete_pipeline_on_completion: bool
    ) -> SubAggregateT:

        self.error_types_to_raise = (
            SystemError, SystemExit, EOFError, KeyboardInterrupt
        ) if continue_loop_on_error else (Exception,)

        with self.context.lock:
            self.refresh(False)

            # TODO: check if we need to delete pipeline here?
            # TODO: active record design pattern such that the dataobjects are the ones containing the repo and saving themselves / finding themselves. Clean Code P.101.
            #     Active record should probably have a weakref of the repo since it doesn<t want to save the repo into itself.
            #     Active record should have its own location integrated into itself!!! When creating a subdataclass, pass the parent into ctor to have a weakref of the parent as well. Pickling to remove refs.
            #     Active record<s loc should not be known to the callers.

            #     TODO: FIND A WAY TO SAVE THE REPO INTO THE DATACLASS. And hide it from the rest. Where does the flow goes as well?

            # Get new split loc:
            split_id: int = self._dataclass.get_next_i()
            split_id = max(0, split_id - 1)
            split_loc = self.loc.with_id(split_id)

            # Get split to return:
            _split_dataclass: TrialSplitDataclass = self.repo.get(split_loc)
            subagg: TrialSplit = TrialSplit(_split_dataclass, self.context)

            self.save_subaggregate(subagg, deep=True)
            return subagg

        # return super()._acquire_managed_subresource(*args, **kwds)

        trial_split: TrialSplit = TrialSplit(_trial_dataclass, self.context)

        # trial_split_dataclass=TrialSplitDataclass(split_number=len(self.validation_splits))
        self.validation_splits.append(trial_split)

        self.save_trial()  # TODO: remove this line or what?
        return trial_split

    def _release_managed_subresource(self, resource: 'TrialSplit', e: Exception = None) -> Optional[Exception]:

        with self.context.lock:
            self.refresh(False)
            
            if e is not None:
                if e.__class__ in self.error_types_to_raise:
                    self.flow.log_failure(e)
                    self.save_trial()
                    return e

            self.save(False)
        return False

    def new_validation_split(self, delete_pipeline_on_completion: bool = True) -> 'TrialSplit':
        """
        Create a new trial split.
        A trial has one split when the validation splitter function is validation split.
        A trial has one or many split when the validation splitter function is kfold_cross_validation_split.

        :param delete_pipeline_on_completion: bool to delete pipeline on completion or not
        :type pipeline: pipeline to execute
        :return: one trial split
        """
        trial_split: TrialSplit = TrialSplit(
            trial=self,
            trial_split_dataclass=TrialSplitDataclass(
                split_number=len(self.validation_splits)
            ),
            delete_pipeline_on_completion=delete_pipeline_on_completion
        )
        self.validation_splits.append(trial_split)

        self.save_trial()  # TODO: remove this line or what?
        return trial_split

    def get_avg_validation_score(self, metric_name: str) -> float:
        """
        Returns the average score for all validation splits's
        best validation score for the specified scoring metric.

        :return: validation score
        """
        scores = [
            val_split.get_best_validation_score(metric_name)
            for val_split in self.validation_splits
            if val_split.is_success()
        ]
        return sum(scores) / len(scores)

    def get_avg_n_epoch_to_best_validation_score(self, metric_name: str) -> float:
        # TODO: use in flow.log_results:
        n_epochs = [
            val_split.get_n_epochs_to_best_validation_score(metric_name)
            for val_split in self.validation_splits if val_split.is_success()
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
        for validation_split in self.validation_splits:
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

    @property
    def metric_results(self) -> Dict[str, 'MetricResults']:
        # TODO: use this.
        return [MetricResults(mr) for mr in self._dataclass]

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
        self.status = TrialStatus.SUCCESS
        self.save_parent_trial()
        return self

    def is_success(self):
        """
        Set trial status to success.
        """
        return self.status == TrialStatus.SUCCESS

    def set_failed(self, error: Exception) -> 'TrialSplit':
        """
        Set failed trial with exception.

        :param error: catched exception
        :return: self
        """
        self.status = TrialStatus.FAILED
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
