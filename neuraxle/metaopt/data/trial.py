"""
Neuraxle's AutoML Scope Manager Classes
====================================
Trial objects used by AutoML algorithm classes.

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
from typing import Any, Callable, Dict, Iterable, List, Tuple, Type

import numpy as np
from neuraxle.base import BaseStep, ExecutionContext, Flow, TrialStatus
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace, RecursiveDict)
from neuraxle.metaopt.data.trial import (RoundManager, TrialManager,
                                         TrialSplitManager)
from neuraxle.metaopt.data.vanilla import (AutoMLFlow, BaseDataclass,
                                           BaseHyperparameterOptimizer,
                                           ClientDataclass,
                                           HyperparamsRepository,
                                           InMemoryHyperparamsRepository,
                                           MetricResultsDataclass,
                                           ProjectDataclass, RecursiveDict,
                                           RootMetadata, RoundDataclass,
                                           ScopedLocation, SubDataclassT,
                                           TrialDataclass, TrialSplitDataclass)


class BaseScope(ABC):
    def __init__(self, context: ExecutionContext):
        self.context: ExecutionContext = context

    @property
    def flow(self) -> AutoMLFlow:
        return self.context.flow

    @property
    def loc(self) -> ScopedLocation:
        return self.flow.loc

    @property
    def repo(self) -> HyperparamsRepository:
        return self.flow.repo

    def log(self, message: str, level: int = logging.INFO):
        return self.flow.log(message, level)


class ProjectScope(BaseScope):
    def __enter__(self) -> 'ProjectScope':
        return self

    def new_client(self) -> 'ClientScope':
        return ClientScope(self.context)

    def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: traceback):
        self.flow.log_error(exc_val)


class ClientScope(BaseScope):
    def __enter__(self) -> 'ClientScope':
        return self

    def new_round(self, hp_space: HyperparameterSamples) -> 'RoundScope':
        return RoundScope(self.context, hp_space)

    def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: traceback):
        self.flow.log_error(exc_val)


class RoundScope(BaseScope):
    def __init__(self, context: ExecutionContext, hp_space: HyperparameterSpace):
        super().__init__(context)
        self.hp_space: HyperparameterSpace = hp_space

    def __enter__(self) -> 'RoundScope':
        # self.flow.log_
        self.flow.add_file_to_logger(self.repo.get_logger_path(self.loc))
        return self

    def new_hyperparametrized_trial(self, hp_optimizer: BaseHyperparameterOptimizer, continue_loop_on_error: bool) -> 'TrialScope':
        with self.context.flow.synchroneous():
            new_hps: HyperparameterSamples = hp_optimizer.find_next_best_hyperparams(self)
            ts = TrialScope(self.context, new_hps, continue_loop_on_error)
        return ts

        # TODO: def retrain_best_model?

    def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: traceback):
        self.flow.log_error(exc_val)
        flow: AutoMLFlow = self.context.flow
        self.flow.log_best_hps(flow.repo.get_best_hyperparams())
        self.flow.log('Finished round hp search!')
        self.flow._free_logger_files(self.repo.log_path(self.loc))

    def _free_logger_file(self):
        """Remove file handlers from logger to free file lock on Windows."""
        for h in self.logger.handlers:
            if isinstance(h, FileHandler):
                self.logger.removeHandler(h)


class TrialScope(BaseScope):

    def __init__(self, context: ExecutionContext, repo: HyperparamsRepository, hps: HyperparameterSamples, continue_loop_on_error: bool):
        super().__init__(context, repo)

        self.hps: HyperparameterSamples = hps
        self.error_types_to_raise = (
            SystemError, SystemExit, EOFError, KeyboardInterrupt
        ) if continue_loop_on_error else (Exception,)

        self.flow.log_planned(hps)

    def __enter__(self) -> 'TrialScope':
        self.flow.log_start()
        raise NotImplementedError("TODO: ???")

        self.trial.status = TrialStatus.RUNNING
        self.logger.info(
            '\nnew trial: {}'.format(
                json.dumps(self.hyperparams.to_nested_dict(), sort_keys=True, indent=4)))
        self.save_trial()
        return self

    def new_trial_split(self) -> 'TrialSplitScope':
        return TrialSplitScope(self.context, self.repo)

    def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: traceback):
        gc.collect()
        raise NotImplementedError("TODO: log split result with MetricResultMetadata?")

        try:
            if exc_type is None:
                self.trial.end(status=TrialStatus.SUCCESS)
            else:
                # TODO: if ctrl+c, raise KeyboardInterrupt? log aborted or log failed?
                self.flow.log_error(exc_val)
                self.trial.end(status=TrialStatus.FAILED)
                raise exc_val
        finally:
            self.save_trial()
            self._free_logger_file()


class TrialSplitScope(BaseScope):
    def __init__(self, context: ExecutionContext, repo: HyperparamsRepository, n_epochs: int):
        super().__init__(context, repo)
        self.n_epochs = n_epochs

    def __enter__(self):
        """
        Start trial, and set the trial status to PLANNED.
        """
        # TODO: ??
        self.trial_split.status = TrialStatus.RUNNING
        self.save_parent_trial()
        return self

    def new_epoch(self, epoch: int) -> 'EpochScope':
        return EpochScope(self.context, self.repo, epoch, self.n_epochs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop trial, and save end time.

        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        self.end_time = datetime.datetime.now()
        if self.delete_pipeline_on_completion:
            del self.pipeline
        if exc_type is not None:
            self.set_failed(exc_val)
            self.trial_split.end(TrialStatus.FAILED)
            self.save_parent_trial()
            return False

        self.trial_split.end(TrialStatus.SUCCESS)
        self.save_parent_trial()
        return True

    def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: traceback):
        raise NotImplementedError("TODO: log split result with MetricResultMetadata?")

        if exc_val is None:
            self.flow.log_success()
        elif exc_type in self.error_types_to_raise:
            self.flow.log_failure(exc_val)
            return False  # re-raise.
        else:
            self.flow.log_error(exc_val)
            self.flow.log_aborted()
            return True  # don't re-raise.


class EpochScope(BaseScope):
    def __init__(self, context: ExecutionContext, repo: HyperparamsRepository, epoch: int, n_epochs: int):
        super().__init__(context, repo)
        self.epoch: int = epoch
        self.n_epochs: int = n_epochs

    def __enter__(self) -> 'EpochScope':
        self.flow.log_epoch(self.epoch, self.n_epochs)
        return self

    def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: traceback):
        self.flow.log_error(exc_val)
        raise NotImplementedError("TODO: log MetricResultMetadata?")



class TrialManager:
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

    def __init__(
            self,
            trial_dataclass: 'TrialDataclass',
    ):
        self._dataclass: TrialDataclass = trial_dataclass

    def new_validation_split(self, delete_pipeline_on_completion: bool = True) -> 'TrialSplitManager':
        """
        Create a new trial split.
        A trial has one split when the validation splitter function is validation split.
        A trial has one or many split when the validation splitter function is kfold_cross_validation_split.

        :param delete_pipeline_on_completion: bool to delete pipeline on completion or not
        :type pipeline: pipeline to execute
        :return: one trial split
        """
        trial_split: TrialSplitManager = TrialSplitManager(
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
        score = sum(scores) / len(scores)
        return score

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

    def set_success(self) -> 'TrialManager':
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

        if success:
            self.status = TrialStatus.SUCCESS
        else:
            self.status = TrialStatus.FAILED

        self.save_trial()  # TODO?

    def set_failed(self, error: Exception) -> 'TrialManager':
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


class TrialSplitManager:
    """
    One split of a trial.

    .. seealso::
        :class:`AutoML`,
        :class:`HyperparamsRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`DataContainer`
    """

    def __init__(
            self,
            trial: TrialManager,
            trial_split_dataclass: TrialSplitDataclass,
            delete_pipeline_on_completion: bool = True
    ):
        self.trial: TrialManager = trial
        self._dataclass: TrialSplitDataclass = trial_split_dataclass
        self.delete_pipeline_on_completion = delete_pipeline_on_completion

    def get_metric_names(self) -> List[str]:
        return list(self._dataclass.metric_results.keys())

    def save_parent_trial(self) -> 'TrialSplitManager':
        """
        Save parent trial.

        :return: self
        """
        # TODO: repo loc?
        self.trial.save_trial()
        return self

    def save_model(self, label: str):
        """
        Saves the pipeline instance the same way a Trial instance would. Will overwrite
        :return:
        """
        # TODO: this is used in callbacks
        self.trial.save_model(self.pipeline, label)

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

    def set_success(self) -> 'TrialSplitManager':
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

    def set_failed(self, error: Exception) -> 'TrialSplitManager':
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
        s = "Trial.from_json({})".format(str(self.to_json()))
        return s


class RoundManager:
    """
    Data object containing auto ml trials.

    .. seealso::
        :class:`RandomSearch`,
        :class:`HyperparamsRepository`,
        :class:`MetaStepMixin`,
        :class:`BaseStep`
    """

    def __init__(
            self,
            trials: List[TrialManager] = None
    ):
        if trials is None:
            trials = []
        self.trials: List[TrialManager] = trials

    def get_best_hyperparams(self) -> HyperparameterSamples:
        """
        Get best hyperparams from all trials.

        :return:
        """
        if len(self) == 0:
            return HyperparameterSamples()

        return self.get_best_trial().get_hyperparams()

    def get_best_trial(self) -> TrialManager:
        """
        :return: trial with best score from all trials
        """
        best_score, best_trial = None, None

        if len(self) == 0:
            raise Exception('Could not get best trial because there were no successful trial.')

        for trial in self.trials:
            trial_score = trial.get_avg_validation_score()

            if best_score is None or self.trials[-1].is_higher_score_better() == (trial_score > best_score):
                best_score = trial_score
                best_trial = trial

        return best_trial

    def split_good_and_bad_trials(self, quantile_threshold: float, number_of_good_trials_max_cap: int) -> Tuple[
            'RoundManager', 'RoundManager']:
        # TODO: move to tpe class? Seems wrongly located.
        success_trials: RoundManager = self.filter(TrialStatus.SUCCESS)

        # Split trials into good and bad using quantile threshold.
        trials_scores = np.array([trial.get_validation_score() for trial in success_trials])

        trial_sorted_indexes = np.argsort(trials_scores)
        if success_trials.is_higher_score_better():
            trial_sorted_indexes = list(reversed(trial_sorted_indexes))

        # In hyperopt they use this to split, where default_gamma_cap = 25. They clip the max of item they use in the good item.
        # default_gamma_cap is link to the number of recent_trial_at_full_weight also.
        # n_below = min(int(np.ceil(gamma * np.sqrt(len(l_vals)))), gamma_cap)
        n_good = int(min(np.ceil(quantile_threshold * len(trials_scores)), number_of_good_trials_max_cap))

        good_trials_indexes = trial_sorted_indexes[:n_good]
        bad_trials_indexes = trial_sorted_indexes[n_good:]

        good_trials = []
        bad_trials = []
        for trial_index, trial in enumerate(success_trials):
            if trial_index in good_trials_indexes:
                good_trials.append(trial)
            if trial_index in bad_trials_indexes:
                bad_trials.append(trial)

        return RoundManager(trials=good_trials), RoundManager(trials=bad_trials)

    def is_higher_score_better(self, metric_name: str) -> bool:
        """
        Return true if higher score is better.

        :return
        """
        if len(self) == 0:
            return False

        return self.trials[-1].is_higher_score_better(metric_name)

    def append(self, trial: TrialManager):
        """
        Add a new trial.

        :param trial: new trial
        :return:
        """
        self.trials.append(trial)
        # TODO: save?

    def filter(self, status: 'TrialStatus') -> 'RoundManager':
        """
        Get all the trials with the given trial status.

        :param status: trial status
        :return:
        """
        trials = RoundManager()
        for trial in self.trials:
            if trial._dataclass.status == status:
                trials.append(trial)

        return trials

    def get_number_of_split(self):
        if len(self) > 0:
            return len(self[0].validation_splits)
        return 0

    def get_metric_names(self) -> List[str]:
        if len(self) > 0:
            return self[-1]._dataclass.validation_splits[-1].metric_results.keys()
        return []

    def __iter__(self) -> Iterable[TrialManager]:
        return iter(self.trials)

    def __getitem__(self, item: int) -> TrialManager:
        """
        Get trial at the given index.

        :param item: trial index
        :return:
        """
        return self.trials[item]

    def __len__(self) -> int:
        return len(self.trials)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        s = "Trials({})".format(str([str(t) for t in self.trials]))
        return s
