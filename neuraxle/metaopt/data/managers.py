

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
from neuraxle.base import BaseStep, ExecutionContext, Flow, TrialStatus
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
                                           RootMetadata, RoundDataclass,
                                           ScopedLocation, ScopedLocationAttr,
                                           SubDataclassT, TrialDataclass,
                                           TrialSplitDataclass)


class _DataclassDomainAggregate(Generic[SubDataclassT]):

    def __init__(self, _dataclass: SubDataclassT, context: AutoMLContext):
        self._dataclass: SubDataclassT = _dataclass
        self.context: AutoMLContext = context

    @property
    def flow(self) -> AutoMLFlow:
        return self.context.flow

    @property
    def loc(self) -> ScopedLocation:
        return self.context.loc

    @property
    def repo(self) -> HyperparamsRepository:
        return self.context.repo


class CanCreateSubManagersMixin:
    def __init__(self):
        assert isinstance(self, _DataclassDomainAggregate), (
            "CanCreateSubManagersMixin can only be used with a "
            "_DataclassDomainAggregate concrete class."
        )

    @abstractmethod
    def create_sub_managers(self) -> Dict[str, '_DataclassDomainAggregate']:
        # TODO: revise this.
        self.repo.create_new(self.loc)


class CanRetrieveSubManagersMixin:
    def __init__(self):
        assert isinstance(self, _DataclassDomainAggregate), (
            "CanCreateSubManagersMixin can only be used with a "
            "_DataclassDomainAggregate concrete class."
        )

    @abstractmethod
    def retrieve_sub_managers(self, loc: ScopedLocation) -> Dict[str, '_DataclassDomainAggregate']:
        # TODO: revise this.
        self.repo.get(self.loc)


class ProjectManager(_DataclassDomainAggregate[ProjectDataclass]):
    pass


class ClientManager(_DataclassDomainAggregate[ClientDataclass]):
    pass


class RoundManager(_DataclassDomainAggregate[RoundDataclass]):

    @property
    def trials(self) -> List['TrialManager']:
        return [TrialManager(t) for t in self._dataclass.trials]

    def get_best_hyperparams(self) -> HyperparameterSamples:
        """
        Get best hyperparams from all trials.

        :return:
        """
        if len(self) == 0:
            return HyperparameterSamples()

        return self.get_best_trial().get_hyperparams()

    def get_best_trial(self) -> 'TrialManager':
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

    def is_higher_score_better(self, metric_name: str) -> bool:
        """
        Return true if higher score is better.

        :return
        """
        if len(self) == 0:
            return False

        return self.trials[-1].is_higher_score_better(metric_name)

    def append(self, trial: 'TrialManager'):
        """
        Add a new trial.

        :param trial: new trial
        :return:
        """
        self.trials.append(trial)
        # TODO: save?

    def filter(self, status: TrialStatus) -> 'RoundManager':
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

    def __iter__(self) -> Iterable['TrialManager']:
        return iter(self.trials)

    def __getitem__(self, item: int) -> 'TrialManager':
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
        return "Trials({})".format(str([str(t) for t in self.trials]))


class TrialManager(_DataclassDomainAggregate[TrialDataclass]):
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

        self.status = TrialStatus.SUCCESS if success else TrialStatus.FAILED

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


class TrialSplitManager(_DataclassDomainAggregate[TrialSplitDataclass]):
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
    def metric_results(self) -> Dict[str, 'MetriMetricResultsManager']:
        # TODO: use this.
        return [MetriMetricResultsManager(mr) for mr in self._dataclass]

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
        return "Trial.from_json({})".format(str(self.to_json()))


class MetriMetricResultsManager(_DataclassDomainAggregate[MetricResultsDataclass]):
    pass
