"""
Neuraxle's Trial Classes
====================================
Trial objects used by AutoML algorithm classes.

..
    Copyright 2019, Neuraxio Inc.

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
import logging
import os
import traceback
import json
from enum import Enum
from logging import FileHandler, Logger
from typing import Any, Dict, List, Callable, Iterable, Tuple

import numpy as np

from neuraxle.base import BaseStep, ExecutionContext
from neuraxle.logging.logging import LOGGING_DATETIME_STR_FORMAT
from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.space import HyperparameterSamples, RecursiveDict


class Trial:
    """
    This class is a data structure most often used under :class:`AutoML` to store information about a trial.
    This information is itself managed by the :class:`HyperparameterRepository` class
    and the :class:`TrialRepo` class within the AutoML.
    """

    def __init__(
            self,
            trial_number: int,
            hyperparams: HyperparameterSamples,
            main_metric_name: str,
            status: 'TRIAL_STATUS' = None,
            validation_splits: List['TrialSplit'] = None,
            error: str = None,
            error_traceback: str = None,
            start_time: datetime.datetime = None,
            end_time: datetime.datetime = None,
            log: str = None,
            introspection_data: RecursiveDict = None
    ):
        self.trial_number = trial_number

        if status is None:
            status = TRIAL_STATUS.PLANNED
        if validation_splits is None:
            validation_splits = []

        self.main_metric_name: str = main_metric_name
        self.status: TRIAL_STATUS = status
        self.hyperparams: HyperparameterSamples = hyperparams
        self.validation_splits: List['TrialSplit'] = validation_splits
        self.error_traceback: str = error_traceback
        self.error: str = error
        self.start_time: datetime.datetime = start_time
        self.end_time: datetime.datetime = end_time
        self.log: str = log if log is not None else ""
        self.introspection_data: RecursiveDict = (
            introspection_data if introspection_data is not None else RecursiveDict())

    def to_json_dict(self) -> dict[str, Any]:
        return {
            'trial_number': self.trial_number,
            'status': self.status.value,
            'hyperparams': self.hyperparams.to_flat_dict(),
            'validation_splits': [v.to_json() for v in self.validation_splits],
            'error': self.error,
            'error_traceback': self.error_traceback,
            'start_time': self.start_time.strftime(LOGGING_DATETIME_STR_FORMAT) if self.start_time is not None else '',
            'end_time': self.end_time.strftime(LOGGING_DATETIME_STR_FORMAT) if self.end_time is not None else '',
            'main_metric_name': self.main_metric_name,
            'log': self.log,
            # TODO: move to val splits?
            'introspection_data': self.introspection_data.to_flat_dict()
        }

    @staticmethod
    def from_json_dict(trial_json: Dict) -> 'Trial':
        trial: Trial = Trial(
            trial_number=trial_json["trial_number"],
            main_metric_name=trial_json['main_metric_name'],
            status=TRIAL_STATUS(trial_json['status']),
            hyperparams=HyperparameterSamples(trial_json['hyperparams']),
            error=trial_json['error'],
            error_traceback=trial_json['error_traceback'],
            start_time=datetime.datetime.strptime(trial_json['start_time'], LOGGING_DATETIME_STR_FORMAT),
            end_time=datetime.datetime.strptime(trial_json['start_time'], LOGGING_DATETIME_STR_FORMAT),
            log=trial_json['log'],
            introspection_data=RecursiveDict(trial_json['introspection_data'])
        )

        trial.validation_splits = [
            TrialSplit.from_json(
                trial=trial,
                trial_split_json=validation_split_json
            ) for validation_split_json in trial_json['validation_splits']
        ]

        return trial

    def __getitem__(self, item) -> 'TrialSplit':
        return self.validation_splits[item]


class TrialRepo:
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
        save_trial_function: Callable,
        pipeline: BaseStep = None,
        logger: Logger = None
    ):
        self.save_trial_function: Callable = save_trial_function

        self.pipeline: BaseStep = pipeline

        if logger is None:
            if self.cache_folder is not None:
                logger = self._initialize_logger_with_file()
            else:
                logger = logging.getLogger()
        self.logger: Logger = logger

        self.trial = Trial()  # TODO: do this.

    def save_trial(self) -> 'Trial':
        """
        Update trial with the hyperparams repository.

        :return: The trial
        """
        self.save_trial_function(self.trial)
        return self

    def save_model(self, context: ExecutionContext):
        """
        Save fitted model in the trial's folder given from the trial's context.
        """
        assert self.cache_folder is not None
        self._save_model(self.pipeline, context)

    def _save_model(self, pipeline: BaseStep, context: ExecutionContext):
        hyperparams = self.hyperparams.to_flat_dict()
        trial_hash = self._get_trial_hash(hyperparams)
        pipeline.set_name(trial_hash).save(context, full_dump=True)

    def load_model(self, context: ExecutionContext) -> BaseStep:
        """
        Load model in the trial hash folder.
        """
        trial_hash: str = self._get_trial_hash(self.hyperparams.to_flat_dict)
        return ExecutionContext.load(
            context=context,
            pipeline_name=trial_hash,
        )

    def new_validation_split(self, pipeline: BaseStep, delete_pipeline_on_completion: bool = True) -> 'TrialSplit':
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
            split_number=len(self.validation_splits),
            main_metric_name=self.main_metric_name,
            pipeline=pipeline,
            delete_pipeline_on_completion=delete_pipeline_on_completion
        )
        self.validation_splits.append(trial_split)

        self.save_trial()
        return trial_split

    def set_main_metric_name(self, name: str) -> 'Trial':
        """
        Set trial main metric name.

        :return: self
        """
        self.main_metric_name = name
        return self

    def is_higher_score_better(self) -> bool:
        """
        Return True if higher scores are better for the main metric.

        :return: if higher score is better
        """
        return self.validation_splits[0].is_higher_score_better()

    def get_validation_score(self) -> float:
        """
        Return the best validation score for the main scoring metric.
        Returns the average score for all validation splits.

        :return: validation score
        """
        scores = [
            validation_split.get_best_validation_score()
            for validation_split in self.validation_splits if validation_split.is_success()
        ]

        score = sum(scores) / len(scores)

        return score

    def get_n_epoch_to_best_validation_score(self) -> float:
        """
        Return the best validation score for the main scoring metric.
        Returns the average score for all validation splits.

        :return: validation score
        """
        n_epochs = [
            validation_split.get_n_epochs_to_best_validation_score()
            for validation_split in self.validation_splits if validation_split.is_success()
        ]

        n_epochs = sum(n_epochs) / len(n_epochs)

        return n_epochs

    def set_success(self) -> 'Trial':
        """
        Set trial status to success.

        :return: self
        """
        self.status = TRIAL_STATUS.SUCCESS
        self.save_trial()

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
            self.status = TRIAL_STATUS.SUCCESS
        else:
            self.status = TRIAL_STATUS.FAILED

        self.save_trial()

    def set_failed(self, error: Exception) -> 'Trial':
        """
        Set failed trial with exception.

        :param error: catched exception
        :return: self
        """
        self.status = TRIAL_STATUS.FAILED
        self.error = str(error)
        self.error_traceback = traceback.format_exc()

        self.save_trial()

        return self

    def get_trained_pipeline(self, split_number: int = 0):
        """
        Get trained pipeline inside the validation splits.

        :param split_number: split number to get trained pipeline from
        :return:
        """
        return self.validation_splits[split_number].get_pipeline()

    def _get_trial_hash(self, hp_dict: Dict):
        """
        Hash hyperparams with blake2s to create a trial hash.

        :param hp_dict: hyperparams dict
        :return:
        """
        current_hyperparameters_hash = hashlib.blake2s(str.encode(str(hp_dict))).hexdigest()
        return current_hyperparameters_hash

    def _initialize_logger_with_file(self) -> logging.Logger:

        os.makedirs(self.cache_folder, exist_ok=True)

        logfile_path = os.path.join(self.cache_folder, f"trial_{self.trial_number}.log")
        logger_name = f"trial_{self.trial_number}"
        logger = logging.getLogger(logger_name)
        formatter = logging.Formatter(fmt=LOGGER_FORMAT, datefmt=DATE_FORMAT)
        file_handler = logging.FileHandler(filename=logfile_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def _free_logger_file(self):
        """Remove file handlers from logger to free file lock on Windows."""
        for h in self.logger.handlers:
            if isinstance(h, FileHandler):
                self.logger.removeHandler(h)

    def __enter__(self):
        """
        Start trial, and set the trial status to PLANNED.
        """
        self.start_time = datetime.datetime.now()
        self.status = TRIAL_STATUS.STARTED

        self.logger.info(
            '\nnew trial: {}'.format(
                json.dumps(self.hyperparams.to_nested_dict(), sort_keys=True, indent=4)))

        self.save_trial()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop trial, and save end time.

        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        self.end_time = datetime.datetime.now()
        del self.pipeline
        if exc_type is not None:
            self.set_failed(exc_val)
            self.save_trial()
            raise exc_val

        self.save_trial()
        self._free_logger_file()
        return self


class TrialSplit:
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
            trial: Trial,
            split_number: int,
            main_metric_name: str,
            status: 'TRIAL_STATUS' = None,
            error: Exception = None,
            error_traceback: str = None,
            metrics_results: Dict[str, Any] = None,
            start_time: datetime.datetime = None,
            end_time: datetime.datetime = None,
            pipeline: BaseStep = None,
            delete_pipeline_on_completion: bool = True
    ):
        if status is None:
            status = TRIAL_STATUS.PLANNED

        self.trial: Trial = trial
        self.split_number: int = split_number
        self.status: TRIAL_STATUS = status
        self.error: Exception = error
        self.error_traceback: str = error_traceback
        if metrics_results is None:
            metrics_results = {}
        self.metrics_results: Dict[str, Any] = metrics_results
        self.end_time: datetime.datetime = end_time
        self.start_time: datetime.datetime = start_time
        self.pipeline: BaseStep = pipeline
        self.main_metric_name: str = main_metric_name
        self.delete_pipeline_on_completion = delete_pipeline_on_completion

    def get_metric_names(self) -> List[str]:
        return list(self.metrics_results.keys())

    def save_parent_trial(self) -> 'TrialSplit':
        """
        Save parent trial.

        :return: self
        """
        self.trial.save_trial()
        return self

    def save_model(self, label: str):
        """
        Saves the pipeline instance the same way a Trial instance would. Will overwrite
        :return:
        """
        self.trial._save_model(self.pipeline, label)

    def fit_trial_split(self, train_data_container: DataContainer, context: ExecutionContext) -> 'TrialSplit':
        """
        Fit the trial split pipeline with the training data container.

        :param train_data_container: training data container
        :param context: execution context
        :return: trial split with its fitted pipeline.
        """
        self.pipeline.set_train(True)
        self.pipeline = self.pipeline.handle_fit(train_data_container, context)
        return self

    def predict_with_pipeline(self, data_container: DataContainer, context: ExecutionContext) -> 'DataContainer':
        """
        Predict data with the fitted trial split pipeline.

        :param data_container: data container to predict
        :param context: execution context
        :return: predicted data container
        """
        return self.pipeline.handle_predict(data_container, context)

    def set_main_metric_name(self, name: str) -> 'TrialSplit':
        """
        Set main metric name.

        :param name: main metric name.
        :return: self
        """
        self.main_metric_name = name

        return self

    def add_metric_results_train(self, name: str, score: float, higher_score_is_better: bool, log_metric: bool = False):
        """
        Add a train metric result in the metric results dictionary.

        :param name: name of the metric
        :param score: score
        :param higher_score_is_better: if higher score is better or not for this metric
        :return:
        """
        if name not in self.metrics_results:
            self.metrics_results[name] = {
                'train_values': [],
                'validation_values': [],
                'higher_score_is_better': higher_score_is_better
            }

        self.metrics_results[name]['train_values'].append(score)

        if log_metric:
            self.trial.logger.info('{} train: {}'.format(name, score))

    def add_metric_results_validation(self, name: str, score: float, higher_score_is_better: bool, log_metric: bool = False):
        """
        Add a validation metric result in the metric results dictionary.

        :param name: name of the metric
        :param score: score
        :param higher_score_is_better: if higher score is better or not for this metric
        :return:
        """
        if name not in self.metrics_results:
            self.metrics_results[name] = {
                'train_values': [],
                'validation_values': [],
                'higher_score_is_better': higher_score_is_better
            }

        self.metrics_results[name]['validation_values'].append(score)

        if log_metric:
            self.trial.logger.info('{} validation: {}'.format(name, score))

    def get_validation_scores(self):
        """
        Return the validation scores for the main scoring metric.
        """
        return self.metrics_results[self.main_metric_name]['validation_values']

    def get_final_validation_score(self):
        """
        Return the latest validation score for the main scoring metric.
        """
        return self.metrics_results[self.main_metric_name]['validation_values'][-1]

    def get_best_validation_score(self):
        """
        Return the best validation score for the main scoring metric.
        """
        higher_score_is_better = self.metrics_results[self.main_metric_name]["higher_score_is_better"]
        if higher_score_is_better is True:
            f = np.max
        elif higher_score_is_better is False:
            f = np.min

        return f(self.metrics_results[self.main_metric_name]['validation_values'])

    def get_n_epochs_to_best_validation_score(self):
        """
        Return the number of epochs
        """
        higher_score_is_better = self.metrics_results[self.main_metric_name]["higher_score_is_better"]
        if higher_score_is_better is True:
            f = np.argmax
        elif higher_score_is_better is False:
            f = np.argmin

        return f(self.metrics_results[self.main_metric_name]['validation_values'])

    def get_pipeline(self):
        """
        Return the trained pipeline.
        """
        return self.pipeline

    def is_higher_score_better(self) -> bool:
        """
        Return True if higher scores are better for the main metric.

        :return:
        """
        return self.metrics_results[self.main_metric_name]['higher_score_is_better']

    def is_new_best_score(self):
        """
        Return True if the latest validation score is the new best score.

        :return:
        """
        higher_score_is_better = self.metrics_results[self.main_metric_name]['higher_score_is_better']
        validation_values = self.metrics_results[self.main_metric_name]['validation_values']
        best_score = validation_values[0]

        for score in validation_values:
            if score > best_score and higher_score_is_better:
                best_score = score
            elif score < best_score and not higher_score_is_better:
                best_score = score

        if best_score == validation_values[-1]:
            return True
        return False

    def get_metric_validation_results(self, metric_name):
        return self.metrics_results[metric_name]['validation_values']

    def get_metric_train_results(self, metric_name):
        return self.metrics_results[metric_name]['train_values']

    def to_json(self) -> dict:
        """
        Return the trial in a json format.

        :return:
        """
        return {
            'status': self.status.value,
            'error': self.error,
            'metric_results': self.metrics_results,
            'error_traceback': self.error_traceback,
            'start_time': self.start_time.strftime(LOGGING_DATETIME_STR_FORMAT) if self.start_time is not None else '',
            'end_time': self.end_time.strftime(LOGGING_DATETIME_STR_FORMAT) if self.end_time is not None else '',
            'split_number': self.split_number,
            'main_metric_name': self.main_metric_name
        }

    @staticmethod
    def from_json(trial: 'Trial', trial_split_json: Dict) -> 'TrialSplit':
        """
        Create a trial split object from json.

        :param trial: parent trial
        :param trial_split_json: trial json
        :return:
        """
        return TrialSplit(
            trial=trial,
            status=TRIAL_STATUS(trial_split_json['status']),
            error=trial_split_json['error'],
            error_traceback=trial_split_json['error_traceback'],
            metrics_results=trial_split_json['metric_results'],
            start_time=datetime.datetime.strptime(trial_split_json['start_time'], LOGGING_DATETIME_STR_FORMAT),
            end_time=datetime.datetime.strptime(trial_split_json['end_time'], LOGGING_DATETIME_STR_FORMAT),
            split_number=trial_split_json['split_number'],
            main_metric_name=trial_split_json['main_metric_name']
        )

    def set_success(self) -> 'TrialSplit':
        """
        Set trial status to success.

        :return: self
        """
        self.status = TRIAL_STATUS.SUCCESS
        self.save_parent_trial()
        return self

    def is_success(self):
        """
        Set trial status to success.
        """
        return self.status == TRIAL_STATUS.SUCCESS

    def set_failed(self, error: Exception) -> 'TrialSplit':
        """
        Set failed trial with exception.

        :param error: catched exception
        :return: self
        """
        self.status = TRIAL_STATUS.FAILED
        self.error = str(error)
        self.error_traceback = traceback.format_exc()
        self.save_parent_trial()
        return self

    def __enter__(self):
        """
        Start trial, and set the trial status to PLANNED.
        """
        self.start_time = datetime.datetime.now()
        self.status = TRIAL_STATUS.STARTED
        self.save_parent_trial()
        return self

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
            raise exc_val

        self.save_parent_trial()
        return self

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = "Trial.from_json({})".format(str(self.to_json()))
        return s


class TRIAL_STATUS(Enum):
    """
    Trial status.
    """
    FAILED = 'failed'
    STARTED = 'started'
    SUCCESS = 'success'
    PLANNED = 'planned'


class Trials:
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
            trials: List[Trial] = None
    ):
        if trials is None:
            trials = []
        self.trials: List[Trial] = trials

    def get_best_hyperparams(self) -> HyperparameterSamples:
        """
        Get best hyperparams from all trials.

        :return:
        """
        if len(self) == 0:
            raise Exception('Could not get best hyperparams because there were no successful trial.')

        return self.get_best_trial().hyperparams

    def get_best_trial(self) -> Trial:
        """
        :return: trial with best score from all trials
        """
        best_score, best_trial = None, None

        if len(self) == 0:
            raise Exception('Could not get best trial because there were no successful trial.')

        for trial in self.trials:
            trial_score = trial.get_validation_score()
            if best_score is None or self.trials[-1].is_higher_score_better() == (trial_score > best_score):
                best_score = trial_score
                best_trial = trial

        return best_trial

    def split_good_and_bad_trials(self, quantile_threshold: float, number_of_good_trials_max_cap: int) -> Tuple[
            'Trials', 'Trials']:
        success_trials: Trials = self.filter(TRIAL_STATUS.SUCCESS)

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

        return Trials(trials=good_trials), Trials(trials=bad_trials)

    def is_higher_score_better(self) -> bool:
        """
        Return true if higher score is better.

        :return
        """
        if len(self) == 0:
            return False

        return self.trials[-1].is_higher_score_better()

    def append(self, trial: Trial):
        """
        Add a new trial.

        :param trial: new trial
        :return:
        """
        self.trials.append(trial)

    def filter(self, status: 'TRIAL_STATUS') -> 'Trials':
        """
        Get all the trials with the given trial status.

        :param status: trial status
        :return:
        """
        trials = Trials()
        for trial in self.trials:
            if trial.status == status:
                trials.append(trial)

        return trials

    def get_number_of_split(self):
        if len(self) > 0:
            return len(self[0].validation_splits)
        return 0

    def get_metric_names(self) -> List[str]:
        if len(self) > 0:
            return self[0].validation_splits[0].get_metric_names()
        return []

    def __iter__(self) -> Iterable[Trial]:
        return iter(self.trials)

    def __getitem__(self, item):
        """
        Get trial at the given index.

        :param item: trial index
        :return:
        """
        return self.trials[item]

    def __len__(self):
        return len(self.trials)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = "Trials({})".format(str([str(t) for t in self.trials]))
        return s
