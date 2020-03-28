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

..
    Thanks to Umaneo Technologies Inc. for their contributions to this Machine Learning
    project, visit https://www.umaneo.com/ for more information on Umaneo Technologies Inc.

"""

import datetime
import hashlib
import traceback
from enum import Enum
from typing import Dict, List

from neuraxle.base import BaseStep, ExecutionContext
from neuraxle.hyperparams.space import HyperparameterSamples


class Trial:
    """
    Trial data container.

    .. seealso::
        :class:`AutoML`,
        :class:`HyperparamsRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`DataContainer`
    """

    def __init__(
            self,
            hyperparams: HyperparameterSamples,
            status: 'TRIAL_STATUS' = None,
            error: Exception = None,
            error_traceback: str = None,
            metrics_results: Dict = None,
            pipeline: BaseStep = None,
            cache_folder = None
    ):
        if status is None:
            status = TRIAL_STATUS.PLANNED
        self.cache_folder = cache_folder
        self.status: TRIAL_STATUS = status
        self.error: Exception = error
        self.error_traceback: str = error_traceback
        if metrics_results is None:
            metrics_results = {}
        self.metrics_results: Dict = metrics_results
        self.pipeline: BaseStep = pipeline
        self.hyperparams: HyperparameterSamples = hyperparams

    def set_fitted_pipeline(self, pipeline: BaseStep):
        """
        Set fitted pipeline.

        :param pipeline: fitted pipeline
        :return:
        """
        self.pipeline = pipeline

    def add_metric_results_train(self, name: str, score: float, higher_score_is_better: bool):
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

    def add_metric_results_validation(self, name: str, score: float, higher_score_is_better: bool):
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

    def save_model(self):
        """
        Save fitted model in the trial hash folder.
        """
        hyperparams = self.hyperparams.to_flat_as_dict_primitive()
        trial_hash = self._get_trial_hash(hyperparams)
        self.pipeline.set_name(trial_hash).save(ExecutionContext(self.cache_folder), full_dump=True)

    def set_success(self):
        """
        Set trial status to success.
        """
        self.status = TRIAL_STATUS.SUCCESS

    def set_hyperparams(self, hyperparams: HyperparameterSamples):
        """
        Set trial hyperparams.

        :param hyperparams: trial hyperparams
        :return:
        """
        self.hyperparams = hyperparams

    def set_failed(self, error: Exception):
        """
        Set failed trial with exception.

        :param error: catched exception
        :return:
        """
        self.status = TRIAL_STATUS.FAILED
        self.error = str(error)
        self.error_traceback = traceback.format_exc()

    def is_new_best_score(self):
        """
        Return True if the latest validation score is the new best score.

        :return:
        """
        higher_score_is_better = self.metrics_results['main']['higher_score_is_better']
        validation_values = self.metrics_results['main']['validation_values']
        best_score = validation_values[0]

        for score in validation_values:
            if score > best_score and higher_score_is_better:
                best_score = score

            if score < best_score and not higher_score_is_better:
                best_score = score

        if best_score == validation_values[-1]:
            return True
        return False

    def get_validation_scores(self):
        """
        Return the validation scores for the main scoring metric.

        :return:
        """
        return self.metrics_results['main']['validation_values']

    def get_higher_score_is_better(self):
        """
        Return True if higher scores are better for the main metric.

        :return:
        """
        return self.metrics_results['main']['higher_score_is_better']

    def to_json(self) -> dict:
        """
        Return the trial in a json format.

        :return:
        """
        return {
            'hyperparams': self.hyperparams.to_flat_as_dict_primitive(),
            'status': self.status.value,
            'error': self.error,
            'metric_results': self.metrics_results,
            'error_traceback': self.error_traceback,
        }

    @staticmethod
    def from_json(trial_json) -> 'Trial':
        """
        Create a trial object from json.

        :param trial_json: trial json
        :return:
        """
        return Trial(
            hyperparams=trial_json['hyperparams'],
            status=trial_json['status'],
            error=trial_json['error'],
            metrics_results=trial_json['metric_results'],
            error_traceback=trial_json['error_traceback']
        )

    def _get_trial_hash(self, hp_dict: Dict):
        """
        Hash hyperparams with md5 to create a trial hash.

        :param hp_dict: hyperparams dict
        :return:
        """
        current_hyperparameters_hash = hashlib.md5(str.encode(str(hp_dict))).hexdigest()
        return current_hyperparameters_hash

    def __enter__(self):
        """
        Start trial, and set the trial status to PLANNED.
        """
        self.start_time = datetime.datetime.now()
        self.status = TRIAL_STATUS.PLANNED
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
    SUCCESS = 'success'
    PLANNED = 'planned'


class Trials:
    """
    Data object containing auto ml trials.

    .. seealso::
        :class:`AutoMLSequentialWrapper`,
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
        best_score = None
        best_hyperparams = None

        higher_score_is_better = self.trials[-1].metrics_results['main']['higher_score_is_better']

        for trial in self.trials:
            trial_score = trial.metrics_results['main']['higher_score_is_better']
            if best_score is None or higher_score_is_better == (trial_score > best_score):
                best_score = trial_score
                best_hyperparams = trial.hyperparams

        return best_hyperparams

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