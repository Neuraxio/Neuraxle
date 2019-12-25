"""
Neuraxle's Automatic Machine Learning Classes
==================================================
All steps, and abstractions needed to build Automatic Machine Learning algorithms in Neuraxle.

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
import copy
import glob
import hashlib
import json
import os
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Callable

from neuraxle.base import MetaStepMixin, BaseStep, NonTransformableMixin, ExecutionContext
from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace
from neuraxle.metaopt.random import BaseCrossValidationWrapper, KFoldCrossValidationWrapper


class HyperparamsRepository(ABC):
    """
    Hyperparams repository that saves hyperparams, and scores for every AutoML trial.

    .. seealso::
        :class:`AutoMLSequentialWrapper`
        :class:`AutoMLAlgorithm`,
        :class:`BaseValidation`,
        :class:`RandomSearchBaseAutoMLStrategy`,
        :class:`HyperparameterSpace`,
        :class:`HyperparameterSamples`
    """

    @abstractmethod
    def create_new_trial(self, hyperparams: HyperparameterSamples):
        """
        Create new hyperperams trial.

        :param hyperparams: hyperparams
        :type hyperparams: HyperparameterSamples
        :return:
        """
        pass

    @abstractmethod
    def load_all_trials(self, status: 'TRIAL_STATUS') -> 'Trials':
        """
        Load all hyperparameter trials with their corresponding score.

        :return: (hyperparams, scores)
        """
        pass

    @abstractmethod
    def save_score_for_success_trial(self, hyperparams: HyperparameterSamples, score: float):
        """
        Save hyperparams, and score for a successful trial.

        :return: (hyperparams, scores)
        """
        pass

    @abstractmethod
    def save_failure_for_trial(self, hyperparams: HyperparameterSamples, exception: Exception):
        """
        Save hyperparams, and score for a failed trial.

        :return: (hyperparams, scores)
        """
        pass

    def _get_trial_hash(self, hp_dict):
        current_hyperparameters_hash = hashlib.md5(str.encode(str(hp_dict))).hexdigest()
        return current_hyperparameters_hash


class InMemoryHyperparamsRepository(HyperparamsRepository):
    """
    In memory hyperparams repository that can print information about trials.
    Useful for debugging.

    Example usage :

    .. code-block:: python

        InMemoryHyperparamsRepository(
            print_new_trial=True,
            print_success_trial=True,
            print_exception=True
        )


    .. seealso::
        :class:`HyperparamsRepository`,
        :class:`HyperparameterSamples`,
        :class:`AutoMLSequentialWrapper`
    """

    def __init__(self, print_new_trial=True, print_success_trial=True, print_exception=True, print_func: Callable = None):
        HyperparamsRepository.__init__(self)
        if print_func is None:
           print_func = print
        self.print_func = print_func

        self.trials = Trials()
        self.print_new_trial = print_new_trial
        self.print_success_trial = print_success_trial
        self.print_exception = print_exception

    def create_new_trial(self, hyperparams: HyperparameterSamples):
        if self.print_new_trial:
            self.print_func('new trial:\n{}'.format(json.dumps(hyperparams.to_nested_dict(), sort_keys=True, indent=4)))

    def load_all_trials(self, status: 'TRIAL_STATUS' = None) -> 'Trials':
        return self.trials.filter(status)

    def save_score_for_success_trial(self, hyperparams: HyperparameterSamples, score: float):
        self.trials.append(Trial(hyperparams, score, TRIAL_STATUS.SUCCESS))

        if self.print_success_trial:
            self.print_func('score: {}'.format(score))
            self.print_func('hyperparams:\n{}'.format(json.dumps(hyperparams.to_nested_dict(), sort_keys=True, indent=4)))

    def save_failure_for_trial(self, hyperparams: HyperparameterSamples, exception: Exception):
        if self.print_exception:
            self.print_func(exception)
            traceback_str = str(traceback.format_tb(exception.__traceback__))
            self.print_func(traceback_str)


class HyperparamsJSONRepository(HyperparamsRepository):
    """
    Hyperparams repository that saves json files for every AutoML trial.

    .. seealso::
        :class:`HyperparamsRepository`,
        :class:`HyperparameterSamples`,
        :class:`AutoMLSequentialWrapper`
    """

    def __init__(self, folder):
        self.cache_folder = folder

    def create_new_trial(self, hyperparams: HyperparameterSamples):
        """
        Create new hyperperams trial json file.

        :param hyperparams: hyperparams
        :type hyperparams: HyperparameterSamples
        :return:
        """
        self._create_trial_json(hyperparams)

    def load_all_trials(self, status: 'TRIAL_STATUS' = None) -> 'Trials':
        """
        Load all hyperparameter trials with their corresponding score.
        Reads all the saved trial json files.

        :return: (hyperparams, scores)
        """
        trials = []

        for base_path in glob.glob(os.path.join(self.cache_folder, '*.json')):
            with open(base_path) as f:
                trial_json = json.load(f)

            if status is None or trial_json['status'] == status.value:
                trials.append(Trial.from_json(trial_json))

        return Trials(trials)

    def save_score_for_success_trial(self, hyperparams: HyperparameterSamples, score: float):
        """
        Save hyperparams, and score for a successful trial.

        :return: (hyperparams, scores)
        """
        self._save_successful_trial_json(hyperparams, score)

    def save_failure_for_trial(self, hyperparams: HyperparameterSamples, exception: Exception):
        """
        Save hyperparams, and score for a failed trial.

        :return: (hyperparams, scores)
        """
        self._save_failed_trial_json(hyperparams, exception)

    def _create_trial_json(self, hyperparams):
        """
        Save new trial json file.

        :return: (hyperparams, scores)
        """
        hp_dict = hyperparams.to_flat_as_dict_primitive()
        current_hyperparameters_hash = self._get_trial_hash(hp_dict)

        with open(os.path.join(self._get_new_trial_json_path(current_hyperparameters_hash)), 'w+') as outfile:
            json.dump({
                'hyperparams': hp_dict,
                'score': None,
                'status': TRIAL_STATUS.PLANNED.value
            }, outfile)

    def _save_successful_trial_json(self, hyperparams, score):
        """
        Save trial json file.

        :return: (hyperparams, scores)
        """
        hp_dict = hyperparams.to_flat_as_dict_primitive()
        current_hyperparameters_hash = self._get_trial_hash(hp_dict)
        self._remove_new_trial_json(current_hyperparameters_hash)

        with open(os.path.join(self.cache_folder,
                               str(float(score)).replace('.', ',') + "_" + current_hyperparameters_hash) + '.json',
                  'w+') as outfile:
            json.dump({
                'hyperparams': hp_dict,
                'score': score,
                'status': TRIAL_STATUS.SUCCESS.value
            }, outfile)

    def _remove_new_trial_json(self, current_hyperparameters_hash):
        new_trial_json = self._get_new_trial_json_path(current_hyperparameters_hash)
        if os.path.exists(new_trial_json):
            os.remove(new_trial_json)

    def _get_new_trial_json_path(self, current_hyperparameters_hash):
        return os.path.join(self.cache_folder, "NEW_" + current_hyperparameters_hash) + '.json'

    def _save_failed_trial_json(self, hyperparams, exception):
        """
        Save trial json file.

        :return: (hyperparams, scores)
        """
        hp_dict = hyperparams.to_flat_as_dict_primitive()
        current_hyperparameters_hash = self._get_trial_hash(hp_dict)
        self._remove_new_trial_json(current_hyperparameters_hash)

        with open(os.path.join(self.cache_folder, 'FAILED_' + current_hyperparameters_hash) + '.json', 'w+') as outfile:
            json.dump({
                'hyperparams': hp_dict,
                'score': None,
                'status': TRIAL_STATUS.FAILED.value,
                'exception': str(exception)
            }, outfile)


class BaseHyperparameterOptimizer(ABC):
    @abstractmethod
    def find_next_best_hyperparams(self, auto_ml_container: 'AutoMLContainer') -> HyperparameterSamples:
        """
        Find the next best hyperparams using previous trials.

        :param auto_ml_container: trials data container
        :type auto_ml_container: Trials
        :return: next best hyperparams
        :rtype: HyperparameterSamples
        """
        raise NotImplementedError()


class AutoMLAlgorithm(MetaStepMixin, BaseStep):
    """
    Pipeline step that executes Automatic Machine Learning strategy.
    It uses an hyperparameter optimizer of type :class:`BaseHyperparameterOptimizer` to find the next best hyperparams.
    It uses a validation technique of type :class:`BaseCrossValidationWrapper` to calculate the score.

    Please refer to :class:`AutoMLSequentialWrapper` for a usage example.

    .. seealso::
        :class:`BaseCrossValidationWrapper`,
        :class:`HyperparameterSamples`,
        :class:`BaseHyperparameterOptimizer`,
        :class:`BaseCrossValidationWrapper`,
        :class:`TrialsContainer`,
        :class:`HyperparameterSpace`,
        :class:`MetaStepMixin`,
        :class:`BaseStep`
    """

    def __init__(
            self,
            hyperparameter_optimizer: BaseHyperparameterOptimizer,
            validation_technique: BaseCrossValidationWrapper = None,
            higher_score_is_better=True
    ):
        MetaStepMixin.__init__(self, None)
        BaseStep.__init__(self)

        if validation_technique is None:
            validation_technique = KFoldCrossValidationWrapper()
        self.validation_technique = validation_technique
        self.higher_score_is_better = higher_score_is_better
        self.hyperparameter_optimizer = hyperparameter_optimizer

    def find_next_best_hyperparams(self, auto_ml_container: 'AutoMLContainer') -> HyperparameterSamples:
        """
        Find the next best hyperparams using previous trials.

        :param auto_ml_container: trials data container
        :type auto_ml_container: Trials
        :return: next best hyperparams
        :rtype: HyperparameterSamples
        """
        return self.hyperparameter_optimizer.find_next_best_hyperparams(auto_ml_container)

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('AutoMLAlgorithm', float):
        """
        Fit cross validation with wrapped step, and return the score.

        :param data_inputs: data inputs
        :param expected_outputs: expected outputs to fit on
        :return: self, step score
        :rtype: (AutoMLAlgorithm, float)
        """
        step: BaseStep = copy.deepcopy(self.wrapped)
        step: BaseCrossValidationWrapper = copy.copy(self.validation_technique).set_step(step)
        step = step.fit(data_inputs, expected_outputs)
        score = step.get_score()

        return self, score

    def get_best_hyperparams(self, trials: 'Trials') -> HyperparameterSamples:
        """
        Get the best hyperparams from all previous trials.

        :return: best hyperparams
        :rtype: HyperparameterSamples
        """
        return trials.get_best_hyperparams(higher_score_is_better=self.higher_score_is_better)

    def fit(self, data_inputs, expected_outputs=None):
        return self

    def transform(self, data_inputs, expected_outputs=None):
        return data_inputs


class TRIAL_STATUS(Enum):
    FAILED = 'failed'
    SUCCESS = 'success'
    PLANNED = 'planned'


class AutoMLContainer:
    """
    Data object for auto ml.

    .. seealso::
        :class:`AutoMLSequentialWrapper`,
        :class:`RandomSearch`,
        :class:`HyperparamsRepository`,
        :class:`MetaStepMixin`,
        :class:`BaseStep`
    """

    def __init__(
            self,
            trials: 'Trials',
            hyperparameter_space: HyperparameterSpace,
            n_iters: int,
            trial_number: int
    ):
        self.trials = trials
        self.hyperparameter_space = hyperparameter_space
        self.n_iters = n_iters
        self.trial_number = trial_number


class Trial:
    def __init__(self, hyperparams: HyperparameterSamples, score: float, status: TRIAL_STATUS):
        self.hyperparams = hyperparams
        self.score = score
        self.status = status

    def to_json(self) -> dict:
        return {
            'hyperparams': self.hyperparams,
            'score': self.score,
            'status': self.status
        }

    @staticmethod
    def from_json(trial_json) -> 'Trial':
        return Trial(
            hyperparams=trial_json['hyperparams'],
            score=trial_json['score'],
            status=trial_json['status']
        )


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

    def get_best_hyperparams(self, higher_score_is_better: bool) -> HyperparameterSamples:
        best_score = None
        best_hyperparams = None

        for trial in self.trials:
            if best_score is None or higher_score_is_better == (trial.score > best_score):
                best_score = trial.score
                best_hyperparams = trial.hyperparams

        return best_hyperparams

    def append(self, trial: Trial):
        self.trials.append(trial)

    def filter(self, status: TRIAL_STATUS) -> 'Trials':
        trials = Trials()
        for trial in self.trials:
            if trial.status == status:
                trials.append(trial)

        return trials

    def __getitem__(self, item):
        return self.trials[item]

    def __len__(self):
        return len(self.trials)


class AutoMLSequentialWrapper(NonTransformableMixin, MetaStepMixin, BaseStep):
    """
    A step to execute any Automatic Machine Learning Algorithms.

    Example usage :

    .. code-block:: python

        auto_ml: AutoMLSequentialWrapper = AutoMLSequentialWrapper(
            step=ForecastingPipeline(),
            auto_ml_algorithm=AutoMLAlgorithm(
                hyperparameter_optimizer=RandomSearchHyperparameterOptimizer(),
                validation_technique=KFoldCrossValidationWrapper(),
                higher_score_is_better=True
            ),
            hyperparams_repository=HyperparamsJSONRepository(),
            n_iters=100
        )

        auto_ml: AutoMLSequentialWrapper = auto_ml.fit(data_inputs, expected_outputs)

        best_model: ForecastingPipeline = auto_ml.get_best_model()


    .. seealso::
        :class:`AutoMLAlgorithm`,
        :class:`HyperparamsRepository`,
        :class:`RandomSearch`,
        :class:`BaseHyperparameterOptimizer`,
        :class:`RandomSearchHyperparameterOptimizer`,
        :class:`MetaStepMixin`,
        :class:`NonTransformableMixin`,
        :class:`BaseStep`
    """

    def __init__(
            self,
            wrapped: BaseStep,
            auto_ml_algorithm: AutoMLAlgorithm,
            hyperparams_repository: HyperparamsRepository = None,
            n_iters: int = 100,
            refit=True
    ):
        NonTransformableMixin.__init__(self)

        self.refit = refit
        auto_ml_algorithm = auto_ml_algorithm.set_step(wrapped)
        MetaStepMixin.__init__(self, auto_ml_algorithm)

        if hyperparams_repository is None:
            hyperparams_repository = InMemoryHyperparamsRepository()
        self.hyperparams_repository = hyperparams_repository
        self.n_iters = n_iters

    def set_step(self, step: BaseStep) -> BaseStep:
        self.wrapped.set_step(step)

    def _fit_transform_data_container(self, data_container, context):
        new_self = self._fit_data_container(data_container, context)
        return new_self, new_self._transform_data_container(data_container, context)

    def fit_transform(self, data_inputs, expected_outputs):
        new_self = self.fit(data_inputs, expected_outputs)
        return new_self, new_self.transform(data_inputs)

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext):
        """
        Find the best hyperparams using the wrapped AutoML strategy.

        :param data_container: data container to fit on
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: fitted self
        :rtype: BaseStep
        """
        for i in range(self.n_iters):
            auto_ml_trial_data_container: AutoMLContainer = self._load_auto_ml_data(i)

            hyperparams = self.wrapped.find_next_best_hyperparams(auto_ml_trial_data_container)
            self.wrapped = self.wrapped.update_hyperparams(hyperparams)

            self.hyperparams_repository.create_new_trial(hyperparams)

            try:
                self.wrapped, data_container_with_score = self.wrapped.handle_fit_transform(data_container.copy(), context)
                score = data_container_with_score.data_inputs

                self.hyperparams_repository.save_score_for_success_trial(hyperparams, score)
            except Exception as error:
                self.hyperparams_repository.save_failure_for_trial(hyperparams, error)

        if self.refit:
            self.best_model = self._load_best_model().handle_fit(data_container.copy(), context)

        return self

    def fit(self, data_inputs, expected_outputs=None) -> BaseStep:
        """
        Find the best hyperparams using the wrapped AutoML strategy.

        :param data_inputs: data inputs
        :param expected_outputs: expected ouptuts to fit on
        :return: fitted self
        :rtype: BaseStep
        """
        for i in range(self.n_iters):
            auto_ml_trial_container: AutoMLContainer = self._load_auto_ml_data(i)

            hyperparams: HyperparameterSamples = self.wrapped.find_next_best_hyperparams(auto_ml_trial_container)
            self.wrapped = self.wrapped.update_hyperparams(hyperparams)

            self.hyperparams_repository.create_new_trial(hyperparams)

            try:
                self.wrapped, score = self.wrapped.fit_transform(data_inputs, expected_outputs)
                self.hyperparams_repository.save_score_for_success_trial(hyperparams, score)
            except Exception as error:
                self.hyperparams_repository.save_failure_for_trial(hyperparams, error)

        if self.refit:
            self.best_model = self._load_best_model().fit(data_inputs, expected_outputs)

        return self

    def _load_auto_ml_data(self, trial_number: int) -> AutoMLContainer:
        """
        Load data for all trials.

        :param trial_number: trial number
        :type trial_number: int
        :return: auto ml data container
        :rtype: Trials
        """
        trials = self.hyperparams_repository.load_all_trials(TRIAL_STATUS.SUCCESS)

        return AutoMLContainer(
            trial_number=trial_number,
            trials=trials,
            hyperparameter_space=self.wrapped.get_hyperparams_space(),
            n_iters=self.n_iters
        )

    def _load_best_model(self) -> BaseStep:
        """
        Get the best model from all of the previous trials.

        :return: best model step
        :rtype: BaseStep
        """
        trials: Trials = self.hyperparams_repository.load_all_trials(TRIAL_STATUS.SUCCESS)
        auto_ml_algorithm: AutoMLAlgorithm = self.wrapped
        best_hyperparams = auto_ml_algorithm.get_best_hyperparams(trials)
        auto_ml_algorithm = auto_ml_algorithm.update_hyperparams(best_hyperparams)

        return copy.deepcopy(auto_ml_algorithm.get_step())

    def transform(self, data_inputs):
        if self.best_model is None:
            raise Exception('Cannot transform AutoMLSequentialWrapper before fit')
        return self.best_model.transform(data_inputs)

    def _transform_data_container(self, data_container, context):
        if self.best_model is None:
            raise Exception('Cannot transform AutoMLSequentialWrapper before fit')
        return self.best_model.handle_transform(data_container, context)

    def get_best_model(self) -> BaseStep:
        return self.best_model


class RandomSearch(AutoMLSequentialWrapper):
    """
    Random Search Automatic Machine Learning Algorithm that randomly samples the space of random variables.

    Example usage :

    .. code-block:: python

        random_search = RandomSearch(
            ForecastingPipeline(),
            hyperparams_repository=HyperparamsJSONRepository(),
            n_iters=100
        )

    .. seealso::
        :class:`AutoMLSequentialWrapper`,
        :class:`AutoMLAlgorithm`,
        :class:`HyperparamsRepository`,
        :class:`InMemoryHyperparamsRepository`,
        :class:`HyperparamsJSONRepository`,
        :class:`RandomSearch`,
        :class:`BaseHyperparameterOptimizer`,
        :class:`RandomSearchHyperparameterOptimizer`
    """

    def __init__(
            self,
            wrapped: BaseStep = None,
            validation_technique=None,
            hyperparams_repository: HyperparamsRepository = None,
            higher_score_is_better=True,
            n_iter: int = 10
    ):
        AutoMLSequentialWrapper.__init__(
            self,
            wrapped=wrapped,
            auto_ml_algorithm=AutoMLAlgorithm(
                hyperparameter_optimizer=RandomSearchHyperparameterOptimizer(),
                validation_technique=validation_technique,
                higher_score_is_better=higher_score_is_better
            ),
            hyperparams_repository=hyperparams_repository,
            n_iters=n_iter
        )


class RandomSearchHyperparameterOptimizer(BaseHyperparameterOptimizer):
    """
    AutoML Hyperparameter Optimizer that randomly samples the space of random variables.

    Please refer to :class:`AutoMLSequentialWrapper` for a usage example.

    .. seealso::
        :class:`AutoMLAlgorithm`,
        :class:`BaseHyperparameterOptimizer`,
        :class:`AutoMLSequentialWrapper`,
        :class:`TrialsContainer`,
        :class:`HyperparameterSamples`,
        :class:`HyperparameterSpace`
    """

    def __init__(self):
        BaseHyperparameterOptimizer.__init__(self)

    def find_next_best_hyperparams(self, auto_ml_container: 'AutoMLContainer') -> HyperparameterSamples:
        """
        Randomly sample the next hyperparams to try.

        :param auto_ml_container: trials data container
        :type auto_ml_container: Trials
        :return: next best hyperparams
        :rtype: HyperparameterSamples
        """
        return auto_ml_container.hyperparameter_space.rvs()
