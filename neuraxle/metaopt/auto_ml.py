"""
Neuraxle's Automatic Machine Learning Classes
====================================
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
from abc import ABC, abstractmethod
from typing import Tuple, List

from neuraxle.base import MetaStepMixin, BaseStep, NonTransformableMixin
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace
from neuraxle.metaopt.random import BaseCrossValidationWrapper, KFoldCrossValidationWrapper


class HyperparamsRepository(ABC):
    """
    Hyperparams repository that saves hyperparams, and scores for every AutoML trial.

    .. seealso::
        :class:`HyperparamsRepository`,
        :class:`HyperparameterSamples`,
        :class:`AutoMLSequentialWrapper`
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
    def load_all_trials(self) -> Tuple[List[HyperparameterSamples], List[float]]:
        """
        Load all hyperparameter trials with their corresponding score.

        :return: (hyperparams, scores)
        """
        pass

    @abstractmethod
    def save_score_for_trial(self, hyperparams: HyperparameterSamples, score: float):
        """
        Save hyperparams, and score for a trial.

        :return: (hyperparams, scores)
        """
        pass

    def get_best_hyperparams(self, higher_score_is_better=True) -> HyperparameterSamples:
        """
        Get the best hyperparams from all previous trials.

        :return: best hyperparams
        :rtype: HyperparameterSamples
        """
        hyperparams, scores = self.load_all_trials()

        best_score = None
        best_hyperparams = None

        for trial_hyperparam, trial_score in zip(hyperparams, scores):
            if best_score is None or higher_score_is_better == (trial_score > best_score):
                best_score = trial_score
                best_hyperparams = trial_hyperparam

        return best_hyperparams

    def _get_trial_hash(self, hp_dict):
        current_hyperparameters_hash = hashlib.md5(str.encode(str(hp_dict))).hexdigest()
        return current_hyperparameters_hash


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
        score = None
        self._save_trial_json(hyperparams, score)

    def load_all_trials(self) -> Tuple[List[HyperparameterSamples], List[float]]:
        """
        Load all hyperparameter trials with their corresponding score.
        Reads all the saved trial json files.

        :return: (hyperparams, scores)
        """
        all_hyperparams = []
        all_scores = []

        for base_path in glob.glob(os.path.join(self.cache_folder, '*.json')):
            with open(base_path) as f:
                trial_json = json.load(f)

            all_hyperparams.append(HyperparameterSamples(trial_json['hyperparams']))
            all_scores.append(trial_json['score'])

        return all_hyperparams, all_scores

    def save_score_for_trial(self, hyperparams: HyperparameterSamples, score: float):
        """
        Save hyperparams, and score for a trial.

        :return: (hyperparams, scores)
        """
        self._save_trial_json(hyperparams, score)

    def _save_trial_json(self, hyperparams, score):
        """
        Save trial json file.

        :return: (hyperparams, scores)
        """
        hp_dict = hyperparams.to_flat_as_dict_primitive()
        current_hyperparameters_hash = self._get_trial_hash(hp_dict)
        with open(os.path.join(self.cache_folder, current_hyperparameters_hash) + '.json', 'w+') as outfile:
            json.dump({
                'hyperparams': hp_dict,
                'score': score
            }, outfile)


class AutoMLStrategyMixin:
    """
    Base class for Automatic Machine Learning strategies.
    Implement your own custom intelligent search of hyperparameters to get most accurate predictive models.

    .. seealso::
        :class:`BaseCrossValidation`,
        :class:`HyperparameterSamples`,
        :class:`HyperparameterSpace`,
        :class:`MetaStepMixin`,
        :class:`BaseStep`
    """

    def __init__(self, validation_technique: BaseCrossValidationWrapper = None, higher_score_is_better=True):
        if validation_technique is None:
            validation_technique = KFoldCrossValidationWrapper()
        self.validation_technique = validation_technique
        self.higher_score_is_better = higher_score_is_better

    @abstractmethod
    def find_next_best_hyperparams(self, trials_data_container: 'AutoMLDataContainer') -> HyperparameterSamples:
        """
        Find the next best hyperparams using previous trials.

        :param trials_data_container: trials data container
        :type trials_data_container: AutoMLDataContainer
        :return: next best hyperparams
        :rtype: HyperparameterSamples
        """
        raise NotImplementedError()

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('AutoMLStrategyMixin', float):
        """
        Fit cross validation with wrapped step, and return the score.

        :param data_inputs: data inputs
        :param expected_outputs: expected outputs to fit on
        :return: self, step score
        :rtype: (AutoMLStrategyMixin, float)
        """
        step = copy.copy(self.wrapped)
        step: BaseCrossValidationWrapper = copy.copy(self.validation_technique).set_step(step)
        step = step.fit(data_inputs, expected_outputs)

        return self, step.get_score()

    def is_higher_score_better(self):
        return self.higher_score_is_better

    def fit(self, data_inputs, expected_outputs=None):
        return self

    def transform(self, data_inputs, expected_outputs=None):
        return data_inputs


class RandomSearchBaseAutoMLStrategy(AutoMLStrategyMixin, MetaStepMixin, BaseStep):
    """
    Random Search Automatic Machine Learning Strategy that randomly samples the space of random variables.

    .. seealso::
        :class:`AutoMLStrategyMixin`,
        :class:`AutoMLDataContainer`,
        :class:`HyperparameterSamples`,
        :class:`HyperparameterSpace`
    """

    def __init__(self, validation_technique=None):
        AutoMLStrategyMixin.__init__(self, validation_technique)
        MetaStepMixin.__init__(self, None)
        BaseStep.__init__(self)

    def find_next_best_hyperparams(self, trials_data_container: 'AutoMLDataContainer') -> HyperparameterSamples:
        return trials_data_container.hyperparameter_space.rvs()


class AutoMLDataContainer:
    """
    Simple data container used by :class:`AutoMLStrategyMixin`, and :class:`AutoMLSequentialWrapper`.
    """

    def __init__(
            self,
            trial_number: int,
            scores: List[float],
            hyperparams: List[HyperparameterSamples],
            hyperparameter_space: HyperparameterSpace,
            n_iters: int
    ):
        self.hyperparams = hyperparams
        self.scores = scores
        self.trial_number = trial_number
        self.hyperparameter_space = hyperparameter_space
        self.n_iters = n_iters


class AutoMLSequentialWrapper(NonTransformableMixin, MetaStepMixin, BaseStep):
    """
    A step to execute any Automatic Machine Learning Algorithms.

    Example usage :

    .. code-block:: python

        auto_ml = AutoMLSequentialWrapper(
            auto_ml_strategy=RandomSearchAutoMLStrategy(),
            step=ForecastingPipeline(),
            hyperparams_repository=HyperparamsJSONRepository(),
            n_iters=100
        )

        auto_ml: AutoMLSequentialWrapper = auto_ml.fit(data_inputs, expected_outputs)

        best_model: ForecastingPipeline = auto_ml.get_best_model()


    .. seealso::
        :class:`AutoMLStrategyMixin`,
        :class:`RandomSearchAutoMLStrategy`,
        :class:`HyperparamsRepository`,
        :class:`MetaStepMixin`,
        :class:`BaseStep`
    """

    def __init__(
            self,
            auto_ml_strategy: AutoMLStrategyMixin,
            step: BaseStep,
            hyperparams_repository: HyperparamsRepository,
            n_iters: int
    ):
        NonTransformableMixin.__init__(self)

        auto_ml_strategy = auto_ml_strategy.set_step(step)
        MetaStepMixin.__init__(self, auto_ml_strategy)

        self.hyperparams_repository = hyperparams_repository
        self.n_iters = n_iters

    def fit(self, data_inputs, expected_outputs=None) -> BaseStep:
        """
        Find the best hyperparams using the wrapped AutoML strategy.

        :param data_inputs: data inputs
        :param expected_outputs: expected ouptuts to fit on
        :return: fitted self
        :rtype: BaseStep
        """
        for i in range(self.n_iters):
            auto_ml_trial_data_container = self._load_auto_ml_data(i)

            hyperparams = self.wrapped.find_next_best_hyperparams(auto_ml_trial_data_container)
            self.wrapped = self.wrapped.set_hyperparams(hyperparams)

            self.hyperparams_repository.create_new_trial(hyperparams)

            self.wrapped, score = self.wrapped.fit_transform(data_inputs, expected_outputs)

            self.hyperparams_repository.save_score_for_trial(hyperparams, score)

        return self

    def _load_auto_ml_data(self, trial_number: int) -> AutoMLDataContainer:
        """
        Load data for all trials.

        :param trial_number: trial number
        :type trial_number: int
        :return: auto ml data container
        :rtype: AutoMLDataContainer
        """
        hps, scores = self.hyperparams_repository.load_all_trials()

        return AutoMLDataContainer(
            trial_number=trial_number,
            scores=scores,
            hyperparams=hps,
            hyperparameter_space=self.wrapped.get_hyperparams_space(),
            n_iters=self.n_iters
        )

    def get_best_model(self) -> BaseStep:
        """
        Get the best model from all of the previous trials.

        :return: best model step
        :rtype: BaseStep
        """
        auto_ml_strategy: AutoMLStrategyMixin = self.wrapped
        is_higher_score_better = auto_ml_strategy.is_higher_score_better()

        best_hyperparams = self.hyperparams_repository.get_best_hyperparams(is_higher_score_better)
        auto_ml_strategy = auto_ml_strategy.set_hyperparams(best_hyperparams)

        return auto_ml_strategy.get_step()
