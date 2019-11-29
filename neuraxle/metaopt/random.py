"""
Random
====================================
Meta steps for hyperparameter tuning, such as random search.

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

import copy
import glob
import hashlib
import json
import math
import os
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from sklearn.metrics import r2_score

from neuraxle.base import MetaStepMixin, BaseStep
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.steps.loop import StepClonerForEachDataInput
from neuraxle.steps.numpy import NumpyConcatenateOuterBatch, NumpyConcatenateOnCustomAxis


class BaseCrossValidation(MetaStepMixin, BaseStep, ABC):
    # TODO: assert that set_step was called.
    # TODO: change default argument of scoring_function...
    def __init__(self, scoring_function=r2_score, joiner=NumpyConcatenateOuterBatch()):
        MetaStepMixin.__init__(self)
        BaseStep.__init__(self)
        self.scoring_function = scoring_function
        self.joiner = joiner

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseCrossValidation':
        train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs = self.split(
            data_inputs, expected_outputs)

        step = StepClonerForEachDataInput(self.wrapped)
        step = step.fit(train_data_inputs, train_expected_outputs)

        results = step.transform(validation_data_inputs)
        self.scores = [self.scoring_function(a, b) for a, b in zip(results, validation_expected_outputs)]
        self.scores_mean = np.mean(self.scores)
        self.scores_std = np.std(self.scores)

        return self

    @abstractmethod
    def split(self, data_inputs, expected_outputs):
        raise NotImplementedError("TODO")

    def transform(self, data_inputs):
        # TODO: use the splits and average the results?? instead of picking best model...
        raise NotImplementedError("TODO: code this method in Neuraxle.")
        data_inputs = self.split(data_inputs)
        predicted_outputs_splitted = self.wrapped.transform(data_inputs)
        return self.joiner.transform(predicted_outputs_splitted)


class KFoldCrossValidation(BaseCrossValidation):

    def __init__(self, scoring_function=r2_score, k_fold=3, joiner=NumpyConcatenateOuterBatch()):
        self.k_fold = k_fold
        BaseCrossValidation.__init__(self, scoring_function=scoring_function, joiner=joiner)

    def split(self, data_inputs, expected_outputs):
        validation_data_inputs, validation_expected_outputs = self.validation_split(
            data_inputs, expected_outputs)

        train_data_inputs, train_expected_outputs = self.train_split(
            validation_data_inputs, validation_expected_outputs)

        return train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs

    def train_split(self, validation_data_inputs, validation_expected_outputs) -> (List, List):
        train_data_inputs = []
        train_expected_outputs = []
        for i in range(len(validation_data_inputs)):
            inputs = validation_data_inputs[:i] + validation_data_inputs[i + 1:]
            outputs = validation_expected_outputs[:i] + validation_expected_outputs[i + 1:]

            inputs = self.joiner.transform(inputs)
            outputs = self.joiner.transform(outputs)

            train_data_inputs.append(inputs)
            train_expected_outputs.append(outputs)

        return train_data_inputs, train_expected_outputs

    def validation_split(self, data_inputs, expected_outputs=None) -> List:
        splitted_data_inputs = self._split(data_inputs)
        if expected_outputs is not None:
            splitted_expected_outputs = self._split(expected_outputs)
            return splitted_data_inputs, splitted_expected_outputs
        return splitted_data_inputs

    def _split(self, data_inputs):
        splitted_data_inputs = []
        step = len(data_inputs) / float(self.k_fold)
        for i in range(self.k_fold):
            a = int(step * i)
            b = int(step * (i + 1))
            if b > len(data_inputs):
                b = len(data_inputs)

            slice = data_inputs[a:b]
            splitted_data_inputs.append(slice)
        return splitted_data_inputs


class AnchoredWalkForwardTimeSeriesCrossValidation(BaseCrossValidation):
    """
    Prform an anchored walk forward cross validation by performing a forward rolling split.
    All training splits start at the beginning of the time series, but finish at different time. The finish time
    increase toward the end at each forward split.

    For the validation split it will start after a certain time delay (if padding is set)
    after their corresponding training split.

    Notes: The data supported by this cross validation is nd.array of shape [batch_size, total_time_steps, n_features].
    The array can have an arbitrary number of dimension, but the time series axis is currently limited to `axis=1`.
    """

    def __init__(self, minimum_training_size, validation_window_size=None, padding_between_training_and_validation=0,
                 drop_remainder=False, scoring_function=r2_score, joiner=NumpyConcatenateOnCustomAxis(axis=1)):
        """
        Create a anchored walk forward time series cross validation object.

        The size of the validation split is defined by `validation_window_size`.
        The difference in start position between two consecutive validation split is also equal to
        `validation_window_size`.

        :param minimum_training_size: size of the smallest training split.
        :param validation_window_size: size of each validation split and also the time step taken between each
            forward roll, by default None. If None : It takes the value `minimum_training_size`.
        :param padding_between_training_and_validation: the size of the padding between the end of the training split
            and the start of the validation split, by default 0.
        :param drop_remainder: drop the last split if the last validation split does not coincide
            with a full validation_window_size, by default False.
        :param scoring_function: scoring function use to validate performance if it is not None, by default r2_score,
        :param joiner the joiner callable that can join the different result together.
        :return: WalkForwardTimeSeriesCrossValidation instance.
        """
        BaseCrossValidation.__init__(self, scoring_function=scoring_function, joiner=joiner)
        self.minimum_training_size = minimum_training_size
        # If validation_window_size is None, we give the same value as training_window_size.
        self.validation_window_size = validation_window_size or self.minimum_training_size
        self.padding_between_training_and_validation = padding_between_training_and_validation
        self.drop_remainder = drop_remainder
        self._validation_initial_start = self.minimum_training_size + self.padding_between_training_and_validation

    def split(self, data_inputs, expected_outputs):
        """
        Split the data into train inputs, train expected outputs, validation inputs, validation expected outputs.

        Notes: The data supported by this cross validation is nd.array of shape [batch_size, total_time_steps, n_features].
        The array can have an arbitrary number of dimension, but the time series axis is currently limited to `axis=1`.

        :param data_inputs: data to perform walk forward cross validation into.
        :param expected_outputs: the expected target/label that will be used during walk forward cross validation.
        :return: train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs
        """
        validation_data_inputs, validation_expected_outputs = self.validation_split(
            data_inputs, expected_outputs)

        train_data_inputs, train_expected_outputs = self.train_split(
            data_inputs, expected_outputs)

        return train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs

    def train_split(self, data_inputs, expected_outputs=None) -> (List, List):
        """
        Split the data into train inputs, train expected outputs

        Notes: The data supported by this cross validation is nd.array of shape [batch_size, total_time_steps, n_features].
        The array can have an arbitrary number of dimension, but the time series axis is currently limited to `axis=1`.

        :param data_inputs: data to perform walk forward cross validation into.
        :param expected_outputs: the expected target*label that will be used during walk forward cross validation.
        :return: train_data_inputs, train_expected_outputs
        """
        splitted_data_inputs = self._train_split(data_inputs)
        if expected_outputs is not None:
            splitted_expected_outputs = self._train_split(expected_outputs)
            return splitted_data_inputs, splitted_expected_outputs

    def validation_split(self, data_inputs, expected_outputs=None) -> List:
        """
        Split the data into validation inputs, validation expected outputs.

        Notes: The data supported by this cross validation is nd.array of shape [batch_size, total_time_steps, n_features].
        The array can have an arbitrary number of dimension, but the time series axis is currently limited to `axis=1`.

        :param data_inputs: data to perform walk forward cross validation into.
        :param expected_outputs: the expected target*label that will be used during walk forward cross validation.
        :return: validation_data_inputs, validation_expected_outputs
        """
        splitted_data_inputs = self._validation_split(data_inputs)
        if expected_outputs is not None:
            splitted_expected_outputs = self._validation_split(expected_outputs)
            return splitted_data_inputs, splitted_expected_outputs
        return splitted_data_inputs

    def _train_split(self, data_inputs):
        splitted_data_inputs = []
        number_step = self._get_number_fold(data_inputs)

        for i in range(number_step):
            # first slice index is always 0 for anchored walk forward cross validation.
            a = 0
            b = int(self.minimum_training_size + i * self.validation_window_size)

            if b > data_inputs.shape[1]:
                b = data_inputs.shape[1]

            # TODO: The slicer could a inverse_transform of the joiner. A len method should also be defined.
            slice = data_inputs[:, a:b]
            splitted_data_inputs.append(slice)
        return splitted_data_inputs

    def _validation_split(self, data_inputs):
        splitted_data_inputs = []
        number_step = self._get_number_fold(data_inputs)
        for i in range(number_step):
            a = int(self._validation_initial_start + i * self.validation_window_size)
            b = int(a + self.validation_window_size)

            if b > data_inputs.shape[1]:
                b = data_inputs.shape[1]

            # TODO: The slicer could a inverse_transform of the joiner. A len method should also be defined.
            slice = data_inputs[:, a:b]
            splitted_data_inputs.append(slice)
        return splitted_data_inputs

    def _get_number_fold(self, data_inputs):
        if self.drop_remainder:
            number_step = math.floor(
                (data_inputs.shape[1] - self._validation_initial_start) / float(self.validation_window_size)
            )
        else:
            number_step = math.ceil(
                (data_inputs.shape[1] - self._validation_initial_start) / float(self.validation_window_size)
            )
        return number_step


class WalkForwardTimeSeriesCrossValidation(AnchoredWalkForwardTimeSeriesCrossValidation):
    """
    Perform a classic walk forward cross validation by performing a forward rolling split.

    All the training split have the same `validation_window_size` size. The start time and end time of each training
    split will increase identically toward the end at each forward split. Same principle apply with the validation
    split, where the start and end will increase in the same manner toward the end. Each validation split start after
    a certain time delay (if padding is set) after their corresponding training split.

    Notes: The data supported by this cross validation is nd.array of shape [batch_size, total_time_steps, n_features].
    The array can have an arbitrary number of dimension, but the time series axis is currently limited to `axis=1`.
    """

    def __init__(self, training_window_size, validation_window_size=None, padding_between_training_and_validation=0,
                 drop_remainder=False, scoring_function=r2_score, joiner=NumpyConcatenateOnCustomAxis(axis=1)):
        """
        Create a classic walk forward time series cross validation object.

        The difference in start position between two consecutive validation split are equal to one
        `validation_window_size`.

        :param training_window_size: the window size of training split.
        :param validation_window_size: the window size of each validation split and also the time step taken between
            each forward roll, by default None. If None : It takes the value `training_window_size`.
        :param padding_between_training_and_validation: the size of the padding between the end of the training split
            and the start of the validation split, by default 0.
        :param drop_remainder: drop the last split if the last validation split does not coincide
            with a full validation_window_size, by default False.
        :param scoring_function: scoring function use to validate performance if it is not None, by default r2_score,
        :param joiner the joiner callable that can join the different result together.
        :return: WalkForwardTimeSeriesCrossValidation instance.
        """
        AnchoredWalkForwardTimeSeriesCrossValidation.__init__(
            self,
            training_window_size,
            validation_window_size=validation_window_size,
            padding_between_training_and_validation=padding_between_training_and_validation,
            drop_remainder=drop_remainder, scoring_function=scoring_function, joiner=joiner
        )

    def _train_split(self, data_inputs):
        splitted_data_inputs = []
        number_step = self._get_number_fold(data_inputs)

        for i in range(number_step):
            a = int(i * self.validation_window_size)
            # Here minimum_training_size = training_size, since each training split has the same length.
            b = int(a + self.minimum_training_size)

            if b > data_inputs.shape[1]:
                b = data_inputs.shape[1]

            # TODO: The slicer could a inverse_transform of the joiner. A len method should also be defined.
            slice = data_inputs[:, a:b]
            splitted_data_inputs.append(slice)
        return splitted_data_inputs


class RandomSearch(MetaStepMixin, BaseStep):
    """Perform a random hyperparameter search."""

    # TODO: CV and rename to RandomSearchCV.

    def __init__(
            self,
            wrapped=None,
            n_iter: int = 10,
            higher_score_is_better: bool = True,
            validation_technique: BaseCrossValidation = KFoldCrossValidation(),
            refit=True,
    ):
        if wrapped is not None:
            MetaStepMixin.__init__(self, wrapped)
        BaseStep.__init__(self)
        self.n_iter = n_iter
        self.higher_score_is_better = higher_score_is_better
        self.validation_technique: BaseCrossValidation = validation_technique
        self.refit = refit

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        started = False
        best_hyperparams = None

        for _ in range(self.n_iter):

            step = copy.copy(self.wrapped)

            new_hyperparams = step.get_hyperparams_space().rvs()
            step.set_hyperparams(new_hyperparams)

            step: BaseCrossValidation = copy.copy(self.validation_technique).set_step(step)

            step = step.fit(data_inputs, expected_outputs)
            score = step.scores_mean

            if not started or self.higher_score_is_better == (score > self.score):
                started = True
                self.score = score
                self.best_validation_wrapper_of_model = copy.copy(step)
                best_hyperparams = new_hyperparams

        self.best_validation_wrapper_of_model.wrapped.set_hyperparams(best_hyperparams)

        if self.refit:
            self.best_model = self.best_validation_wrapper_of_model.wrapped.fit(
                data_inputs,
                expected_outputs
            )

        return self

    def get_best_model(self):
        return self.best_model

    def transform(self, data_inputs):
        if self.best_validation_wrapper_of_model is None:
            raise Exception('Cannot transform RandomSearch before fit')
        return self.best_validation_wrapper_of_model.wrapped.transform(data_inputs)


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

# class HyperparameterFeaturizer:
#     pass
#
#
# class AutoMLStrategyMixin:
#     @abstractmethod
#     def fit(self, hyperparameters: HyperparameterSamples, scores: List[float]):
#         pass
#
#     @abstractmethod
#     def guess_next_best_params(self, hyperparameters: HyperparameterSamples,
#                                scores: List[float]) -> HyperparameterSamples:
#         pass
#
#
# class AutoMLSequentialWrapper(MetaStepMixin, BaseStep):
#     def __init__(
#             self,
#             wrapped,
#             auto_ml_strategy,
#             validation_technique,
#             score_function,
#             hyperparams_repository,
#             n_iters
#     ):
#         MetaStepMixin.__init__(self, wrapped)
#         self.hyperparams_repository = hyperparams_repository
#         self.score_function = score_function
#         self.validation_technique = validation_technique
#         self.auto_ml_strategy = auto_ml_strategy
#         self.n_iters = n_iters
#
#     def fit(self, data_inputs, expected_outputs=None):
#         for i in self.n_iters:
#             hps, scores = self.hyperparams_repository.load_all_trials()
#
#             self.auto_ml_strategy = self.auto_ml_strategy.fit(hps, scores)
#
#             next_model_to_try_hps = self.auto_ml_strategy.guess_next_best_params(i, self.n_iters,
#                                                                                  self.wrapped.get_hyperparams_space())
#
#             self.hyperparams_repository.create_new_trial(next_model_to_try_hps)
#
#             validation_wrapper = self.validation_technique(
#                 copy.copy(self.wrapped).set_hyperparams(next_model_to_try_hps)
#             )
#
#             validation_wrapper, predicted = validation_wrapper.fit_transform(data_inputs, expected_outputs)
#
#             score = self.score_function(expected_outputs, predicted)
#
#             self.hyperparams_repository.save_score_for_trial(next_model_to_try_hps, score)
