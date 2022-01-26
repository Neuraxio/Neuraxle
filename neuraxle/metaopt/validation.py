"""
Validation
====================================
Classes for hyperparameter tuning, such as random search.

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

..
    Thanks to Umaneo Technologies Inc. for their contributions to this Machine Learning
    project, visit https://www.umaneo.com/ for more information on Umaneo Technologies Inc.

"""
import copy
import math
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import (Any, Callable, Dict, Generic, Iterable, List, Optional,
                    Tuple, Type, TypeVar, Union)

import numpy as np
from neuraxle.base import (BaseStep, EvaluableStepMixin, ExecutionContext,
                           ForceHandleOnlyMixin, MetaStep, TrialStatus)
from neuraxle.data_container import DataContainer
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.distributions import (
    ContinuousHyperparameterDistribution, DiscreteHyperparameterDistribution)
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace, RecursiveDict)
from neuraxle.metaopt.data.aggregates import Round
from neuraxle.metaopt.data.vanilla import BaseHyperparameterOptimizer
from neuraxle.steps.loop import StepClonerForEachDataInput
from neuraxle.steps.numpy import (NumpyConcatenateInnerFeatures,
                                  NumpyConcatenateOnAxis,
                                  NumpyConcatenateOuterBatch)
from sklearn.metrics import r2_score


class BaseValidation(MetaStep, ABC):
    """
    Base class For validation wrappers.
    It has a scoring function to calculate the score for the validation split.

    .. seealso::
        :class`neuraxle.metaopt.validation.ValidationSplitWrapper`,
        :class`Kneuraxle.metaopt.validation.FoldCrossValidationWrapper`,
        :class`neuraxle.metaopt.validation.AnchoredWalkForwardTimeSeriesCrossValidationWrapper`,
        :class`neuraxle.metaopt.validation.WalkForwardTimeSeriesCrossValidationWrapper`

    """

    def __init__(self, wrapped=None, scoring_function: Callable = r2_score):
        """
        Base class For validation wrappers.
        It has a scoring function to calculate the score for the validation split.

        :param scoring_function: scoring function with two arguments (y_true, y_pred)
        :type scoring_function: Callable
        """
        BaseStep.__init__(self)
        MetaStep.__init__(self, wrapped)
        self.scoring_function = scoring_function

    @abstractmethod
    def split_data_container(self, data_container) -> Tuple[DataContainer, DataContainer]:
        pass


class BaseCrossValidationWrapper(EvaluableStepMixin, ForceHandleOnlyMixin, BaseValidation, ABC):
    # TODO: change default argument of scoring_function...
    def __init__(self, wrapped=None, scoring_function=r2_score, joiner=NumpyConcatenateOuterBatch(),
                 cache_folder_when_no_handle=None,
                 split_data_container_during_fit=True, predict_after_fit=True):
        BaseValidation.__init__(self, wrapped=wrapped, scoring_function=scoring_function)
        ForceHandleOnlyMixin.__init__(self, cache_folder=cache_folder_when_no_handle)
        EvaluableStepMixin.__init__(self)

        self.split_data_container_during_fit = split_data_container_during_fit
        self.predict_after_fit = predict_after_fit
        self.joiner = joiner

    def train(self, train_data_container: DataContainer, context: ExecutionContext):
        step = StepClonerForEachDataInput(self.wrapped)
        step = step.handle_fit(train_data_container, context)

        return step

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> BaseStep:
        assert self.wrapped is not None

        step = StepClonerForEachDataInput(self.wrapped)
        step = step.handle_fit(data_container, context)

        return step

    def calculate_score(self, results):
        self.scores = [self.scoring_function(a, b) for a, b in zip(results.data_inputs, results.expected_outputs)]
        self.scores_mean = np.mean(self.scores)
        self.scores_std = np.std(self.scores)

    def split_data_container(self, data_container: DataContainer) -> Tuple[DataContainer, DataContainer]:
        train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs = self.split(
            data_container.data_inputs,
            data_container.expected_outputs
        )

        train_data_container = DataContainer(data_inputs=train_data_inputs, expected_outputs=train_expected_outputs)
        validation_data_container = DataContainer(
            data_inputs=validation_data_inputs,
            expected_outputs=validation_expected_outputs
        )

        return train_data_container, validation_data_container

    def get_score(self):
        return self.scores_mean

    def get_scores_std(self):
        return self.scores_std

    @abstractmethod
    def split(self, data_inputs, expected_outputs):
        raise NotImplementedError("TODO")


class ValidationSplitWrapper(BaseCrossValidationWrapper):
    """
    Wrapper for validation split that calculates the score for the validation split.

    .. code-block:: python

        random_search = Pipeline([
            RandomSearch(
                ValidationSplitWrapper(
                    Identity(),
                    test_size=0.1
                    scoring_function=mean_absolute_relative_error,
                    run_validation_split_in_test_mode=False
                ),
                n_iter= 10,
                higher_score_is_better= True,
                validation_technique=KFoldCrossValidationWrapper(),
                refit=True
            )
        ])

    .. note::
        The data is not shuffled before split. Please refer to the :class`DataShuffler` step for data shuffling.

    .. seealso::
        :class`BaseValidation`,
        :class`BaseCrossValidationWrapper`,
        :class`neuraxle.metaopt.auto_ml.RandomSearch`,
        :class`neuraxle.steps.data.DataShuffler`

    """

    def __init__(
            self,
            wrapped: BaseStep = None,
            test_size: float = 0.2,
            scoring_function=r2_score,
            run_validation_split_in_test_mode=True,
            cache_folder_when_no_handle=None
    ):
        """
        :param wrapped: wrapped step
        :param test_size: ratio for test size between 0 and 1
        :param scoring_function: scoring function with two arguments (y_true, y_pred)
        """
        BaseCrossValidationWrapper.__init__(self, wrapped=wrapped,
                                            cache_folder_when_no_handle=cache_folder_when_no_handle)

        self.run_validation_split_in_test_mode = run_validation_split_in_test_mode
        self.test_size = test_size
        self.scoring_function = scoring_function

    def _fit_data_container(
        self, data_container: DataContainer, context: ExecutionContext
    ) -> Tuple['ValidationSplitWrapper', DataContainer]:
        """
        Fit using the training split.
        Calculate the scores using the validation split.

        :param context: execution context
        :param data_container: data container
        :type context: ExecutionContext
        :type data_container: DataContainer
        :return: fitted self
        """
        new_self, results_data_container = self._fit_transform_data_container(data_container, context)
        return new_self

    def _fit_transform_data_container(
        self, data_container: DataContainer, context: ExecutionContext
    ) -> Tuple['BaseStep', DataContainer]:
        """
        Fit Transform given data inputs without splitting.

        :param context:
        :param data_container: DataContainer
        :type data_container: DataContainer
        :type context: ExecutionContext
        :return: outputs
        """
        train_data_container, validation_data_container = self.split_data_container(data_container)

        self.wrapped, results_data_container = self.wrapped.handle_fit_transform(train_data_container,
                                                                                 context.push(self.wrapped))

        self._update_scores_train(results_data_container.data_inputs, results_data_container.expected_outputs)

        results_data_container = self.wrapped.handle_predict(validation_data_container, context.push(self.wrapped))

        self._update_scores_validation(results_data_container.data_inputs, results_data_container.expected_outputs)

        self.wrapped.apply('disable_metrics')
        data_container = self.wrapped.handle_predict(data_container, context.push(self.wrapped))
        self.wrapped.apply('enable_metrics')

        return self, data_container

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
        """
        Transform given data inputs without splitting.

        :param context: execution context
        :param data_container: DataContainer
        :type data_container: DataContainer
        :type context: ExecutionContext
        :return: outputs
        """
        return self.wrapped.handle_transform(data_container, context.push(self.wrapped))

    def _update_scores_validation(self, data_inputs, expected_outputs):
        self.scores_validation = self.scoring_function(expected_outputs, data_inputs)
        self.scores_validation_mean = np.mean(self.scores_validation)
        self.scores_validation_std = np.std(self.scores_validation)

    def _update_scores_train(self, data_inputs, expected_outputs):
        self.scores_train = self.scoring_function(expected_outputs, data_inputs)
        self.scores_train_mean = np.mean(self.scores_train)
        self.scores_train_std = np.std(self.scores_train)

    def get_score(self):
        return self.scores_validation_mean

    def get_score_validation(self):
        return self.scores_validation_mean

    def get_score_train(self):
        return self.scores_validation_mean

    def split_data_container(self, data_container) -> Tuple[DataContainer, DataContainer]:
        """
        Split data container into a training set, and a validation set.

        :param data_container: data container
        :type data_container: DataContainer
        :return: train_data_container, validation_data_container
        """

        train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs = \
            self.split(data_container.data_inputs, data_container.expected_outputs)

        train_ids = self.train_split(data_container.ids)
        train_data_container = DataContainer(
            data_inputs=train_data_inputs,
            ids=train_ids,
            expected_outputs=train_expected_outputs
        )

        validation_ids = self.validation_split(data_container.ids)
        validation_data_container = DataContainer(
            data_inputs=validation_data_inputs,
            ids=validation_ids,
            expected_outputs=validation_expected_outputs
        )

        return train_data_container, validation_data_container

    def split(self, data_inputs, expected_outputs=None) -> Tuple[List, List, List, List]:
        """
        Split data inputs, and expected outputs into a training set, and a validation set.

        :param data_inputs: data inputs to split
        :param expected_outputs: expected outputs to split
        :return: train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs
        """
        validation_data_inputs = self.validation_split(data_inputs)
        validation_expected_outputs = None
        if expected_outputs is not None:
            validation_expected_outputs = self.validation_split(expected_outputs)

        train_data_inputs = self.train_split(data_inputs)
        train_expected_outputs = None
        if expected_outputs is not None:
            train_expected_outputs = self.train_split(expected_outputs)

        return train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs

    def train_split(self, data_inputs) -> List:
        """
        Split training set.

        :param data_inputs: data inputs to split
        :return: train_data_inputs
        """
        return data_inputs[0:self._get_index_split(data_inputs)]

    def validation_split(self, data_inputs) -> List:
        """
        Split validation set.

        :param data_inputs: data inputs to split
        :return: validation_data_inputs
        """
        return data_inputs[self._get_index_split(data_inputs):]

    def disable_metrics(self):
        self.metrics_enabled = False
        if self.wrapped is not None:
            self.wrapped.apply('disable_metrics')
        return RecursiveDict()

    def enable_metrics(self):
        self.metrics_enabled = True
        if self.wrapped is not None:
            self.wrapped.apply('enable_metrics')
        return RecursiveDict()

    def _get_index_split(self, data_inputs):
        return math.floor(len(data_inputs) * (1 - self.test_size))


def average_kfold_scores(metric_function):
    def calculate(y_true_kfolds, y_pred_kfolds):
        kfold_scores = []
        for y_true, y_pred in zip(y_true_kfolds, y_pred_kfolds):
            kfold_scores.append(metric_function(y_true, y_pred))

        return np.mean(kfold_scores)

    return calculate


class KFoldCrossValidationWrapper(BaseCrossValidationWrapper):
    def __init__(
            self,
            scoring_function=r2_score,
            k_fold=3,
            joiner=NumpyConcatenateOuterBatch(),
            cache_folder_when_no_handle=None
    ):
        self.k_fold = k_fold
        BaseCrossValidationWrapper.__init__(
            self,
            scoring_function=scoring_function,
            joiner=joiner,
            cache_folder_when_no_handle=cache_folder_when_no_handle
        )

    def split(self, data_inputs, expected_outputs):
        validation_data_inputs, validation_expected_outputs = self.validation_split(data_inputs, expected_outputs)
        train_data_inputs, train_expected_outputs = self.train_split(data_inputs, expected_outputs)

        return train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs

    def train_split(self, data_inputs, expected_outputs) -> Tuple[List, List]:
        train_data_inputs = []
        train_expected_outputs = []
        data_inputs = np.array(data_inputs)
        expected_outputs = np.array(expected_outputs)

        for i in range(len(data_inputs)):
            before_di = data_inputs[:i]
            after_di = data_inputs[i + 1:]
            inputs = (before_di, after_di)

            before_eo = expected_outputs[:i]
            after_eo = expected_outputs[i + 1:]
            outputs = (before_eo, after_eo)

            inputs = self.joiner.transform(inputs)
            outputs = self.joiner.transform(outputs)

            train_data_inputs.append(inputs)
            train_expected_outputs.append(outputs)

        return train_data_inputs, train_expected_outputs

    def validation_split(self, data_inputs, expected_outputs=None) -> Tuple[List, List]:
        splitted_data_inputs = self._split(data_inputs)
        if expected_outputs is not None:
            splitted_expected_outputs = self._split(expected_outputs)
            return splitted_data_inputs, splitted_expected_outputs

        return splitted_data_inputs, [None] * len(splitted_data_inputs)

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


class AnchoredWalkForwardTimeSeriesCrossValidationWrapper(BaseCrossValidationWrapper):
    """
    Perform an anchored walk forward cross validation by performing a forward rolling split.
    All training splits start at the beginning of the time series, but finish at different time. The finish time
    increase toward the end at each forward split.

    For the validation split it will start after a certain time delay (if padding is set)
    after their corresponding training split.

    Notes: The data supported by this cross validation is nd.array of shape [batch_size, total_time_steps, n_features].
    The array can have an arbitrary number of dimension, but the time series axis is currently limited to `axis=1`.
    """

    def __init__(self, minimum_training_size, validation_window_size=None, padding_between_training_and_validation=0,
                 drop_remainder=False, scoring_function=r2_score, joiner=NumpyConcatenateInnerFeatures()):
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
        BaseCrossValidationWrapper.__init__(self, scoring_function=scoring_function, joiner=joiner)
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

    def train_split(self, data_inputs, expected_outputs=None) -> Tuple[List, List]:
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


class WalkForwardTimeSeriesCrossValidationWrapper(AnchoredWalkForwardTimeSeriesCrossValidationWrapper):
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
                 drop_remainder=False, scoring_function=r2_score, joiner=NumpyConcatenateOnAxis(axis=1)):
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
        AnchoredWalkForwardTimeSeriesCrossValidationWrapper.__init__(
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


class RandomSearch(BaseHyperparameterOptimizer):
    """
    AutoML Hyperparameter Optimizer that randomly samples the space of random variables.
    Please refer to :class:`AutoML` for a usage example.

    .. seealso::
        :class:`Trainer`,
        :class:`~neuraxle.metaopt.data.trial.Trial`,
        :class:`~neuraxle.metaopt.data.trial.Trials`,
        :class:`HyperparamsRepository`,
        :class:`HyperparamsJSONRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(self):
        BaseHyperparameterOptimizer.__init__(self)

    def find_next_best_hyperparams(self, round_scope: 'Round') -> HyperparameterSamples:
        """
        Randomly sample the next hyperparams to try.

        :param round_scope: round scope
        :return: next random hyperparams
        """
        return round_scope.hp_space.rvs()


class GridExplorationSampler(BaseHyperparameterOptimizer):
    """
    This hyperparameter space optimizer is similar to a grid search, however, it does
    try to sample maximally different points in the space to explore it. This space
    optimizer has a fixed pseudorandom exploration method that makes the sampling
    reproductible.

    It may be good for space exploration before a TPE or for unit tests.
    """

    def __init__(self, expected_n_trials: int):
        BaseHyperparameterOptimizer.__init__(self)
        self.expected_n_trials: int = expected_n_trials

        self._i: int = 0
        self.flat_hp_grid_values: OrderedDict[str, List[Any]] = {}
        self.flat_hp_grid_lens: List[int] = []

    def find_next_best_hyperparams(self, round_scope: 'Round') -> HyperparameterSamples:
        """
        Sample the next hyperparams to try.

        :param round_scope: round scope
        :return: next hyperparams
        """
        if self._i == 0:
            self._generate_grid(round_scope.hp_space)
        else:
            self._reshuffle_grid()
        self._i += 1

        i_grid_keys: List[int] = self._gen_keys_for_grid()
        flat_result: OrderedDict[str, Any] = self[i_grid_keys]
        return HyperparameterSamples(flat_result)

    def _generate_grid(self, hp_space: HyperparameterSpace):
        """
        Generate the grid of hyperparameters to pick from.

        :param hp_space: hyperparameter space
        """
        # Start with discrete params:
        for hp_name, hp_dist in hp_space.to_flat_dict().items():
            if isinstance(hp_dist, DiscreteHyperparameterDistribution):
                hp_samples: List[Any] = hp_dist.values()

                reordered_hp_samples = self.disorder(hp_samples)
                self.flat_hp_grid_values[hp_name] = reordered_hp_samples
                self.flat_hp_grid_lens.append(len(reordered_hp_samples))

        # Then fill the remaining continous params using the expected_n_trials:
        remainder: int = max(3, self.expected_n_trials)
        for hp_name, hp_dist in hp_space.to_flat_dict().items():
            if isinstance(hp_dist, ContinuousHyperparameterDistribution):
                hp_samples: List[Any] = [
                    hp_dist.mean(),
                    hp_dist.min(),
                    hp_dist.max(),
                    hp_dist.mean() + hp_dist.std(),
                    hp_dist.mean() - hp_dist.std(),
                    hp_dist.mean() + hp_dist.std() / 2,
                    hp_dist.mean() - hp_dist.std() / 2,
                    hp_dist.mean() + hp_dist.std() * 1.5,
                    hp_dist.mean() - hp_dist.std() * 1.5,
                    hp_dist.mean() + hp_dist.std() / 4,
                    hp_dist.mean() - hp_dist.std() / 4,
                    hp_dist.mean() + hp_dist.std() * 2.5,
                    hp_dist.mean() - hp_dist.std() * 2.5,
                ]
                hp_samples: List[Any] = [x for x in hp_samples[:remainder] if x >= hp_dist.min() and x <= hp_dist.max()]

                self.flat_hp_grid_values[hp_name] = hp_samples
                self.flat_hp_grid_lens.append(len(hp_samples))

        # Then readjust the expected_n_trials to be a multiple of the number of hyperparameters:
        _sum = sum(self.flat_hp_grid_lens)
        if self.expected_n_trials != _sum:
            warnings.warn(
                f"Warning: changed {self.__class__.__name__}.expected_n_trials="
                f"{self.expected_n_trials} to {_sum}."
            )
        self.expected_n_trials = _sum

    @staticmethod
    def disorder(x: list, seed: int = 0) -> list:
        """
        Shuffle a list to create a pseudo-random order that is interesting.
        """
        x = copy.copy(x)
        for i in reversed(range(len(x))):
            v = x[i]
            if (len(x) + i + seed) % 2 and (i + seed) != 0:
                del x[i]
                x.insert(1 - (seed % 2), v)
        return x

    def _gen_keys_for_grid(self) -> List[int]:
        """
        Generate the keys for the grid.

        :param i: index
        :return: keys
        """
        flat_idx: List[int] = []
        for _len in self.flat_hp_grid_lens:
            flat_idx.append(self._i % _len)
        return flat_idx

    def _reshuffle_grid(self):
        """
        Reshuffling with pseudo-random seed the hyperparameters' values:
        """
        for seed, (k, v) in enumerate(self.flat_hp_grid_values.items()):
            if self._i % len(v) == 0:
                # reshuffling the grid for when it comes to the end of each of its sublist.
                # TODO: make a test for this to ensure that over an infinity of trials that the lists are shuffled evenly, or use a real and repeatable pseudorandom rng generator for it.
                new_v = self.disorder(v, seed % (self._i + 1) + self._i % (seed + 1))
                self.flat_hp_grid_values[k] = new_v

    def __getitem__(self, i_grid_keys: List[int]) -> OrderedDict[str, Any]:
        """
        Access the keys for the grid.

        :param i_grid_keys: keys
        :return: hyperparams
        """
        flat_result: OrderedDict[str, Any] = OrderedDict()
        for j, hp_name in enumerate(self.flat_hp_grid_values.keys()):
            flat_result[hp_name] = self.flat_hp_grid_values[hp_name][i_grid_keys[j]]
        return flat_result


class BaseValidationSplitter(ABC):
    def split_dact(self, data_container: DACT, context: ExecutionContext) -> List[
            Tuple[DACT, DACT]]:
        """
        Wrap a validation split function with a split data container function.
        A validation split function takes two arguments:  data inputs, and expected outputs.

        :param data_container: data container to split
        :return: a function that returns the pairs of training, and validation data containers for each validation split.
        """
        train_data_inputs, train_expected_outputs, train_ids, validation_data_inputs, validation_expected_outputs, validation_ids = self.split(
            data_inputs=data_container.data_inputs,
            expected_outputs=data_container.expected_outputs,
            context=context
        )

        train_data_container = DACT(data_inputs=train_data_inputs,
                                    ids=train_ids,
                                    expected_outputs=train_expected_outputs)
        validation_data_container = DACT(data_inputs=validation_data_inputs,
                                         ids=validation_ids,
                                         expected_outputs=validation_expected_outputs)

        splits = []
        for (train_id, train_di, train_eo), (validation_id, validation_di, validation_eo) in zip(
                train_data_container, validation_data_container):
            # TODO: use ListDACT instead of DACT
            train_data_container_split = DACT(
                data_inputs=train_di,
                expected_outputs=train_eo
            )

            validation_data_container_split = DACT(
                data_inputs=validation_di,
                expected_outputs=validation_eo
            )

            splits.append((train_data_container_split, validation_data_container_split))

        return splits

    @abstractmethod
    def split(self, data_inputs, ids=None, expected_outputs=None, context: ExecutionContext = None) \
            -> Tuple[List, List, List, List, List, List]:
        """
        Train/Test split data inputs and expected outputs.

        :param data_inputs: data inputs
        :param ids: id associated with each data entry (optional)
        :param expected_outputs: expected outputs (optional)
        :param context: execution context (optional)
        :return: train_data_inputs, train_expected_outputs, train_ids, validation_data_inputs, validation_expected_outputs, validation_ids
        """
        pass


class KFoldCrossValidationSplitter(BaseValidationSplitter):
    """
    Create a function that splits data with K-Fold Cross-Validation resampling.

    .. code-block:: python

        # create a kfold cross validation splitter with 2 kfold
        kfold_cross_validation_split(0.20)


    :param k_fold: number of folds.
    :return:
    """

    def __init__(self, k_fold: int):
        BaseValidationSplitter.__init__(self)
        self.k_fold = k_fold

    def split(self, data_inputs, ids=None, expected_outputs=None, context: ExecutionContext = None) \
            -> Tuple[List, List, List, List, List, List]:
        data_inputs_train, data_inputs_val = kfold_cross_validation_split(
            data_inputs=data_inputs,
            k_fold=self.k_fold
        )

        if ids is not None:
            ids_train, ids_val = kfold_cross_validation_split(
                data_inputs=ids,
                k_fold=self.k_fold
            )
        else:
            ids_train, ids_val = [None] * len(data_inputs_train), [None] * len(data_inputs_val)

        if expected_outputs is not None:
            expected_outputs_train, expected_outputs_val = kfold_cross_validation_split(
                data_inputs=expected_outputs,
                k_fold=self.k_fold
            )
        else:
            expected_outputs_train, expected_outputs_val = [None] * len(data_inputs_train), [None] * len(
                data_inputs_val)

        return data_inputs_train, expected_outputs_train, ids_train, \
            data_inputs_val, expected_outputs_val, ids_val


def kfold_cross_validation_split(data_inputs, k_fold):
    splitted_train_data_inputs = []
    splitted_validation_inputs = []

    step = len(data_inputs) / float(k_fold)
    for i in range(k_fold):
        a = int(step * i)
        b = int(step * (i + 1))
        if b > len(data_inputs):
            b = len(data_inputs)

        validation = data_inputs[a:b]
        train = np.concatenate((data_inputs[:a], data_inputs[b:]), axis=0)

        splitted_validation_inputs.append(validation)
        splitted_train_data_inputs.append(train)

    return splitted_train_data_inputs, splitted_validation_inputs


class ValidationSplitter(BaseValidationSplitter):
    """
    Create a function that splits data into a training, and a validation set.

    .. code-block:: python

        # create a validation splitter function with 80% train, and 20% validation
        validation_splitter(0.20)


    :param test_size: test size in float
    :return:
    """

    def __init__(self, test_size: float):
        self.test_size = test_size

    def split(
        self, data_inputs, ids=None, expected_outputs=None, context: ExecutionContext = None
    ) -> Tuple[List, List, List, List]:
        train_data_inputs, train_expected_outputs, train_ids, validation_data_inputs, validation_expected_outputs, validation_ids = validation_split(
            test_size=self.test_size,
            data_inputs=data_inputs,
            ids=ids,
            expected_outputs=expected_outputs
        )

        return [train_data_inputs], [train_expected_outputs], [train_ids], \
               [validation_data_inputs], [validation_expected_outputs], [validation_ids]


def validation_split(test_size: float, data_inputs, ids=None, expected_outputs=None) \
        -> Tuple[List, List, List, List, List, List]:
    """
    Split data inputs, and expected outputs into a training set, and a validation set.

    :param test_size: test size in float
    :param data_inputs: data inputs to split
    :param ids: ids associated with each data entry
    :param expected_outputs: expected outputs to split
    :return: train_data_inputs, train_expected_outputs, ids_train, validation_data_inputs, validation_expected_outputs, ids_val
    """
    validation_data_inputs = _validation_split(data_inputs, test_size)
    validation_expected_outputs, ids_val = None, None
    if expected_outputs is not None:
        validation_expected_outputs = _validation_split(expected_outputs, test_size)
    if ids is not None:
        ids_val = _validation_split(ids, test_size)

    train_data_inputs = _train_split(data_inputs, test_size)
    train_expected_outputs, ids_train = None, None
    if expected_outputs is not None:
        train_expected_outputs = _train_split(expected_outputs, test_size)
    if ids is not None:
        ids_train = _train_split(ids, test_size)

    return train_data_inputs, train_expected_outputs, ids_train, \
        validation_data_inputs, validation_expected_outputs, ids_val


def _train_split(data_inputs, test_size) -> List:
    """
    Split training set.

    :param data_inputs: data inputs to split
    :return: train_data_inputs
    """
    return data_inputs[0:_get_index_split(data_inputs, test_size)]


def _validation_split(data_inputs, test_size) -> List:
    """
    Split validation set.

    :param data_inputs: data inputs to split
    :return: validation_data_inputs
    """
    return data_inputs[_get_index_split(data_inputs, test_size):]


def _get_index_split(data_inputs, test_size):
    return math.floor(len(data_inputs) * (1 - test_size))
