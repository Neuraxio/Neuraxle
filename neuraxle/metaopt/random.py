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
import math
from abc import ABC, abstractmethod
from typing import List, Callable, Tuple

import numpy as np
from sklearn.metrics import r2_score

from neuraxle.base import MetaStepMixin, BaseStep, ExecutionContext, HandleOnlyMixin, ForceHandleOnlyMixin, \
    EvaluableStepMixin
from neuraxle.data_container import DataContainer
from neuraxle.steps.loop import StepClonerForEachDataInput
from neuraxle.steps.numpy import NumpyConcatenateOuterBatch, NumpyConcatenateOnCustomAxis

VALIDATION_SUB_DATA_CONTAINER_NAME = 'validation'


class BaseValidation(MetaStepMixin, BaseStep, ABC):
    """
    Base class For validation wrappers.
    It has a scoring function to calculate the score for the validation split.

    .. seealso::
        :class`neuraxle.metaopt.random.ValidationSplitWrapper`,
        :class`Kneuraxle.metaopt.random.FoldCrossValidationWrapper`,
        :class`neuraxle.metaopt.random.AnchoredWalkForwardTimeSeriesCrossValidationWrapper`,
        :class`neuraxle.metaopt.random.WalkForwardTimeSeriesCrossValidationWrapper`

    """

    def __init__(self, wrapped=None, scoring_function: Callable = r2_score):
        """
        Base class For validation wrappers.
        It has a scoring function to calculate the score for the validation split.

        :param scoring_function: scoring function with two arguments (y_true, y_pred)
        :type scoring_function: Callable
        """
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        self.scoring_function = scoring_function

    @abstractmethod
    def split_data_container(self, data_container) -> Tuple[DataContainer, DataContainer]:
        pass


class BaseCrossValidationWrapper(EvaluableStepMixin, ForceHandleOnlyMixin, BaseValidation, ABC):
    # TODO: change default argument of scoring_function...
    def __init__(self, wrapped=None, scoring_function=r2_score, joiner=NumpyConcatenateOuterBatch(), cache_folder_when_no_handle=None,
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
        BaseCrossValidationWrapper.__init__(self, wrapped=wrapped, cache_folder_when_no_handle=cache_folder_when_no_handle)

        self.run_validation_split_in_test_mode = run_validation_split_in_test_mode
        self.test_size = test_size
        self.scoring_function = scoring_function

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('ValidationSplitWrapper', DataContainer):
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

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        """
        Fit Transform given data inputs without splitting.

        :param context:
        :param data_container: DataContainer
        :type data_container: DataContainer
        :type context: ExecutionContext
        :return: outputs
        """
        train_data_container, validation_data_container = self.split_data_container(data_container)

        # add sub data container for the validation metrics calculated in MetricsWrapper
        train_data_container.add_sub_data_container(
            name=VALIDATION_SUB_DATA_CONTAINER_NAME,
            data_container=validation_data_container
        )

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

        train_ids = self.train_split(data_container.current_ids)
        train_data_container = DataContainer(
            data_inputs=train_data_inputs,
            current_ids=train_ids,
            summary_id=data_container.summary_id,
            expected_outputs=train_expected_outputs
        )

        validation_ids = self.validation_split(data_container.current_ids)
        validation_data_container = DataContainer(
            data_inputs=validation_data_inputs,
            current_ids=validation_ids,
            summary_id=data_container.summary_id,
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

    def enable_metrics(self):
        self.metrics_enabled = True
        if self.wrapped is not None:
            self.wrapped.apply('enable_metrics')

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

    def train_split(self, data_inputs, expected_outputs) -> (List, List):
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

    def validation_split(self, data_inputs, expected_outputs=None) -> (List, List):
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
