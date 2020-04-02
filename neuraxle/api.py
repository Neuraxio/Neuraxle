"""
Neuraxle's API classes
========================================
Neuraxle's high-level API classes. Useful to make complex Deep Learning pipelines by calling just a few minimal things.

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
from typing import Dict, Callable, Union

from neuraxle.base import BaseStep, NamedTupleList, EvaluableStepMixin, ForceHandleMixin, MetaStepMixin
from neuraxle.metaopt.random import ValidationSplitWrapper
from neuraxle.metrics import MetricsWrapper
from neuraxle.pipeline import MiniBatchSequentialPipeline
from neuraxle.steps.data import EpochRepeater, TrainShuffled

VALIDATION_SPLIT_STEP_NAME = 'validation_split_wrapper'
EPOCH_METRICS_STEP_NAME = 'epoch_metrics'
BATCH_METRICS_STEP_NAME = 'batch_metrics'


class DeepLearningPipeline(EvaluableStepMixin, ForceHandleMixin, MetaStepMixin, BaseStep):
    """
    Adds an epoch loop, a validation split, and mini batching to a pipeline.
    It also tracks batch metrics, and epoch metrics.


    Example usage :

    .. code-block:: python

        p = DeepLearningPipeline(
            pipeline,
            validation_size=VALIDATION_SIZE,
            batch_size=BATCH_SIZE,
            batch_metrics={'mse': to_numpy_metric_wrapper(mean_squared_error)},
            shuffle_in_each_epoch_at_train=True,
            n_epochs=N_EPOCHS,
            epochs_metrics={'mse': to_numpy_metric_wrapper(mean_squared_error)},
            scoring_function=to_numpy_metric_wrapper(mean_squared_error),
        )

        p, outputs = p.fit_transform(data_inputs, expected_outputs)

        batch_mse_train = p.get_batch_metric_train('mse')
        epoch_mse_train = p.get_epoch_metric_train('mse')
        batch_mse_validation = p.get_batch_metric_validation('mse')
        epoch_mse_validation = p.get_epoch_metric_validation('mse')

    It uses :class:`EpochRepeater`, :class:`ValidationSplitWrapper`, and :class:`MiniBatchSequentialPipeline`

    .. seealso::
        :class:`~neuraxle.steps.data.EpochRepeater`,
        :class:`~neuraxle.metaopt.random.ValidationSplitWrapper`,
        :class:`~neuraxle.pipeline.MiniBatchSequentialPipeline`,
        :class:`~neuraxle.pipeline.Pipeline`,
        :class:`~neuraxle.pipeline.CustomPipelineMixin`,
        :class:`~neuraxle.metrics.MetricsWrapper`
    """

    def __init__(
            self,
            pipeline: Union[BaseStep, NamedTupleList],
            validation_size: float = None,
            batch_size: int = None,
            batch_metrics: Dict[str, Callable] = None,
            shuffle_in_each_epoch_at_train: bool = True,
            seed: int = None,
            n_epochs: int = 1,
            epochs_metrics: Dict[str, Callable] = None,
            scoring_function: Callable = None,
            cache_folder: str = None,
            print_epoch_metrics=False,
            print_batch_metrics=False
    ):
        """
        :param pipeline: pipeline to wrap with an epoch repeater, a validation split wrapper, and a mini batch sequential pipeline
        :param validation_size: ratio for validation size between 0 and 1
        :param batch_size: batch size for the mini batch sequential pipeline
        :param batch_metrics: metrics to calculate for each processed mini batch
        :param shuffle_in_each_epoch_at_train:
        :param seed: random seed for the data shuffling that can be done at each epoch when the param shuffle_in_each_epoch_at_train is True
        :param n_epochs: number of epochs
        :param epochs_metrics: metrics to calculate for each epoch
        :param scoring_function: scoring function with two arguments (y_true, y_pred)
        :param cache_folder: cache folder to be used inside the pipeline
        :param print_epoch_metrics: whether or not to print epoch metrics
        :param print_batch_metrics: whether or not to print batch metrics
        """
        if epochs_metrics is None:
            epochs_metrics = {}
        if batch_metrics is None:
            batch_metrics = {}

        self.final_scoring_metric = scoring_function
        self.epochs_metrics = epochs_metrics
        self.n_epochs = n_epochs
        self.shuffle_in_each_epoch_at_train = shuffle_in_each_epoch_at_train
        self.batch_size = batch_size
        self.batch_metrics = batch_metrics
        self.validation_size = validation_size
        self.print_batch_metrics = print_batch_metrics
        self.print_epoch_metrics = print_epoch_metrics

        wrapped = pipeline
        wrapped = self._create_mini_batch_pipeline(wrapped)

        if shuffle_in_each_epoch_at_train:
            wrapped = TrainShuffled(wrapped=wrapped, seed=seed)

        wrapped = self._create_validation_split(wrapped)
        wrapped = self._create_epoch_repeater(wrapped)

        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        EvaluableStepMixin.__init__(self)
        ForceHandleMixin.__init__(self, cache_folder)

    def _create_mini_batch_pipeline(self, wrapped: BaseStep) -> BaseStep:
        """
        Add mini batching and batch metrics by wrapping the step with :class:`MetricsWrapper`, and  :class:̀MiniBatchSequentialPipeline`.

        :param wrapped: pipeline step
        :type wrapped: BaseStep
        :return: wrapped pipeline step
        :rtype: MetricsWrapper
        """
        if self.batch_size is not None:
            wrapped = MetricsWrapper(wrapped=wrapped, metrics=self.batch_metrics, name=BATCH_METRICS_STEP_NAME,
                                     print_metrics=self.print_batch_metrics)
            wrapped = MiniBatchSequentialPipeline(
                [wrapped],
                batch_size=self.batch_size
            )

        return wrapped

    def _create_validation_split(self, wrapped: BaseStep) -> BaseStep:
        """
        Add validation split and epoch metrics by wrapping the step with :class:`MetricsWrapper`, and  :class:̀ValidationSplitWrapper`.

        :param wrapped: pipeline step
        :type wrapped: BaseStep
        :return: wrapped pipeline step
        :rtype: MetricsWrapper
        """
        wrapped = MetricsWrapper(
            wrapped=wrapped,
            metrics=self.epochs_metrics,
            name=EPOCH_METRICS_STEP_NAME,
            print_metrics=self.print_epoch_metrics
        )

        if self.validation_size is not None:
            wrapped = ValidationSplitWrapper(
                wrapped=wrapped,
                test_size=self.validation_size,
                scoring_function=self.final_scoring_metric
            ).set_name(VALIDATION_SPLIT_STEP_NAME)

        return wrapped

    def _create_epoch_repeater(self, wrapped: BaseStep) -> BaseStep:
        """
        Add epoch loop by wrapping the step with :class:`EpochRepeater`.

        :param wrapped: pipeline step
        :type wrapped: BaseStep
        :return: wrapped pipeline step
        :rtype: BaseStep
        """
        if self.n_epochs is not None:
            wrapped = EpochRepeater(wrapped, epochs=self.n_epochs, repeat_in_test_mode=False)
        return wrapped

    def get_score(self):
        """
        Get latest score. This function had to be defined for the hyperparameter optimization steps.

        :return: score
        :rtype: float
        """
        return self.get_step_by_name(VALIDATION_SPLIT_STEP_NAME).get_score()

    def get_score_validation(self):
        """
        Get latest score validation.

        :return: score
        :rtype: float
        """
        return self.get_step_by_name(VALIDATION_SPLIT_STEP_NAME).get_score_validation()

    def get_score_train(self) -> float:
        """
        Get latest score train.

        :return: score
        :rtype: float
        """
        return self.get_step_by_name(VALIDATION_SPLIT_STEP_NAME).get_score_train()

