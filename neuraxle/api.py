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
from typing import Dict, List, Tuple, Callable, Union

from neuraxle.base import BaseStep, ExecutionContext, NamedTupleList
from neuraxle.data_container import DataContainer
from neuraxle.metaopt.random import ValidationSplitWrapper
from neuraxle.metrics import MetricsWrapper
from neuraxle.pipeline import MiniBatchSequentialPipeline, Pipeline, CustomPipelineMixin
from neuraxle.steps.data import EpochRepeater, TrainShuffled

VALIDATION_SPLIT_STEP_NAME = 'validation_split_wrapper'
EPOCH_METRICS_STEP_NAME = 'epoch_metrics'
BATCH_METRICS_STEP_NAME = 'batch_metrics'


class DeepLearningPipeline(CustomPipelineMixin, Pipeline):
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
        :class:`EpochRepeater`,
        :class:`ValidationSplitWrapper`,
        :class:`MiniBatchSequentialPipeline`,
        :class:`Pipeline`,
        :class:`CustomPipelineMixin`,
        :class:`MetricsWrapper`
    """

    def __init__(
            self,
            pipeline: Union[BaseStep, NamedTupleList],
            validation_size: float = 0.0,
            batch_size: int = None,
            batch_metrics: Dict[str, Callable] = None,
            shuffle_in_each_epoch_at_train: bool = True,
            seed: int = None,
            n_epochs: int = 1,
            epochs_metrics: Dict[str, Callable] = None,
            scoring_function: Callable = None,
            metrics_plotting_step: BaseStep = None,
            cache_folder: str = None,
            print_epoch_metrics=False,
            print_batch_metrics=False
    ):

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
        self.metrics_plotting_step = metrics_plotting_step
        self.print_batch_metrics = print_batch_metrics
        self.print_epoch_metrics = print_epoch_metrics

        wrapped = pipeline
        wrapped = self._create_mini_batch_pipeline(wrapped)

        if shuffle_in_each_epoch_at_train:
            wrapped = TrainShuffled(wrapped=wrapped, seed=seed)

        wrapped = self._create_validation_split(wrapped)
        wrapped = self._create_epoch_repeater(wrapped)

        BaseStep.__init__(self)
        Pipeline.__init__(self, [wrapped], cache_folder=cache_folder)

    def _create_mini_batch_pipeline(self, wrapped: BaseStep) -> BaseStep:
        """
        Add mini batching and batch metrics by wrapping the step with :class:`MetricsWrapper`, and  :class:̀MiniBatchSequentialPipeline`.

        :param wrapped: pipeline step
        :type wrapped: BaseStep
        :return: wrapped pipeline step
        :rtype: MetricsWrapper
        """
        if self.batch_size is not None:
            wrapped = MetricsWrapper(wrapped=wrapped, metrics=self.batch_metrics, name=BATCH_METRICS_STEP_NAME, print_metrics=self.print_batch_metrics)
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
        if self.validation_size != 0.0:
            wrapped = MetricsWrapper(wrapped=wrapped, metrics=self.epochs_metrics, name=EPOCH_METRICS_STEP_NAME, print_metrics=self.print_epoch_metrics)
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
            wrapped = EpochRepeater(wrapped, epochs=self.n_epochs, fit_only=False)
        return wrapped

    def _did_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Visualize metrics after fit transform if there is a metrics plotting step.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: data container
        """
        self._visualize_metrics()
        return data_container

    def _did_fit(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Visualize metrics after fit if there is a metrics plotting step.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: data container
        """
        self._visualize_metrics()
        return data_container

    def _visualize_metrics(self):
        """
        Visualize epoch metrics, and batch metrics using the metrics plotting step.
        """
        batch_metrics, epoch_metrics = self._get_metrics_results()

        if self.metrics_plotting_step is not None:
            self.metrics_plotting_step.transform((batch_metrics, epoch_metrics))

    def _get_metrics_results(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Get epoch metrics, and batch metrics using :func:`~neuraxle.base.BaseStep.get_step_by_name`.
        """
        batch_metrics = []
        if self.validation_size != 0.0:
            batch_metrics = self.get_step_by_name(BATCH_METRICS_STEP_NAME)

        epoch_metrics = []
        if self.batch_size is not None:
            epoch_metrics = self.get_step_by_name(EPOCH_METRICS_STEP_NAME)

        return batch_metrics, epoch_metrics

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

    def get_batch_metric_train(self, name) -> List:
        """
        Get training batch metric results for a given batch metric name.

        :return: score
        :rtype: float
        """
        return self.get_step_by_name(BATCH_METRICS_STEP_NAME).get_metric_train(name)

    def get_epoch_metric_train(self, name) -> List:
        """
        Get training epoch metric results for a given epoch metric name.

        :return: score
        :rtype: float
        """
        return self.get_step_by_name(EPOCH_METRICS_STEP_NAME).get_metric_train(name)

    def get_batch_metric_validation(self, name) -> List:
        """
        Get validation batch metric for a given metric name.

        :return: score
        :rtype: float
        """
        return self.get_step_by_name(BATCH_METRICS_STEP_NAME).get_metric_validation(name)

    def get_epoch_metric_validation(self, name) -> List:
        """
        Get validation epoch metric for a given metric name.

        :param name:
        :return:
        """
        return self.get_step_by_name(EPOCH_METRICS_STEP_NAME).get_metric_validation(name)

    def get_all_epoch_metrics_train(self) -> Dict:
        """
        Get all training epoch metrics.

        :return: metrics results dict
        :rtype: Dict
        """
        return self.get_step_by_name(EPOCH_METRICS_STEP_NAME).get_all_metrics_train()

    def get_all_batch_metrics_train(self) -> Dict:
        """
        Get all training batch metrics.

        :return: metrics results dict
        :rtype: Dict
        """
        return self.get_step_by_name(BATCH_METRICS_STEP_NAME).get_all_metrics_train()

    def get_all_epoch_metrics_validation(self) -> Dict:
        """
        Get all validation epoch metrics.

        :return: metrics results dict
        :rtype: Dict
        """
        return self.get_step_by_name(EPOCH_METRICS_STEP_NAME).get_all_metrics_validation()

    def get_all_batch_metrics_validation(self) -> Dict:
        """
        Get all validation batch metrics.

        :return: metrics results dict
        :rtype: Dict
        """
        return self.get_step_by_name(BATCH_METRICS_STEP_NAME).get_all_metrics_validation()
