from typing import Dict, List, Tuple

from neuraxle.base import BaseStep, ExecutionContext
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
        :class:`CustomPipelineMixin`
    """
    def __init__(
            self,
            pipeline,
            validation_size=0.0,
            batch_size=None,
            batch_metrics=None,
            shuffle_in_each_epoch_at_train=True,
            seed=None,
            n_epochs=1,
            epochs_metrics=None,
            scoring_function=None,
            metrics_plotting_step=None,
            cache_folder=None
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
        self.pipeline = pipeline
        self.metrics_plotting_step = metrics_plotting_step

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
            wrapped = MetricsWrapper(wrapped=wrapped, metrics=self.batch_metrics, name=BATCH_METRICS_STEP_NAME)
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
            wrapped = MetricsWrapper(wrapped=wrapped, metrics=self.epochs_metrics, name=EPOCH_METRICS_STEP_NAME)
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
        return self.get_step_by_name(VALIDATION_SPLIT_STEP_NAME).get_score()

    def get_score_validation(self):
        return self.get_step_by_name(VALIDATION_SPLIT_STEP_NAME).get_score_validation()

    def get_score_train(self):
        return self.get_step_by_name(VALIDATION_SPLIT_STEP_NAME).get_score_train()

    def get_batch_metric_train(self, name):
        return self.get_step_by_name(BATCH_METRICS_STEP_NAME).get_metric_train(name)

    def get_epoch_metric_train(self, name):
        return self.get_step_by_name(EPOCH_METRICS_STEP_NAME).get_metric_train(name)

    def get_batch_metric_validation(self, name):
        return self.get_step_by_name(BATCH_METRICS_STEP_NAME).get_metric_validation(name)

    def get_epoch_metric_validation(self, name):
        return self.get_step_by_name(EPOCH_METRICS_STEP_NAME).get_metric_validation(name)
