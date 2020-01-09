from typing import Dict, List, Tuple

from neuraxle.base import BaseStep, MetaStepMixin, ExecutionContext
from neuraxle.data_container import DataContainer
from neuraxle.metaopt.random import ValidationSplitWrapper
from neuraxle.metrics import MetricsWrapper
from neuraxle.pipeline import MiniBatchSequentialPipeline
from neuraxle.steps.data import EpochRepeater, TrainShuffled

VALIDATION_SPLIT_STEP_NAME = 'validation_split_wrapper'
EPOCH_METRICS_STEP_NAME = 'epoch_metrics'
BATCH_METRICS_STEP_NAME = 'batch_metrics'


class DeepLearningPipeline(MetaStepMixin, BaseStep):
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
            final_scoring_metrics=None,
            metrics_plotting_step=None
    ):
        if epochs_metrics is None:
            epochs_metrics = []
        if batch_metrics is None:
            batch_metrics = []
        if final_scoring_metrics is None:
            final_scoring_metrics = []

        self.final_scoring_metric = scoring_function
        self.final_scoring_metrics = final_scoring_metrics
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
        MetaStepMixin.__init__(self, wrapped)

    def _create_mini_batch_pipeline(self, wrapped):
        if self.batch_size is not None:
            wrapped = MetricsWrapper(wrapped=wrapped, metrics=self.batch_metrics, name=BATCH_METRICS_STEP_NAME)
            wrapped = MiniBatchSequentialPipeline(
                [wrapped],
                batch_size=self.batch_size
            )

        return wrapped

    def _create_validation_split(self, wrapped):
        if self.validation_size != 0.0:
            wrapped = MetricsWrapper(wrapped=wrapped, metrics=self.final_scoring_metrics, name=EPOCH_METRICS_STEP_NAME)
            wrapped = ValidationSplitWrapper(
                wrapped=wrapped,
                test_size=self.validation_size,
                scoring_function=self.final_scoring_metric
            ).set_name(VALIDATION_SPLIT_STEP_NAME)

        return wrapped

    def _create_epoch_repeater(self, wrapped):
        if self.n_epochs is not None:
            wrapped = EpochRepeater(wrapped, epochs=self.n_epochs)
        return wrapped

    def _did_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        self._visualize_metrics()
        return data_container

    def _did_fit(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        self._visualize_metrics()
        return data_container

    def _visualize_metrics(self):
        batch_metrics, epoch_metrics = self._get_metrics_results()

        if self.metrics_plotting_step is not None:
            self.metrics_plotting_step.transform((batch_metrics, epoch_metrics))

    def _get_metrics_results(self) -> Tuple[List[Dict], List[Dict]]:
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

