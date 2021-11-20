"""
Neuraxle's training callbacks classes.
=========================================
Training callback classes.

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

import traceback
from abc import ABC, abstractmethod
from typing import Callable

import warnings
from neuraxle.base import BaseStep, BaseTransformer, ExecutionContext, MixinForBaseTransformer
from neuraxle.logging.warnings import warn_deprecated_class, warn_deprecated_arg
from neuraxle.data_container import DataContainer
from neuraxle.metaopt.data.trial import TrialSplit


class BaseCallback(ABC):
    """
    Base class for a training callback.
    Callbacks are called after each epoch inside the fit function of the :class:`~neuraxle.metaopt.automl.Trainer`.

    .. seealso::
        :class:`MetaCallback`,
        :class:`EarlyStoppingCallback`,
        :class:`IfBestScore`,
        :class:`IfLastStep`,
        :class:`StepSaverCallback`,
        :class:`~neuraxle.metaopt.auto_ml.AutoML`,
        :class:`~neuraxle.metaopt.auto_ml.Trainer`,
        :class:`~neuraxle.metaopt.data.trial.Trial`,
        :class:`~neuraxle.metaopt.auto_ml.InMemoryHyperparamsRepository`,
        :class:`~neuraxle.metaopt.auto_ml.HyperparamsJSONRepository`,
        :class:`~neuraxle.metaopt.auto_ml.BaseHyperparameterSelectionStrategy`,
        :class:`~neuraxle.metaopt.auto_ml.RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.base.HyperparameterSamples`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    @abstractmethod
    def call(self, trial_split: TrialSplit, epoch_number: int, total_epochs: int, input_train: DataContainer,
             pred_train: DataContainer, input_val: DataContainer, pred_val: DataContainer, context: ExecutionContext,
             is_finished_and_fitted: bool):
        pass


class EarlyStoppingCallback(BaseCallback):
    """
    Perform early stopping when there is multiple epochs in a row that didn't improve the performance of the model.

    .. seealso::
        :class:`BaseCallback`,
        :class:`MetaCallback`,
        :class:`IfBestScore`,
        :class:`IfLastStep`,
        :class:`StepSaverCallback`,
        :class:`~neuraxle.metaopt.auto_ml.AutoML`,
        :class:`~neuraxle.metaopt.auto_ml.Trainer`,
        :class:`~neuraxle.metaopt.data.trial.Trial`,
        :class:`~neuraxle.metaopt.auto_ml.InMemoryHyperparamsRepository`,
        :class:`~neuraxle.metaopt.auto_ml.HyperparamsJSONRepository`,
        :class:`~neuraxle.metaopt.auto_ml.BaseHyperparameterSelectionStrategy`,
        :class:`~neuraxle.metaopt.auto_ml.RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.base.HyperparameterSamples`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def __init__(self, max_epochs_without_improvement, metric_name=None):
        """

        :param max_epochs_without_improvement: The number of step without improvement on the validation score before an early stopping is triggered.
        :param metric_name: The name of the metric on which we want to condition the early stopping. If None, the main metric will be used.
        """
        self.n_epochs_without_improvement = max_epochs_without_improvement
        self.metric_name = None

    def call(
            self,
            trial_split: TrialSplit,
            epoch_number: int,
            total_epochs: int,
            input_train: DataContainer,
            pred_train: DataContainer,
            input_val: DataContainer,
            pred_val: DataContainer,
            context: ExecutionContext,
            is_finished_and_fitted: bool
    ):
        if self.metric_name is None:
            validation_scores = trial_split.get_validation_scores()
        else:
            validation_scores = trial_split.get_metric_validation_results(self.metric_name)

        if len(validation_scores) > self.n_epochs_without_improvement:
            higher_score_is_better = trial_split.is_higher_score_better()
            if (higher_score_is_better) and \
                    all(validation_scores[-self.n_epochs_without_improvement] >= v for v in
                        validation_scores[-self.n_epochs_without_improvement:]):
                return True
            elif (not higher_score_is_better) and \
                    all(validation_scores[-self.n_epochs_without_improvement] <= v for v in
                        validation_scores[-self.n_epochs_without_improvement:]):
                return True
        return False


class MetaCallback(BaseCallback):
    """
    Meta callback wraps another callback.
    It can be useful to test conditions before executing certain callbacks.

    .. seealso::
        :class:`BaseCallback`,
        :class:`IfBestScore`,
        :class:`IfLastStep`,
        :class:`StepSaverCallback`,
        :class:`~neuraxle.metaopt.auto_ml.AutoML`,
        :class:`~neuraxle.metaopt.auto_ml.Trainer`,
        :class:`~neuraxle.metaopt.data.trial.Trial`,
        :class:`~neuraxle.metaopt.auto_ml.InMemoryHyperparamsRepository`,
        :class:`~neuraxle.metaopt.auto_ml.HyperparamsJSONRepository`,
        :class:`~neuraxle.metaopt.auto_ml.BaseHyperparameterSelectionStrategy`,
        :class:`~neuraxle.metaopt.auto_ml.RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.base.HyperparameterSamples`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def __init__(self, wrapped_callback: BaseCallback):
        self.wrapped_callback = wrapped_callback

    @abstractmethod
    def call(
            self,
            trial_split: TrialSplit,
            epoch_number: int,
            total_epochs: int,
            input_train: DataContainer,
            pred_train: DataContainer,
            input_val: DataContainer,
            pred_val: DataContainer,
            context: ExecutionContext,
            is_finished_and_fitted: bool
    ):
        pass


class IfBestScore(MetaCallback):
    """
    Meta callback that only execute when the trial is a new best score.

    .. seealso::
        :class:`BaseCallback`,
        :class:`MetaCallback`,
        :class:`IfBestScore`,
        :class:`IfLastStep`,
        :class:`StepSaverCallback`,
        :class:`~neuraxle.metaopt.auto_ml.AutoML`,
        :class:`~neuraxle.metaopt.auto_ml.Trainer`,
        :class:`~neuraxle.metaopt.data.trial.Trial`,
        :class:`~neuraxle.metaopt.auto_ml.InMemoryHyperparamsRepository`,
        :class:`~neuraxle.metaopt.auto_ml.HyperparamsJSONRepository`,
        :class:`~neuraxle.metaopt.auto_ml.BaseHyperparameterSelectionStrategy`,
        :class:`~neuraxle.metaopt.auto_ml.RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.base.HyperparameterSamples`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def call(self, trial_split: TrialSplit, epoch_number: int, total_epochs: int, input_train: DataContainer,
             pred_train: DataContainer, input_val: DataContainer, pred_val: DataContainer, context: ExecutionContext,
             is_finished_and_fitted: bool):
        if trial_split.is_new_best_score():
            if self.wrapped_callback.call(
                    trial_split,
                    epoch_number,
                    total_epochs,
                    input_train,
                    pred_train,
                    input_val,
                    pred_val,
                    context,
                    is_finished_and_fitted
            ):
                return True
        return False


class IfLastStep(MetaCallback):
    """
    Meta callback that only execute when the training is finished or fitted, or when it is the last epoch.

    .. seealso::
        :class:`BaseCallback`,
        :class:`MetaCallback`,
        :class:`IfBestScore`,
        :class:`StepSaverCallback`,
        :class:`~neuraxle.metaopt.auto_ml.AutoML`,
        :class:`~neuraxle.metaopt.auto_ml.Trainer`,
        :class:`~neuraxle.metaopt.data.trial.Trial`,
        :class:`~neuraxle.metaopt.auto_ml.InMemoryHyperparamsRepository`,
        :class:`~neuraxle.metaopt.auto_ml.HyperparamsJSONRepository`,
        :class:`~neuraxle.metaopt.auto_ml.BaseHyperparameterSelectionStrategy`,
        :class:`~neuraxle.metaopt.auto_ml.RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.base.HyperparameterSamples`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def call(self, trial_split: TrialSplit, epoch_number: int, total_epochs: int, input_train: DataContainer,
             pred_train: DataContainer, input_val: DataContainer, pred_val: DataContainer, context: ExecutionContext,
             is_finished_and_fitted: bool):
        if epoch_number == total_epochs - 1 or is_finished_and_fitted:
            self.wrapped_callback.call(
                trial_split,
                epoch_number,
                total_epochs,
                input_train,
                pred_train,
                input_val,
                pred_val,
                context,
                is_finished_and_fitted
            )
            return True
        return False


class StepSaverCallback(BaseCallback):
    """
    Callback that saves the trial model.

    .. seealso::
        :class:`BaseCallback`,
        :class:`~neuraxle.metaopt.auto_ml.AutoML`,
        :class:`~neuraxle.metaopt.auto_ml.Trainer`,
        :class:`~neuraxle.metaopt.data.trial.Trial`,
        :class:`~neuraxle.metaopt.auto_ml.InMemoryHyperparamsRepository`,
        :class:`~neuraxle.metaopt.auto_ml.HyperparamsJSONRepository`,
        :class:`~neuraxle.metaopt.auto_ml.BaseHyperparameterSelectionStrategy`,
        :class:`~neuraxle.metaopt.auto_ml.RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.base.HyperparameterSamples`,
    """

    def __init__(self, label):
        BaseCallback.__init__(self)
        self.label = label

    def call(self, trial_split: TrialSplit, epoch_number: int, total_epochs: int, input_train: DataContainer,
             pred_train: DataContainer, input_val: DataContainer, pred_val: DataContainer, context: ExecutionContext,
             is_finished_and_fitted: bool):
        trial_split.save_model(self.label)
        return False


class BestModelCheckpoint(IfBestScore):
    """
    Saves the pipeline model in a folder named "best" when the a new best validation score is reached.
    It is important to note that when refit=True, an AutoML loop will overwrite the best model after refitting.
    """

    def __init__(self):
        IfBestScore.__init__(self, StepSaverCallback('best'))


class CallbackList(BaseCallback):
    """
    Callback list that be executed.

    .. seealso::
        :class:`BaseCallback`,
        :class:`MetaCallback`,
        :class:`EarlyStoppingCallback`,
        :class:`IfBestScore`,
        :class:`IfLastStep`,
        :class:`StepSaverCallback`,
        :class:`~neuraxle.metaopt.auto_ml.AutoML`,
        :class:`~neuraxle.metaopt.auto_ml.Trainer`,
        :class:`~neuraxle.metaopt.data.trial.Trial`,
        :class:`~neuraxle.metaopt.auto_ml.InMemoryHyperparamsRepository`,
        :class:`~neuraxle.metaopt.auto_ml.HyperparamsJSONRepository`,
        :class:`~neuraxle.metaopt.auto_ml.BaseHyperparameterSelectionStrategy`,
        :class:`~neuraxle.metaopt.auto_ml.RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.base.HyperparameterSamples`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def __init__(self, callbacks):
        self.callbacks = callbacks

    def __getitem__(self, item):
        return self.callbacks[item]

    def call(self, context: ExecutionContext, epoch: int, tot_epochs: int, input_train: DataContainer,
             pred_train: DataContainer, input_val: DataContainer, pred_val: DataContainer,
             is_finished_and_fitted: bool = False):
        is_finished_and_fitted = False
        for callback in self.callbacks:
            try:
                if callback.call(
                        trial_split=context,
                        epoch_number=epoch,
                        total_epochs=tot_epochs,
                        input_train=input_train,
                        pred_train=pred_train,
                        input_val=input_val,
                        pred_val=pred_val,
                        context=context,
                        is_finished_and_fitted=is_finished_and_fitted
                ):
                    is_finished_and_fitted = True
            except Exception as _:
                track = traceback.format_exc()
                context.trial.logger.error(track)

        return is_finished_and_fitted

    def append(self, callback: BaseCallback) -> 'CallbackList':
        if callback is not None:
            self.callbacks.append(callback)
        return self


class MetricCallback(BaseCallback):
    """
    Callback that calculates metric results.
    Adds the results into the trial repository.

    .. seealso::
        :class:`BaseCallback`,
        :class:`MetaCallback`,
        :class:`EarlyStoppingCallback`,
        :class:`IfBestScore`,
        :class:`IfLastStep`,
        :class:`StepSaverCallback`,
        :class:`CallbackList`,
        :class:`~neuraxle.metaopt.auto_ml.AutoML`,
        :class:`~neuraxle.metaopt.auto_ml.Trainer`,
        :class:`~neuraxle.metaopt.data.trial.Trial`,
        :class:`~neuraxle.metaopt.auto_ml.InMemoryHyperparamsRepository`,
        :class:`~neuraxle.metaopt.auto_ml.HyperparamsJSONRepository`,
        :class:`~neuraxle.metaopt.auto_ml.BaseHyperparameterSelectionStrategy`,
        :class:`~neuraxle.metaopt.auto_ml.RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.base.HyperparameterSamples`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def __init__(self, name: str, metric_function: Callable, higher_score_is_better: bool, log_metrics=True,
                 pass_context_to_metric_function: bool = False):
        self.name = name
        self.metric_function = metric_function
        self.higher_score_is_better = higher_score_is_better
        self.log_metrics = log_metrics
        self.pass_context_to_metric_function = pass_context_to_metric_function

    def call(self, trial_split: TrialSplit, epoch_number: int, total_epochs: int, input_train: DataContainer,
             pred_train: DataContainer, input_val: DataContainer, pred_val: DataContainer, context: ExecutionContext,
             is_finished_and_fitted: bool):

        if self.pass_context_to_metric_function:
            train_score = self.metric_function(pred_train.expected_outputs, pred_train.data_inputs, context=context)
            validation_score = self.metric_function(pred_val.expected_outputs, pred_val.data_inputs, context=context)
        else:
            train_score = self.metric_function(pred_train.expected_outputs, pred_train.data_inputs)
            validation_score = self.metric_function(pred_val.expected_outputs, pred_val.data_inputs)

        trial_split.add_metric_results_train(
            name=self.name,
            score=train_score,
            higher_score_is_better=self.higher_score_is_better,
            log_metric=self.log_metrics
        )

        trial_split.add_metric_results_validation(
            name=self.name,
            score=validation_score,
            higher_score_is_better=self.higher_score_is_better,
            log_metric=self.log_metrics
        )

        return False


class ScoringCallback(MetricCallback):
    """
    Metric Callback that calculates metric results for the main scoring metric.
    Adds the results into the trial repository.

    .. seealso::
        :class:`BaseCallback`,
        :class:`MetaCallback`,
        :class:`EarlyStoppingCallback`,
        :class:`IfBestScore`,
        :class:`IfLastStep`,
        :class:`StepSaverCallback`,
        :class:`CallbackList`,
        :class:`~neuraxle.metaopt.auto_ml.AutoML`,
        :class:`~neuraxle.metaopt.auto_ml.Trainer`,
        :class:`~neuraxle.metaopt.data.trial.Trial`,
        :class:`~neuraxle.metaopt.auto_ml.InMemoryHyperparamsRepository`,
        :class:`~neuraxle.metaopt.auto_ml.HyperparamsJSONRepository`,
        :class:`~neuraxle.metaopt.auto_ml.BaseHyperparameterSelectionStrategy`,
        :class:`~neuraxle.metaopt.auto_ml.RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.base.HyperparameterSamples`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def __init__(self, metric_function: Callable, name='main', higher_score_is_better: bool = True,
                 log_metrics: bool = True, pass_context_to_metric_function: bool = False):
        MetricCallback.__init__(
            self,
            name=name,
            metric_function=metric_function,
            higher_score_is_better=higher_score_is_better,
            log_metrics=log_metrics,
            pass_context_to_metric_function=pass_context_to_metric_function
        )
        warn_deprecated_class(self, MetricCallback)
