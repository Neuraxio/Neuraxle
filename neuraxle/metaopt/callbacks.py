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

from neuraxle.data_container import DataContainer
from neuraxle.metaopt.trial import TrialSplit


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
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`~neuraxle.metaopt.auto_ml.InMemoryHyperparamsRepository`,
        :class:`~neuraxle.metaopt.auto_ml.HyperparamsJSONRepository`,
        :class:`~neuraxle.metaopt.auto_ml.BaseHyperparameterSelectionStrategy`,
        :class:`~neuraxle.metaopt.auto_ml.RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.base.HyperparameterSamples`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    @abstractmethod
    def call(self, trial: TrialSplit, epoch_number: int, total_epochs: int, input_train: DataContainer,
             pred_train: DataContainer, input_val: DataContainer, pred_val: DataContainer,
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
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`~neuraxle.metaopt.auto_ml.InMemoryHyperparamsRepository`,
        :class:`~neuraxle.metaopt.auto_ml.HyperparamsJSONRepository`,
        :class:`~neuraxle.metaopt.auto_ml.BaseHyperparameterSelectionStrategy`,
        :class:`~neuraxle.metaopt.auto_ml.RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.base.HyperparameterSamples`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def __init__(self, max_epochs_without_improvement):
        self.n_epochs_without_improvement = max_epochs_without_improvement

    def call(
            self,
            trial: TrialSplit,
            epoch_number: int,
            total_epochs: int,
            input_train: DataContainer,
            pred_train: DataContainer,
            input_val: DataContainer,
            pred_val: DataContainer,
            is_finished_and_fitted: bool
    ):
        validation_scores = trial.get_validation_scores()
        if len(validation_scores) > self.n_epochs_without_improvement:
            higher_score_is_better = trial.is_higher_score_better()
            if validation_scores[-self.n_epochs_without_improvement] >= validation_scores[
                -1] and higher_score_is_better:
                return True
            if validation_scores[-self.n_epochs_without_improvement] <= validation_scores[
                -1] and not higher_score_is_better:
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
        :class:`~neuraxle.metaopt.trial.Trial`,
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
            trial: TrialSplit,
            epoch_number: int,
            total_epochs: int,
            input_train: DataContainer,
            pred_train: DataContainer,
            input_val: DataContainer,
            pred_val: DataContainer,
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
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`~neuraxle.metaopt.auto_ml.InMemoryHyperparamsRepository`,
        :class:`~neuraxle.metaopt.auto_ml.HyperparamsJSONRepository`,
        :class:`~neuraxle.metaopt.auto_ml.BaseHyperparameterSelectionStrategy`,
        :class:`~neuraxle.metaopt.auto_ml.RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.base.HyperparameterSamples`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def call(self, trial: TrialSplit, epoch_number: int, total_epochs: int, input_train: DataContainer,
             pred_train: DataContainer, input_val: DataContainer, pred_val: DataContainer,
             is_finished_and_fitted: bool):
        if trial.is_new_best_score():
            if self.wrapped_callback.call(
                    trial,
                    epoch_number,
                    total_epochs,
                    input_train,
                    pred_train,
                    input_val,
                    pred_val,
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
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`~neuraxle.metaopt.auto_ml.InMemoryHyperparamsRepository`,
        :class:`~neuraxle.metaopt.auto_ml.HyperparamsJSONRepository`,
        :class:`~neuraxle.metaopt.auto_ml.BaseHyperparameterSelectionStrategy`,
        :class:`~neuraxle.metaopt.auto_ml.RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.base.HyperparameterSamples`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def call(self, trial: TrialSplit, epoch_number: int, total_epochs: int, input_train: DataContainer,
             pred_train: DataContainer, input_val: DataContainer, pred_val: DataContainer,
             is_finished_and_fitted: bool):
        if epoch_number == total_epochs - 1 or is_finished_and_fitted:
            self.wrapped_callback.call(
                trial,
                epoch_number,
                total_epochs,
                input_train,
                pred_train,
                input_val,
                pred_val,
                is_finished_and_fitted
            )
            return True
        return False


class StepSaverCallback(BaseCallback):
    """
    Callback that saves the trial model.

    .. seealso::
        :class:`BaseCallback`,
        :class:`MetaCallback`,
        :class:`EarlyStoppingCallback`,
        :class:`IfBestScore`,
        :class:`IfLastStep`,
        :class:`StepSaverCallback`,
        :class:`~neuraxle.metaopt.auto_ml.AutoML`,
        :class:`~neuraxle.metaopt.auto_ml.Trainer`,
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`~neuraxle.metaopt.auto_ml.InMemoryHyperparamsRepository`,
        :class:`~neuraxle.metaopt.auto_ml.HyperparamsJSONRepository`,
        :class:`~neuraxle.metaopt.auto_ml.BaseHyperparameterSelectionStrategy`,
        :class:`~neuraxle.metaopt.auto_ml.RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.base.HyperparameterSamples`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def call(self, trial: TrialSplit, epoch_number: int, total_epochs: int, input_train: DataContainer,
             pred_train: DataContainer, input_val: DataContainer, pred_val: DataContainer,
             is_finished_and_fitted: bool):
        trial.save_model()
        return False


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
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`~neuraxle.metaopt.auto_ml.InMemoryHyperparamsRepository`,
        :class:`~neuraxle.metaopt.auto_ml.HyperparamsJSONRepository`,
        :class:`~neuraxle.metaopt.auto_ml.BaseHyperparameterSelectionStrategy`,
        :class:`~neuraxle.metaopt.auto_ml.RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.base.HyperparameterSamples`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def __init__(self, callbacks, print_func: Callable = None):
        self.callbacks = callbacks
        if print_func is None:
            print_func = print
        self.print_func = print_func

    def __getitem__(self, item):
        return self.callbacks[item]

    def call(self, trial: TrialSplit, epoch_number: int, total_epochs: int, input_train: DataContainer,
             pred_train: DataContainer, input_val: DataContainer, pred_val: DataContainer,
             is_finished_and_fitted: bool):
        is_finished_and_fitted = False
        for callback in self.callbacks:
            try:
                if callback.call(
                        trial=trial,
                        epoch_number=epoch_number,
                        total_epochs=total_epochs,
                        input_train=input_train,
                        pred_train=pred_train,
                        input_val=input_val,
                        pred_val=pred_val,
                        is_finished_and_fitted=is_finished_and_fitted
                ):
                    is_finished_and_fitted = True
            except Exception as error:
                track = traceback.format_exc()
                self.print_func(track)
        return is_finished_and_fitted


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
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`~neuraxle.metaopt.auto_ml.InMemoryHyperparamsRepository`,
        :class:`~neuraxle.metaopt.auto_ml.HyperparamsJSONRepository`,
        :class:`~neuraxle.metaopt.auto_ml.BaseHyperparameterSelectionStrategy`,
        :class:`~neuraxle.metaopt.auto_ml.RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.base.HyperparameterSamples`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def __init__(self, name: str, metric_function: Callable, higher_score_is_better: bool, print_metrics=True,
                 print_function=None):
        self.name = name
        self.metric_function = metric_function
        self.higher_score_is_better = higher_score_is_better
        self.print_metrics = print_metrics
        if print_function is None:
            print_function = print
        self.print_function = print_function

    def call(self, trial: TrialSplit, epoch_number: int, total_epochs: int, input_train: DataContainer,
             pred_train: DataContainer, input_val: DataContainer, pred_val: DataContainer,
             is_finished_and_fitted: bool):
        train_score = self.metric_function(pred_train.data_inputs, pred_train.expected_outputs)
        validation_score = self.metric_function(pred_val.data_inputs, pred_val.expected_outputs)

        trial.add_metric_results_train(
            name=self.name,
            score=train_score,
            higher_score_is_better=self.higher_score_is_better
        )

        trial.add_metric_results_validation(
            name=self.name,
            score=validation_score,
            higher_score_is_better=self.higher_score_is_better
        )

        if self.print_metrics:
            self.print_function('{} train: {}'.format(self.name, train_score))
            self.print_function('{} validation: {}'.format(self.name, validation_score))

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
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`~neuraxle.metaopt.auto_ml.InMemoryHyperparamsRepository`,
        :class:`~neuraxle.metaopt.auto_ml.HyperparamsJSONRepository`,
        :class:`~neuraxle.metaopt.auto_ml.BaseHyperparameterSelectionStrategy`,
        :class:`~neuraxle.metaopt.auto_ml.RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.base.HyperparameterSamples`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def __init__(self, metric_function: Callable, higher_score_is_better: bool):
        super().__init__(
            name='main',
            metric_function=metric_function,
            higher_score_is_better=higher_score_is_better
        )
