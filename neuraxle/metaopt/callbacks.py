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

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from neuraxle.base import ExecutionContext as CX
from neuraxle.data_container import ARG_Y_EXPECTED, ARG_Y_PREDICTD, EvalEOTDACT
from neuraxle.logging.warnings import warn_deprecated_class
from neuraxle.metaopt.data.aggregates import MetricResults, TrialSplit

OPTIONAL_ARG_CONTEXT = Optional[CX]
RETURNS_SCORE = float


class BaseCallback(ABC):
    """
    Base class for a training callback.
    Callbacks are called after each epoch inside the fit function of the :class:`~neuraxle.metaopt.automl.Trainer`.

    .. seealso::
        :class:`MetaCallback`,
        :class:`~neuraxle.metaopt.auto_ml.AutoML`,
        :class:`~neuraxle.metaopt.auto_ml.Trainer`,
    """

    @abstractmethod
    def call(
        self,
        trial_split: TrialSplit,
        dact_train: EvalEOTDACT,
        dact_valid: EvalEOTDACT,
        is_finished_and_fitted: bool = False
    ) -> bool:
        pass


class EarlyStoppingCallback(BaseCallback):
    """
    Perform early stopping when there is multiple epochs in a row that didn't improve the performance of the model.
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
        dact_train: EvalEOTDACT,
        dact_valid: EvalEOTDACT,
        is_finished_and_fitted: bool = False
    ) -> bool:
        metric_name = self.metric_name or trial_split.main_metric_name
        unmanaged_metric_results: MetricResults = trial_split.metric_result(metric_name)
        validation_scores = unmanaged_metric_results.get_valid_scores()

        if len(validation_scores) > self.n_epochs_without_improvement:
            higher_score_is_better = unmanaged_metric_results.is_higher_score_better()
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
    """

    def __init__(self, wrapped_callback: BaseCallback):
        self.wrapped_callback = wrapped_callback

    @abstractmethod
    def call(
        self,
        trial_split: TrialSplit,
        dact_train: EvalEOTDACT,
        dact_valid: EvalEOTDACT,
        is_finished_and_fitted: bool = False
    ) -> bool:
        return self.wrapped_callback.call(
            trial_split,
            dact_train,
            dact_valid,
            is_finished_and_fitted
        )


class IfBestScore(MetaCallback):
    """
    Meta callback that only execute when the trial is a new best score.

    .. seealso::
        :class:`MetaCallback`,
    """

    def call(
        self,
        trial_split: TrialSplit,
        dact_train: EvalEOTDACT,
        dact_valid: EvalEOTDACT,
        is_finished_and_fitted: bool = False
    ) -> bool:
        if trial_split.metric_result().is_new_best_score():
            return self.wrapped_callback.call(
                trial_split,
                dact_train,
                dact_valid,
                is_finished_and_fitted
            )
        return False


class IfLastStep(MetaCallback):
    """
    Meta callback that only execute when the training is finished or fitted, or when it is the last epoch.

    .. seealso::
        :class:`MetaCallback`,
    """

    def call(
        self,
        trial_split: TrialSplit,
        dact_train: EvalEOTDACT,
        dact_valid: EvalEOTDACT,
        is_finished_and_fitted: bool = False
    ) -> bool:
        if trial_split.epoch == trial_split.n_epochs or is_finished_and_fitted:
            self.wrapped_callback.call(
                trial_split,
                dact_train,
                dact_valid,
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
    """

    def __init__(self, label):
        BaseCallback.__init__(self)
        self.label = label

    def call(
        self,
        trial_split: TrialSplit,
        dact_train: EvalEOTDACT,
        dact_valid: EvalEOTDACT,
        is_finished_and_fitted: bool = False
    ) -> bool:
        # TODO: maybe the trial split could contain the model again. This would imply that the trainer would let the trial split train itself. This is a bit weird, still.
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
        :class:`~neuraxle.metaopt.auto_ml.AutoML`,
        :class:`~neuraxle.metaopt.auto_ml.Trainer`,
        :class:`BaseCallback`,
    """

    def __init__(self, callbacks: List[BaseCallback]):
        self.callbacks: List[BaseCallback] = callbacks

    def __getitem__(self, item):
        return self.callbacks[item]

    def call(
        self,
        trial_split: TrialSplit,
        dact_train: EvalEOTDACT,
        dact_valid: EvalEOTDACT,
        is_finished_and_fitted: bool = False
    ) -> bool:
        for callback in self.callbacks:
            try:
                if callback.call(
                    trial_split=trial_split,
                    dact_train=dact_train,
                    dact_valid=dact_valid,
                    is_finished_and_fitted=is_finished_and_fitted
                ):
                    is_finished_and_fitted = True
            except Exception as e:
                trial_split.context.flow.log_error(e)
                raise e

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
    """

    def __init__(
        self,
        name: str,
        metric_function: Callable[[ARG_Y_EXPECTED, ARG_Y_PREDICTD, OPTIONAL_ARG_CONTEXT], RETURNS_SCORE],
        higher_score_is_better: bool,
        log_metrics=True,
        pass_context_to_metric_function: bool = False
    ):
        # TODO: make a dictionnary somewhere to store predefined metrics?
        self.name: str = name
        self.metric_function: Callable[
            [ARG_Y_EXPECTED, ARG_Y_PREDICTD, OPTIONAL_ARG_CONTEXT], RETURNS_SCORE] = metric_function
        self.higher_score_is_better = higher_score_is_better
        self.log_metrics = log_metrics
        self.pass_context_to_metric_function = pass_context_to_metric_function

    def call(
        self,
        trial_split: TrialSplit,
        dact_train: EvalEOTDACT,
        dact_valid: EvalEOTDACT,
        is_finished_and_fitted: bool = False
    ) -> bool:
        f = self.metric_function

        if self.pass_context_to_metric_function:
            train_score: float = f(dact_train.eo, dact_train.di, trial_split.validation_context())
            if dact_valid is not None:
                valid_score: float = f(dact_valid.eo, dact_valid.di, trial_split.validation_context())
        else:
            train_score: float = f(dact_train.eo, dact_train.di)
            if dact_valid is not None:
                valid_score: float = f(dact_valid.eo, dact_valid.di)

        with trial_split.managed_metric(self.name, self.higher_score_is_better) as metric:
            metric: MetricResults = metric  # just a typing comment here for convenience.

            metric.add_train_result(train_score)
            if dact_valid is not None:
                metric.add_valid_result(valid_score)

        return is_finished_and_fitted


class ScoringCallback(MetricCallback):
    """
    Metric Callback that calculates metric results for the main scoring metric.
    Adds the results into the trial repository.

    .. seealso::
        :class:`MetricCallback`,
    """

    def __init__(
        self,
        metric_function: Callable[[ARG_Y_EXPECTED, ARG_Y_PREDICTD, OPTIONAL_ARG_CONTEXT], RETURNS_SCORE],
        name='main',
        higher_score_is_better: bool = True,
        log_metrics: bool = True,
        pass_context_to_metric_function: bool = False
    ):
        MetricCallback.__init__(
            self,
            name=name,
            metric_function=metric_function,
            higher_score_is_better=higher_score_is_better,
            log_metrics=log_metrics,
            pass_context_to_metric_function=pass_context_to_metric_function
        )
        warn_deprecated_class(self, MetricCallback)
