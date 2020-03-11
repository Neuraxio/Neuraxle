import traceback
from abc import ABC, abstractmethod
from typing import Callable

from neuraxle.data_container import DataContainer
from neuraxle.metaopt.trial import Trial


class BaseCallback(ABC):
    @abstractmethod
    def call(self, trial: Trial, epoch_number: int, total_epochs: int, input_train: DataContainer,
             pred_train: DataContainer, input_val: DataContainer, pred_val: DataContainer, is_finished_and_fitted: bool):
        pass


class BaseRefitCallback(ABC):
    @abstractmethod
    def call(self, scores):
        pass


class EarlyStoppingCallback(BaseCallback):
    """
    Perform early stopping when there is multiple epochs in a row that didn't improve the performance of the model.

    Example usage :

    .. code-block:: python

        trainer = Trainer(
            metrics=self.metrics,
            callbacks=self.callbacks,
            score=self.scoring_function,
            epochs=self.epochs
        )

        trial = trainer.execute_trial(
            p=p,
            trial_repository=repo_trial,
            train_data_container=training_data_container,
            validation_data_container=validation_data_container,
            context=context
        )

        pipeline = trainer.refit(repo_trial.pipeline, data_container, context)


    .. seealso::
        :class:`AutoML`,
        :class:`Trial`,
        :class:`HyperparamsRepository`,
        :class:`HyperparameterOptimizer`,
        :class:`RandomSearchHyperparameterOptimizer`,
        :class:`DataContainer`
    """

    def __init__(self, max_epochs_without_improvement):
        self.n_epochs_without_improvement = max_epochs_without_improvement

    def call(
            self,
            trial: Trial,
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
            higher_score_is_better = trial.get_higher_score_is_better()
            if validation_scores[-self.n_epochs_without_improvement] >= validation_scores[-1] and higher_score_is_better:
                return True
            if validation_scores[-self.n_epochs_without_improvement] <= validation_scores[-1] and not higher_score_is_better:
                return True
        return False


class MetaCallback(BaseCallback):
    def __init__(self, wrapped_callback: BaseCallback):
        self.wrapped_callback = wrapped_callback

    @abstractmethod
    def call(
            self,
            trial: Trial,
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
    def call(self, trial: Trial, epoch_number: int, total_epochs: int, input_train: DataContainer,
             pred_train: DataContainer, input_val: DataContainer, pred_val: DataContainer, is_finished_and_fitted: bool):
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
    def call(self, trial: Trial, epoch_number: int, total_epochs: int, input_train: DataContainer, pred_train: DataContainer, input_val: DataContainer, pred_val: DataContainer, is_finished_and_fitted: bool):
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


class StepSaverCallback(MetaCallback):
    def call(self, trial: Trial, epoch_number: int, total_epochs: int, input_train: DataContainer,
             pred_train: DataContainer, input_val: DataContainer, pred_val: DataContainer, is_finished_and_fitted: bool):
        trial.save_model()
        return False


class CallbackList(BaseCallback):
    def __init__(self, callbacks, print_func: Callable = None):
        self.callbacks = callbacks
        if print_func is None:
            print_func = print
        self.print_func = print_func

    def call(self, trial: Trial, epoch_number: int, total_epochs: int, input_train: DataContainer,
             pred_train: DataContainer, input_val: DataContainer, pred_val: DataContainer, is_finished_and_fitted: bool):
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
    def __init__(self, name: str, metric_function: Callable, higher_score_is_better: bool):
        self.name = name
        self.metric_function = metric_function
        self.higher_score_is_better = higher_score_is_better

    def call(self, trial: Trial, epoch_number: int, total_epochs: int, input_train: DataContainer,
             pred_train: DataContainer, input_val: DataContainer, pred_val: DataContainer, is_finished_and_fitted: bool):
        train_score = self.metric_function(pred_train.data_inputs, pred_train.expected_outputs)
        validation_score = self.metric_function(pred_val.data_inputs, pred_val.expected_outputs)

        trial.set_metric_results_train(
            name=self.name,
            score=train_score,
            higher_score_is_better=self.higher_score_is_better
        )

        trial.set_metric_results_validation(
            name=self.name,
            score=validation_score,
            higher_score_is_better=self.higher_score_is_better
        )

        return False


class ScoringCallback(MetricCallback):
    def __init__(self, metric_function: Callable, higher_score_is_better: bool):
        super().__init__(
            name='main',
            metric_function=metric_function,
            higher_score_is_better=higher_score_is_better
        )


class EarlyStoppingRefitCallback(BaseRefitCallback):
    """
    Perform early stopping when there is multiple epochs in a row that didn't improve the performance of the model.

    Example usage :

    .. code-block:: python

        trainer = Trainer(
            metrics=self.metrics,
            callbacks=self.callbacks,
            refit_callbacks=self.callbacks,
            score=self.scoring_function,
            epochs=self.epochs
        )

        trial = trainer.fit(
            p=p,
            trial_repository=repo_trial,
            train_data_container=training_data_container,
            validation_data_container=validation_data_container,
            context=context
        )

        pipeline = trainer.refit(repo_trial.pipeline, data_container, context)

    .. seealso::
        :class:`AutoML`,
        :class:`Trial`,
        :class:`HyperparamsRepository`,
        :class:`HyperparameterOptimizer`,
        :class:`RandomSearchHyperparameterOptimizer`,
        :class:`DataContainer`
    """

    def __init__(self, n_epochs_without_improvement, higher_score_is_better):
        self.higher_score_is_better = higher_score_is_better
        self.n_epochs_without_improvement = n_epochs_without_improvement

    def call(self, scores):
        if len(scores) > self.n_epochs_without_improvement:
            if scores[-self.n_epochs_without_improvement] >= scores[-1] and self.higher_score_is_better:
                return True
            if scores[-self.n_epochs_without_improvement] <= scores[-1] and not self.higher_score_is_better:
                return True
        return False