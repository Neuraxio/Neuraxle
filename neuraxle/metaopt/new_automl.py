import copy
import datetime
import hashlib
import json
import traceback
from abc import ABC, abstractmethod
from typing import Callable, Dict, List

from neuraxle.base import BaseStep, ExecutionContext, ForceHandleOnlyMixin
from neuraxle.data_container import DataContainer
from neuraxle.metaopt.auto_ml import Trials, TRIAL_STATUS, AutoMLContainer
from neuraxle.metaopt.random import BaseValidation


class HyperparamsRepository(ABC):
    """
    Hyperparams repository that saves hyperparams, and scores for every AutoML trial.

    .. seealso::
        :class:`AutoMLSequentialWrapper`
        :class:`AutoMLAlgorithm`,
        :class:`BaseValidation`,
        :class:`RandomSearchBaseAutoMLStrategy`,
        :class:`HyperparameterSpace`,
        :class:`HyperparameterSamples`
    """

    @abstractmethod
    def load_all_trials(self, status: 'TRIAL_STATUS') -> 'Trials':
        """
        Load all hyperparameter trials with their corresponding score.
        Sorted by creation date.

        :return: Trials (hyperparams, scores)
        """
        pass

    @abstractmethod
    def save_trial(self, trial: 'Trial'):
        pass

    @abstractmethod
    def save_model(self, step: BaseStep):
        pass

    @abstractmethod
    def new_experiment(self, auto_ml_container: AutoMLContainer):
        """
        Save hyperparams, and score for a failed trial.

        :return: (hyperparams, scores)
        """
        pass

    def _get_trial_hash(self, hp_dict):
        current_hyperparameters_hash = hashlib.md5(str.encode(str(hp_dict))).hexdigest()
        return current_hyperparameters_hash


class InMemoryHyperparamsRepository(HyperparamsRepository):
    """
    In memory hyperparams repository that can print information about trials.
    Useful for debugging.

    Example usage :

    .. code-block:: python

        InMemoryHyperparamsRepository(print_func=print)

    .. seealso::
        :class:`HyperparamsRepository`,
        :class:`HyperparameterSamples`,
        :class:`AutoMLSequentialWrapper`
    """

    def __init__(self, hyperparameter_optimizer, print_func: Callable = None, cache_folder: str = None):
        HyperparamsRepository.__init__(self)
        if print_func is None:
            print_func = print
        self.print_func = print_func
        self.cache_folder = cache_folder

        self.trials = Trials()
        self.hyperparameter_optimizer = hyperparameter_optimizer

    def load_all_trials(self, status: 'TRIAL_STATUS' = None) -> 'Trials':
        return self.trials.filter(status)

    def save_trial(self, trial: 'Trial'):
        self.print_func(trial)
        self.trials.append(trial)

    def new_experiment(self, auto_ml_container: AutoMLContainer):
        hyperparams = self.hyperparameter_optimizer.find_next_best_hyperparams(auto_ml_container)
        self.print_func('new trial:\n{}'.format(json.dumps(hyperparams.to_nested_dict(), sort_keys=True, indent=4)))

        return Trial(hyperparams)

    def save_model(self, step: BaseStep):
        trial_hash = self._get_trial_hash(step.get_hyperparams().to_flat_as_dict_primitive())
        step.set_name(trial_hash).save(full_dump=True)
        return step


class Trial:
    def __init__(self, hyperparams):
        self.status = None
        self.hyperparams = hyperparams

    def set_failed_trial(self, error):
        self.status = TRIAL_STATUS.FAILED
        self.error = error
        self.error_traceback = traceback.format_exc()

    def __enter__(self):
        self.start_time = datetime.datetime.now()
        self.status = TRIAL_STATUS.PLANNED
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.datetime.now()
        self.status = TRIAL_STATUS.SUCCESS
        self.status = TRIAL_STATUS.FAILED
        return self


class Trainer:
    def __init__(self, p, score, metrics, callbacks, trial_repository):
        self.p = p

        self.score = score

        self.metrics = metrics
        self.callbacks = callbacks

        self.metrics_results_train = None
        self.metrics_results_validation = None

        self.trial_repository = trial_repository

    def train(self, train_data_container, validation_data_container, epochs, context: ExecutionContext):
        early_stopping = False

        for i in range(epochs):
            self.p = self.p.handle_fit(train_data_container, context)

            y_pred_train = self.p.handle_predict(train_data_container, context)
            y_pred_val = self.p.handle_predict(validation_data_container, context)

            self.metrics_results_train = calculate_metrics_results(y_pred_train, self.metrics)
            self.metrics_results_validation = calculate_metrics_results(y_pred_val, self.metrics)

            train_score = self.score(y_pred_train)
            validation_score = self.score(y_pred_val)

            for callback in self.callbacks:
                if callback.call(y_pred_train, y_pred_val, train_score, validation_score):
                    self.trial_repository.update_trial(
                        training_score=train_score,
                        validation_score=validation_score,
                        model_to_checkpoint_if_better_score=self.p
                    )
                    early_stopping = True

            if early_stopping:
                break

        return self, train_data_container, validation_data_container


class AutoML(ForceHandleOnlyMixin, BaseStep):
    def __init__(
            self,
            pipeline,
            validation_technique: BaseValidation,
            hyperparams_repository: HyperparamsRepository,
            scoring_function: Callable,
            n_trial: int,
            metrics: Dict,
            epochs: int,
            callbacks: List[Callable],
            higher_score_is_better=False,
            print_func: Callable=None,
            only_save_new_best_model=True
    ):
        BaseStep.__init__(self)
        ForceHandleOnlyMixin.__init__(self)

        if print_func is None:
            print_func = print

        self.only_save_new_best_model = only_save_new_best_model
        self.pipeline = pipeline
        self.validation_technique = validation_technique
        self.higher_score_is_better = higher_score_is_better
        self.print_func = print_func

        self.n_trial = n_trial
        self.hyperparams_repository = hyperparams_repository
        self.scoring_function = scoring_function

        self.callbacks = callbacks
        self.epochs = epochs
        self.metrics = metrics

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> 'BaseStep':
        training_data_container, validation_data_container = self.validation_technique.split_data_container(data_container)
        best_score = None

        for trial_number in range(self.n_trial):
            auto_ml_data = self._load_auto_ml_data(trial_number)
            p = self.validation_technique.set_step(copy.copy(self.pipeline))

            try:
                with self.hyperparams_repository.new_experiment(auto_ml_data) as repo_trial:
                    p.update_hyperparams(repo_trial.hyperparams)

                    p, training_data_container, validation_data_container = Trainer(
                        p=p,
                        metrics=self.metrics,
                        callbacks=self.callbacks,
                        score=self.scoring_function,
                        trial_repository=repo_trial
                    ).train(
                        train_data_container=training_data_container,
                        validation_data_container=validation_data_container,
                        epochs=self.epochs,
                        context=context
                    )

                    train_score = self.scoring_function(training_data_container.train_data_container)
                    validation_score = self.scoring_function(training_data_container.validation_data_container)

                    repo_trial.set_score(train_score, validation_score)
            except Exception as error:
                track = traceback.format_exc()
                self.print_func(track)
                repo_trial.set_failed(error)

            is_new_best_score = self._get_best_score(best_score, repo_trial)

            self.hyperparams_repository.save_trial(repo_trial)
            self._save_model_if_needed(is_new_best_score, p)

        return self

    def _get_best_score(self, best_score, repo_trial):
        new_best_score = False
        if repo_trial.status == TRIAL_STATUS.FAILED:
            return new_best_score

        if repo_trial.validation_score < best_score and not self.higher_score_is_better:
            best_score = repo_trial.validation_score
            new_best_score = True
        if repo_trial.validation_score > best_score and self.higher_score_is_better:
            best_score = repo_trial.validation_score
            self.print_func('new best score: {}'.format(best_score))
            self.print_func('new best hyperparams: {}'.format(repo_trial.hyperparams))
            new_best_score = True
        return new_best_score

    def _save_model_if_needed(self, is_new_best_score, p):
        if self.only_save_new_best_model:
            if is_new_best_score:
                self.hyperparams_repository.save_model(p)
        else:
            self.hyperparams_repository.save_model(p)

    def _load_auto_ml_data(self, trial_number: int) -> AutoMLContainer:
        """
        Load data for all trials.

        :param trial_number: trial number
        :type trial_number: int
        :return: auto ml data container
        :rtype: Trials
        """
        trials = self.hyperparams_repository.load_all_trials(TRIAL_STATUS.SUCCESS)

        return AutoMLContainer(
            trial_number=trial_number,
            trials=trials,
            hyperparameter_space=self.pipeline.get_hyperparams_space(),
        )


def calculate_metrics_results(data_container: DataContainer, metrics):
    result = {}
    for metric_name, metric_function in metrics.items():
        result_metric = metric_function(data_container.data_inputs, data_container.expected_outputs)
        result[metric_name] = result_metric
    return result
