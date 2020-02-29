import copy
import datetime
import hashlib
import json
import traceback
from abc import ABC, abstractmethod
from typing import Callable, Dict, List

from neuraxle.base import BaseStep, ExecutionContext, ForceHandleOnlyMixin
from neuraxle.data_container import DataContainer
from neuraxle.metaopt.auto_ml import Trials, TRIAL_STATUS, AutoMLContainer, BaseHyperparameterOptimizer
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
    def __init__(self, hyperparameter_optimizer=None):
        self.hyperparameter_optimizer = hyperparameter_optimizer

    def set_optimizer(self, hyperparameter_optimizer: BaseHyperparameterOptimizer):
        """
        Set optimizer.

        :param hyperparameter_optimizer: hyperparameter optimizer
        :return:
        """
        self.hyperparameter_optimizer = hyperparameter_optimizer

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
    def new_trial(self, auto_ml_container: AutoMLContainer):
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

        InMemoryHyperparamsRepository(
            hyperparameter_optimizer=RandomSearchHyperparameterOptimizer(),
            rint_func=print,
            cache_folder='cache'
        )

    .. seealso::
        :class:`HyperparamsRepository`,
        :class:`HyperparameterSamples`,
        :class:`AutoMLSequentialWrapper`
    """

    def __init__(self, hyperparameter_optimizer=None, print_func: Callable = None, cache_folder: str = None):
        HyperparamsRepository.__init__(self)
        if print_func is None:
            print_func = print
        self.print_func = print_func
        self.cache_folder = cache_folder

        self.trials = Trials()
        self.hyperparameter_optimizer = hyperparameter_optimizer

    def load_all_trials(self, status: 'TRIAL_STATUS' = None) -> 'Trials':
        """
        Load all trials with the given status.

        :param status: trial status
        :return: list of trials
        """
        return self.trials.filter(status)

    def save_trial(self, trial: 'Trial'):
        """
        Save trial.

        :param trial: trial to save
        :return:
        """
        self.print_func(trial)
        self.trials.append(trial)

    def new_trial(self, auto_ml_container: AutoMLContainer) -> 'Trial':
        """
        Create a new trial with the best next hyperparams.

        :param auto_ml_container: auto ml data container
        :return: trial
        """
        hyperparams = self.hyperparameter_optimizer.find_next_best_hyperparams(auto_ml_container)
        self.print_func('new trial:\n{}'.format(json.dumps(hyperparams.to_nested_dict(), sort_keys=True, indent=4)))

        return Trial(hyperparams)

    def save_model(self, step: BaseStep) -> BaseStep:
        """
        Save step inside the trial hash folder.

        :param step: step to save
        :return: saved step
        """
        trial_hash = self._get_trial_hash(step.get_hyperparams().to_flat_as_dict_primitive())
        step.set_name(trial_hash).save(ExecutionContext(self.cache_folder), full_dump=True)

        return step


class Trial:
    def __init__(self, hyperparams):
        self.status = None
        self.train_score = None
        self.validation_score = None
        self.error = None
        self.error_traceback = None
        self.metrics_results_train = None
        self.metrics_results_validation = None
        self.pipeline = None

        self.hyperparams = hyperparams

    def update_trial(self, train_score, validation_score, metrics_results_train, metrics_results_validation, pipeline):
        self.train_score = train_score
        self.validation_score = validation_score
        self.metrics_results_train = metrics_results_train
        self.metrics_results_validation = metrics_results_validation
        self.pipeline = pipeline

    def set_failed_trial(self, error):
        self.status = TRIAL_STATUS.FAILED
        self.error = error
        self.error_traceback = traceback.format_exc()

    def get_step_wrapped_by_validation_technique(self):
        return self.pipeline.get_step()

    def __enter__(self):
        self.start_time = datetime.datetime.now()
        self.status = TRIAL_STATUS.PLANNED
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.datetime.now()
        self.status = TRIAL_STATUS.SUCCESS
        return self


class Trainer:
    def __init__(self, score, epochs, metrics, callbacks=None, refit_callbacks=None, print_metrics=True, print_func=None):
        self.score = score

        self.epochs = epochs
        self.metrics = metrics
        self._initialize_metrics(metrics)

        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks

        if refit_callbacks is None:
            refit_callbacks = []
        self.refit_callbacks = refit_callbacks

        if print_func is None:
            print_func = print

        self.print_func = print_func
        self.print_metrics = print_metrics

    def train(self, p, train_data_container, validation_data_container, trial_repository, context: ExecutionContext):
        early_stopping = False

        for i in range(self.epochs):
            p = p.handle_fit(train_data_container, context)

            y_pred_train = p.handle_predict(train_data_container, context)
            y_pred_val = p.handle_predict(validation_data_container, context)

            self._update_metrics_results(y_pred_train, train_metrics=True)
            self._update_metrics_results(y_pred_val, train_metrics=False)

            train_score = self.score(y_pred_train.data_inputs, y_pred_train.expected_outputs)
            validation_score = self.score(y_pred_val.data_inputs, y_pred_val.expected_outputs)

            trial_repository.update_trial(
                train_score=train_score,
                validation_score=validation_score,
                metrics_results_train=self.metrics_results_train,
                metrics_results_validation=self.metrics_results_validation,
                pipeline=p
            )

            for callback in self.callbacks:
                if callback.call(trial_repository):
                    early_stopping = True

            if early_stopping:
                break

        return train_data_container, validation_data_container

    def refit(self, trial: Trial, data_container: DataContainer, context: ExecutionContext):
        p = trial.get_step_wrapped_by_validation_technique()

        early_stopping = False

        for i in range(self.epochs):
            p = p.handle_fit(data_container, context)
            pred = p.handle_predict(data_container, context)

            train_score = self.score(pred.data_inputs, pred.expected_outputs)

            for callback in self.refit_callbacks:
                if callback.call(p, train_score):
                    early_stopping = True

            if early_stopping:
                break

        return p

    def _initialize_metrics(self, metrics):
        """
        Initialize metrics results dict for train, and validation using the metrics function dict.

        :param metrics: metrics function dict
        :type metrics: dict
        :return:
        """
        self.metrics_results_train = {}
        self.metrics_results_validation = {}

        for m in metrics:
            self.metrics_results_train[m] = []
            self.metrics_results_validation[m] = []

    def _update_metrics_results(self, data_container, train_metrics):
        result = {}
        for metric_name, metric_function in self.metrics.items():
            result_metric = metric_function(data_container.data_inputs, data_container.expected_outputs)
            result[metric_name] = result_metric

            if train_metrics:
                self.metrics_results_train[metric_name].append(result_metric)
            else:
                self.metrics_results_validation[metric_name].append(result_metric)

        if self.print_metrics:
            self.print_func(result)


class AutoML(ForceHandleOnlyMixin, BaseStep):
    def __init__(
            self,
            pipeline,
            validation_technique: BaseValidation,
            hyperparams_optimizer: BaseHyperparameterOptimizer,
            hyperparams_repository: HyperparamsRepository,
            scoring_function: Callable,
            n_trial: int,
            metrics: Dict,
            epochs: int,
            callbacks: List[Callable],
            higher_score_is_better=False,
            print_func: Callable = None,
            only_save_new_best_model=True,
            refit_trial=True
    ):
        BaseStep.__init__(self)
        ForceHandleOnlyMixin.__init__(self)

        if print_func is None:
            print_func = print

        hyperparams_repository.set_optimizer(hyperparams_optimizer)

        self.only_save_new_best_model = only_save_new_best_model
        self.pipeline = pipeline
        self.validation_technique = validation_technique
        self.higher_score_is_better = higher_score_is_better
        self.print_func = print_func

        self.n_trial = n_trial
        self.hyperparams_repository = hyperparams_repository
        self.hyperparameter_optimizer = hyperparams_optimizer
        self.scoring_function = scoring_function

        self.callbacks = callbacks
        self.epochs = epochs
        self.metrics = metrics
        self.refit_trial = refit_trial

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> 'BaseStep':
        training_data_container, validation_data_container = self.validation_technique.split_data_container(data_container)
        best_score = None
        trainer = Trainer(
            metrics=self.metrics,
            callbacks=self.callbacks,
            score=self.scoring_function,
            epochs=self.epochs
        )

        for trial_number in range(self.n_trial):
            auto_ml_data = self._load_auto_ml_data(trial_number)
            p = self.validation_technique.set_step(copy.copy(self.pipeline))

            with self.hyperparams_repository.new_trial(auto_ml_data) as repo_trial:
                try:
                    p.update_hyperparams(repo_trial.hyperparams)

                    training_data_container, validation_data_container = trainer.train(
                        p=p,
                        train_data_container=training_data_container,
                        validation_data_container=validation_data_container,
                        trial_repository=repo_trial,
                        context=context
                    )
                except Exception as error:
                    track = traceback.format_exc()
                    self.print_func(track)
                    repo_trial.set_failed(error)

            is_new_best_score = self._get_best_score(best_score, repo_trial)
            if is_new_best_score:
                best_score = repo_trial.validation_score

            self.hyperparams_repository.save_trial(repo_trial)

            pipeline = repo_trial.pipeline
            if self.refit_trial:
                pipeline = trainer.refit(repo_trial.pipeline, data_container, context)

            self._save_pipeline_if_needed(pipeline, is_new_best_score)

        return self

    def _get_best_score(self, best_score, repo_trial):
        if best_score is None:
            return True

        new_best_score = False
        if repo_trial.status == TRIAL_STATUS.FAILED:
            return new_best_score

        if repo_trial.validation_score < best_score and not self.higher_score_is_better:
            new_best_score = True

        if repo_trial.validation_score > best_score and self.higher_score_is_better:
            self.print_func('new best score: {}'.format(best_score))
            self.print_func('new best hyperparams: {}'.format(repo_trial.hyperparams))
            new_best_score = True

        return new_best_score

    def _save_pipeline_if_needed(self, pipeline, is_new_best_score):
        """

        :param pipeline:
        :param is_new_best_score:
        :return:
        """
        if self.only_save_new_best_model:
            if is_new_best_score:
                self.hyperparams_repository.save_model(pipeline)
        else:
            self.hyperparams_repository.save_model(pipeline)

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

