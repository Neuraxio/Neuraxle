import copy
import datetime
import glob
import hashlib
import json
import os
import time
import traceback
from abc import ABC, abstractmethod
from typing import Callable, Dict, List

from neuraxle.base import BaseStep, ExecutionContext, ForceHandleOnlyMixin
from neuraxle.data_container import DataContainer
from neuraxle.metaopt.auto_ml import Trials, TRIAL_STATUS, AutoMLContainer, BaseHyperparameterOptimizer
from neuraxle.metaopt.random import BaseCrossValidationWrapper


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
        HyperparamsRepository.__init__(self, hyperparameter_optimizer)
        if print_func is None:
            print_func = print
        self.print_func = print_func
        self.cache_folder = cache_folder

        self.trials = Trials()

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
        hyperparams = step.get_hyperparams().to_flat_as_dict_primitive()
        trial_hash = self._get_trial_hash(hyperparams)
        step.set_name(trial_hash).save(ExecutionContext(self.cache_folder), full_dump=True)

        return step


class HyperparamsJSONRepository(HyperparamsRepository):
    """
    Hyperparams repository that saves json files for every AutoML trial.

    .. seealso::
        :class:`HyperparamsRepository`,
        :class:`HyperparameterSamples`,
        :class:`AutoMLSequentialWrapper`
    """

    def __init__(self, hyperparameter_optimizer, folder):
        HyperparamsRepository.__init__(self, hyperparameter_optimizer)
        self.cache_folder = folder

    def save_trial(self, trial: 'Trial'):
        pass

    def save_model(self, step: BaseStep):
        pass

    def new_trial(self, auto_ml_container: AutoMLContainer):
        """
        Create new hyperperams trial json file.

        :param auto_ml_container: auto ml container
        :return:
        """
        hyperparams = self.hyperparameter_optimizer.find_next_best_hyperparams(auto_ml_container)
        self._create_trial_json(hyperparams)
        return Trial(hyperparams)

    def load_all_trials(self, status: 'TRIAL_STATUS' = None) -> 'Trials':
        """
        Load all hyperparameter trials with their corresponding score.
        Reads all the saved trial json files, sorted by creation date.

        :return: (hyperparams, scores)
        """
        trials = Trials()

        files = glob.glob(os.path.join(self.cache_folder, '*.json'))

        # sort by created date:
        def getmtimens(filename):
            return os.stat(filename).st_mtime_ns

        files.sort(key=getmtimens)

        for base_path in files:
            with open(base_path) as f:
                trial_json = json.load(f)

            if status is None or trial_json['status'] == status.value:
                trials.append(Trial.from_json(trial_json))

        return trials

    def _create_trial_json(self, hyperparams):
        """
        Save new trial json file.

        :return: (hyperparams, scores)
        """
        hp_dict = hyperparams.to_flat_as_dict_primitive()
        current_hyperparameters_hash = self._get_trial_hash(hp_dict)

        with open(os.path.join(self._get_new_trial_json_path(current_hyperparameters_hash)), 'w+') as outfile:
            json.dump({
                'hyperparams': hp_dict,
                'score': None,
                'status': TRIAL_STATUS.PLANNED.value
            }, outfile)

    def _save_successful_trial_json(self, hyperparams, score):
        """
        Save trial json file.

        :return: (hyperparams, scores)
        """
        hp_dict = hyperparams.to_flat_as_dict_primitive()
        current_hyperparameters_hash = self._get_trial_hash(hp_dict)
        self._remove_new_trial_json(current_hyperparameters_hash)

        with open(os.path.join(self.cache_folder, str(float(score)).replace('.', ',') + "_" + current_hyperparameters_hash) + '.json', 'w+') as outfile:
            json.dump({
                'hyperparams': hp_dict,
                'score': score,
                'status': TRIAL_STATUS.SUCCESS.value
            }, outfile)
        # Sleeping to have a valid time difference between files when reloading them to sort them by creation time:
        time.sleep(0.1)

    def _remove_new_trial_json(self, current_hyperparameters_hash):
        new_trial_json = self._get_new_trial_json_path(current_hyperparameters_hash)
        if os.path.exists(new_trial_json):
            os.remove(new_trial_json)

    def _get_new_trial_json_path(self, current_hyperparameters_hash):
        return os.path.join(self.cache_folder, "NEW_" + current_hyperparameters_hash) + '.json'

    def _save_failed_trial_json(self, hyperparams, exception):
        """
        Save trial json file.

        :return: (hyperparams, scores)
        """
        hp_dict = hyperparams.to_flat_as_dict_primitive()
        current_hyperparameters_hash = self._get_trial_hash(hp_dict)
        self._remove_new_trial_json(current_hyperparameters_hash)

        with open(os.path.join(self.cache_folder, 'FAILED_' + current_hyperparameters_hash) + '.json', 'w+') as outfile:
            json.dump({
                'hyperparams': hp_dict,
                'score': None,
                'status': TRIAL_STATUS.FAILED.value,
                'exception': str(exception)
            }, outfile)


class Trial:
    def __init__(self, hyperparams):
        self.status = None
        self.train_score = None
        self.validation_score = None
        self.train_scores = []
        self.validation_scores = []
        self.error = None
        self.error_traceback = None
        self.metrics_results_train = None
        self.metrics_results_validation = None
        self.pipeline = None

        self.hyperparams = hyperparams

    def update_trial(self, train_score, validation_score, metrics_results_train, metrics_results_validation, pipeline):
        self.train_score = train_score
        self.validation_score = validation_score
        self.train_scores.append(train_score)
        self.validation_scores.append(validation_score)
        self.metrics_results_train = metrics_results_train
        self.metrics_results_validation = metrics_results_validation
        self.pipeline = pipeline

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
        return self


class BaseCallback(ABC):
    @abstractmethod
    def call(self, trial: Trial):
        pass


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, n_epochs_without_improvement, higher_score_is_better):
        self.higher_score_is_better = higher_score_is_better
        self.n_epochs_without_improvement = n_epochs_without_improvement

    def call(self, trial: Trial):
        if len(trial.validation_scores) > self.n_epochs_without_improvement:
            if trial.validation_scores[-self.n_epochs_without_improvement] >= trial.validation_scores[-1] and self.higher_score_is_better:
                return True
            if trial.validation_scores[-self.n_epochs_without_improvement] <= trial.validation_scores[-1] and not self.higher_score_is_better:
                return True
        return False


class Trainer:
    """

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

    def __init__(self, score, refit_score, epochs, metrics=None, callbacks=None, refit_callbacks=None,
                 print_metrics=True, print_func=None):
        self.score = score
        self.refit_score = refit_score

        self.epochs = epochs
        if metrics is None:
            metrics = {}
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

    def train(self, p, train_data_container: DataContainer, validation_data_container: DataContainer, trial: Trial,
              context: ExecutionContext) -> Trial:
        """
        Train pipeline using the training data container.
        Track training, and validation metrics for each epoch.

        :param p: pipeline to train on
        :param train_data_container: train data container
        :param validation_data_container: validation data container
        :param trial: trial to execute
        :param context: execution context

        :return: executed trial
        """
        early_stopping = False

        for i in range(self.epochs):
            p = p.handle_fit(train_data_container, context)

            y_pred_train = p.handle_predict(train_data_container, context)
            y_pred_val = p.handle_predict(validation_data_container, context)

            self._update_metrics_results(y_pred_train, train_metrics=True)
            self._update_metrics_results(y_pred_val, train_metrics=False)

            train_score = self.score(y_pred_train.data_inputs, y_pred_train.expected_outputs)
            validation_score = self.score(y_pred_val.data_inputs, y_pred_val.expected_outputs)

            trial.update_trial(
                train_score=train_score,
                validation_score=validation_score,
                metrics_results_train=self.metrics_results_train,
                metrics_results_validation=self.metrics_results_validation,
                pipeline=p
            )

            for callback in self.callbacks:
                if callback.call(trial):
                    early_stopping = True

            if early_stopping:
                break

        return trial

    def refit(self, p: BaseStep, data_container: DataContainer, context: ExecutionContext) -> BaseStep:
        """
        Refit the pipeline on the whole dataset (without any validation technique).

        :param p: trial to refit
        :param data_container: data container
        :param context: execution context

        :return: fitted pipeline
        """
        early_stopping = False

        for i in range(self.epochs):
            p = p.handle_fit(data_container, context)
            pred = p.handle_predict(data_container, context)

            train_score = self.refit_score(pred.data_inputs, pred.expected_outputs)

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

    def _update_metrics_results(self, data_container: DataContainer, train_metrics: bool):
        """
        Update metrics results.

        :param data_container: data container
        :param train_metrics: bool for training mode

        :return:
        """
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
    """
    A step to execute any Automatic Machine Learning Algorithms.

    Example usage :

    .. code-block:: python

        auto_ml = AutoML(
            pipeline=Pipeline([
                MultiplyByN(2),
                NumpyReshape(shape=(-1, 1)),
                linear_model.LinearRegression()
            ]),
            validation_technique=KFoldCrossValidationWrapper(
                k_fold=2,
                scoring_function=average_kfold_scores(mean_squared_error),
                split_data_container_during_fit=False,
                predict_after_fit=False
            ),
            hyperparams_optimizer=RandomSearchHyperparameterOptimizer(),
            hyperparams_repository=InMemoryHyperparamsRepository(),
            scoring_function=average_kfold_scores(mean_squared_error),
            n_trial=1,
            metrics={'mse': average_kfold_scores(mean_squared_error)},
            epochs=2
        )

        auto_ml = auto_ml.fit(data_inputs, expected_outputs)

    .. seealso::
        :class:`BaseValidation`,
        :class:`BaseHyperparameterOptimizer`,
        :class:`HyperparamsRepository`,
        :class:`RandomSearchHyperparameterOptimizer`,
        :class:`ForceHandleOnlyMixin`,
        :class:`BaseStep`
    """

    def __init__(
            self,
            pipeline,
            validation_technique: BaseCrossValidationWrapper,
            hyperparams_optimizer: BaseHyperparameterOptimizer,
            hyperparams_repository: HyperparamsRepository,
            scoring_function: Callable,
            n_trial: int,
            metrics: Dict,
            epochs: int,
            callbacks: List[BaseCallback],
            refit_scoring_function: Callable = None,
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
        if refit_scoring_function is None:
            refit_scoring_function = scoring_function
        self.refit_scoring_function = refit_scoring_function

        self.callbacks = callbacks
        self.epochs = epochs
        self.metrics = metrics
        self.refit_trial = refit_trial

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> 'BaseStep':
        """
        Run Auto ML Loop.
        Find the best hyperparams using the hyperparameter optmizer.
        Evaluate the pipeline on each trial using a validation technique.

        :param data_container: data container to fit
        :param context: execution context

        :return: self
        """
        best_score = None
        training_data_container, validation_data_container = self.validation_technique.split_data_container(
            data_container)

        trainer = Trainer(
            metrics=self.metrics,
            callbacks=self.callbacks,
            score=self.scoring_function,
            refit_score=self.refit_scoring_function,
            epochs=self.epochs
        )

        for trial_number in range(self.n_trial):
            auto_ml_data = self._load_auto_ml_data(trial_number)
            p: BaseCrossValidationWrapper = self.validation_technique.set_step(copy.copy(self.pipeline))

            with self.hyperparams_repository.new_trial(auto_ml_data) as repo_trial:
                try:
                    p.update_hyperparams(repo_trial.hyperparams)

                    repo_trial = trainer.train(
                        p=p,
                        train_data_container=training_data_container,
                        validation_data_container=validation_data_container,
                        trial=repo_trial,
                        context=context
                    )
                except Exception as error:
                    track = traceback.format_exc()
                    self.print_func(track)
                    repo_trial.set_failed(error)

            is_new_best_score = self._trial_has_a_new_best_score(best_score, repo_trial)
            if is_new_best_score:
                best_score = repo_trial.validation_score

            self.hyperparams_repository.save_trial(repo_trial)

            p = p.get_step()
            if self.refit_trial:
                p = trainer.refit(p=p, data_container=data_container, context=context)

            self._save_pipeline_if_needed(p, is_new_best_score)

        return self

    def _trial_has_a_new_best_score(self, best_score, repo_trial):
        """
        Return True if trial has a new best score.

        :param best_score:
        :param repo_trial:
        :return:
        """
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
        Save pipeline if a new best score has been found, or if self.only_save_new_best_model is False.

        :param pipeline: pipeline to save
        :param is_new_best_score: bool that is True if a new best score has been found

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
