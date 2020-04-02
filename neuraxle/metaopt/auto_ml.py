"""
Neuraxle's AutoML Classes
====================================
Classes used to build any Automatic Machine Learning strategies.

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

import copy
import glob
import hashlib
import json
import math
import os
import time
import traceback
from abc import ABC, abstractmethod
from typing import Callable, List, Union, Tuple

import numpy as np

from neuraxle.base import BaseStep, ExecutionContext, ForceHandleOnlyMixin
from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace
from neuraxle.metaopt.callbacks import BaseCallback, CallbackList, ScoringCallback
from neuraxle.metaopt.random import BaseCrossValidationWrapper
from neuraxle.metaopt.trial import Trial, TrialSplit, TRIAL_STATUS, Trials


class HyperparamsRepository(ABC):
    """
    Hyperparams repository that saves hyperparams, and scores for every AutoML trial.

    .. seealso::
        :class:`AutoML`,
        :class:`Trainer`,
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`InMemoryHyperparamsRepository`,
        :class:`HyperparamsJSONRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(self, hyperparameter_selection_strategy=None, cache_folder=None, best_retrained_model_folder=None):
        if best_retrained_model_folder is None:
            best_retrained_model_folder = os.path.join(cache_folder, 'best')
        self.best_retrained_model_folder = best_retrained_model_folder

        self.hyperparameter_selection_strategy = hyperparameter_selection_strategy
        self.cache_folder = cache_folder

    def set_strategy(self, hyperparameter_selection_strategy: 'BaseHyperparameterSelectionStrategy'):
        """
        Set hyperparameter selection strategy.

        :param hyperparameter_selection_strategy: hyperparameter selection strategy.
        :return:
        """
        self.hyperparameter_selection_strategy = hyperparameter_selection_strategy

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
        """
        Save trial.

        :param trial: trial to save.
        :return:
        """
        pass

    def get_best_hyperparams(self) -> HyperparameterSamples:
        """
        Get best hyperparams from all of the saved trials.

        :return: best hyperparams.
        """
        trials = self.load_all_trials(status=TRIAL_STATUS.SUCCESS)
        best_hyperparams = HyperparameterSamples(trials.get_best_hyperparams())
        return best_hyperparams

    def get_best_model(self):
        """
        Load the best model saved inside the best retrained model folder.

        :return:
        """
        hyperparams: HyperparameterSamples = self.get_best_hyperparams()
        trial_hash: str = self._get_trial_hash(HyperparameterSamples(hyperparams).to_flat_as_dict_primitive())
        p: BaseStep = ExecutionContext(str(self.best_retrained_model_folder)).load(trial_hash)

        return p

    def save_best_model(self, step: BaseStep):
        """
        Save the best model inside the best retrained model folder.

        :param step: step to save
        :return: saved step
        """
        hyperparams = step.get_hyperparams().to_flat_as_dict_primitive()
        trial_hash = self._get_trial_hash(hyperparams)
        step.set_name(trial_hash).save(ExecutionContext(self.best_retrained_model_folder), full_dump=True)

        return step

    @abstractmethod
    def new_trial(self, auto_ml_container: 'AutoMLContainer'):
        """
        Save hyperparams, and score for a failed trial.

        :return: (hyperparams, scores)
        """
        pass

    def _get_trial_hash(self, hp_dict):
        """
        Hash hyperparams with md5 to create a trial hash.

        :param hp_dict:
        :return:
        """
        current_hyperparameters_hash = hashlib.md5(str.encode(str(hp_dict))).hexdigest()
        return current_hyperparameters_hash


class InMemoryHyperparamsRepository(HyperparamsRepository):
    """
    In memory hyperparams repository that can print information about trials.
    Useful for debugging.

    Example usage :

    .. code-block:: python

        InMemoryHyperparamsRepository(
            hyperparameter_selection_strategy=RandomSearchHyperparameterSelectionStrategy(),
            print_func=print,
            cache_folder='cache',
            best_retrained_model_folder='best'
        )

    .. seealso::
        :class:`AutoML`,
        :class:`Trainer`,
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`HyperparamsJSONRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(self, hyperparameter_selection_strategy=None, print_func: Callable = None, cache_folder: str = None,
                 best_retrained_model_folder=None):
        HyperparamsRepository.__init__(
            self,
            hyperparameter_selection_strategy=hyperparameter_selection_strategy,
            cache_folder=cache_folder,
            best_retrained_model_folder=best_retrained_model_folder
        )
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

    def new_trial(self, auto_ml_container: 'AutoMLContainer') -> 'Trial':
        """
        Create a new trial with the best next hyperparams.

        :param auto_ml_container: auto ml data container
        :return: trial
        """
        hyperparams = self.hyperparameter_selection_strategy.find_next_best_hyperparams(auto_ml_container)
        self.print_func('new trial:\n{}'.format(json.dumps(hyperparams.to_nested_dict(), sort_keys=True, indent=4)))

        return Trial(hyperparams=hyperparams, main_metric_name=auto_ml_container.main_scoring_metric_name)


class HyperparamsJSONRepository(HyperparamsRepository):
    """
    Hyperparams repository that saves json files for every AutoML trial.

    Example usage :

    .. code-block:: python

        HyperparamsJSONRepository(
            hyperparameter_selection_strategy=RandomSearchHyperparameterSelectionStrategy(),
            cache_folder='cache',
            best_retrained_model_folder='best'
        )


    .. seealso::
        :class:`AutoML`,
        :class:`Trainer`,
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`InMemoryHyperparamsRepository`,
        :class:`HyperparamsJSONRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.hyperparams.trial.HyperparameterSamples`
    """

    def __init__(
            self,
            hyperparameter_selection_strategy: 'BaseHyperparameterSelectionStrategy' = None,
            cache_folder=None,
            best_retrained_model_folder=None
    ):
        HyperparamsRepository.__init__(
            self,
            hyperparameter_selection_strategy=hyperparameter_selection_strategy,
            cache_folder=cache_folder,
            best_retrained_model_folder=best_retrained_model_folder
        )

    def save_trial(self, trial: 'Trial'):
        """
        Save trial json.

        :param trial: trial to save
        :return:
        """
        hp_dict = trial.hyperparams.to_flat_as_dict_primitive()
        current_hyperparameters_hash = self._get_trial_hash(hp_dict)
        self._remove_new_trial_json(current_hyperparameters_hash)

        if trial.status == TRIAL_STATUS.SUCCESS:
            trial_file_path = self._get_successful_trial_json_file_path(trial)
        else:
            trial_file_path = self._get_failed_trial_json_file_path(trial)

        with open(trial_file_path, 'w+') as outfile:
            json.dump(trial.to_json(), outfile)

        # Sleeping to have a valid time difference between files when reloading them to sort them by creation time:
        time.sleep(0.1)

    def new_trial(self, auto_ml_container: 'AutoMLContainer'):
        """
        Create new hyperperams trial json file.

        :param auto_ml_container: auto ml container
        :return:
        """
        hyperparams = self.hyperparameter_selection_strategy.find_next_best_hyperparams(auto_ml_container)
        trial = Trial(hyperparams, cache_folder=self.cache_folder,
                      main_metric_name=auto_ml_container.main_scoring_metric_name)
        self._create_trial_json(trial=trial)

        return trial

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
                trials.append(Trial.from_json(trial_json=trial_json))

        return trials

    def _create_trial_json(self, trial: 'Trial'):
        """
        Save new trial json file.

        :return: (hyperparams, scores)
        """
        hp_dict = trial.hyperparams.to_flat_as_dict_primitive()
        current_hyperparameters_hash = self._get_trial_hash(hp_dict)

        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        with open(os.path.join(self._get_new_trial_json_path(current_hyperparameters_hash)), 'w+') as outfile:
            json.dump(trial.to_json(), outfile)

    def _get_successful_trial_json_file_path(self, trial: 'Trial') -> str:
        """
        Get the json path for the given successful trial.

        :param trial: trial
        :return: str
        """
        trial_hash = self._get_trial_hash(trial.hyperparams.to_flat_as_dict_primitive())
        return os.path.join(
            self.cache_folder,
            str(float(trial.get_validation_score())).replace('.', ',') + "_" + trial_hash
        ) + '.json'

    def _get_failed_trial_json_file_path(self, trial: 'Trial'):
        """
        Get the json path for the given failed trial.

        :param trial:
        :return:
        """
        trial_hash = self._get_trial_hash(trial.hyperparams.to_flat_as_dict_primitive())
        return os.path.join(self.cache_folder, 'FAILED_' + trial_hash) + '.json'

    def _remove_new_trial_json(self, current_hyperparameters_hash):
        """
        Remove trial file associated with the given hyperparameters hash.

        :param current_hyperparameters_hash:
        :return:
        """
        new_trial_json = self._get_new_trial_json_path(current_hyperparameters_hash)
        if os.path.exists(new_trial_json):
            os.remove(new_trial_json)

    def _get_new_trial_json_path(self, current_hyperparameters_hash):
        """
        Get new trial json path.

        :param current_hyperparameters_hash:
        :return:
        """
        return os.path.join(self.cache_folder, "NEW_" + current_hyperparameters_hash) + '.json'


class BaseHyperparameterSelectionStrategy(ABC):
    @abstractmethod
    def find_next_best_hyperparams(self, auto_ml_container: 'AutoMLContainer') -> HyperparameterSamples:
        """
        Find the next best hyperparams using previous trials.

        :param auto_ml_container: trials data container
        :return: next best hyperparams
        """
        raise NotImplementedError()


class Trainer:
    """

    Example usage :

    .. code-block:: python

        trainer = Trainer(
            callbacks=[],
            epochs=10,
            print_func=print
        )

        repo_trial = trainer.fit(
            p=p,
            trial_repository=repo_trial,
            train_data_container=training_data_container,
            validation_data_container=validation_data_container,
            context=context
        )

        pipeline = trainer.refit(repo_trial.pipeline, data_container, context)


    .. seealso::
        :class:`AutoML`,
        :class:`Trainer`,
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`InMemoryHyperparamsRepository`,
        :class:`HyperparamsJSONRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(
            self,
            epochs,
            metrics=None,
            callbacks=None,
            print_metrics=True,
            print_func=None
    ):
        self.epochs = epochs
        if metrics is None:
            metrics = {}
        self.metrics = metrics
        self._initialize_metrics(metrics)

        self.callbacks = CallbackList(callbacks)

        if print_func is None:
            print_func = print

        self.print_func = print_func
        self.print_metrics = print_metrics

    def fit_trial_split(
            self,
            trial_split: TrialSplit,
            train_data_container: DataContainer,
            validation_data_container: DataContainer,
            context: ExecutionContext
    ) -> TrialSplit:
        """
        Train pipeline using the training data container.
        Track training, and validation metrics for each epoch.

        :param train_data_container: train data container
        :param validation_data_container: validation data container
        :param trial_split: trial to execute
        :param context: execution context

        :return: executed trial
        """
        early_stopping = False

        for i in range(self.epochs):
            self.print_func('\nepoch {}/{}'.format(i + 1, self.epochs))
            trial_split = trial_split.fit_trial_split(train_data_container, context)
            y_pred_train = trial_split.predict_with_pipeline(train_data_container, context)
            y_pred_val = trial_split.predict_with_pipeline(validation_data_container, context)

            if self.callbacks.call(
                    trial=trial_split,
                    epoch_number=i,
                    total_epochs=self.epochs,
                    input_train=train_data_container,
                    pred_train=y_pred_train,
                    input_val=validation_data_container,
                    pred_val=y_pred_val,
                    is_finished_and_fitted=early_stopping
            ):
                break

        return trial_split

    def refit(self, p: BaseStep, data_container: DataContainer, context: ExecutionContext) -> BaseStep:
        """
        Refit the pipeline on the whole dataset (without any validation technique).

        :param p: trial to refit
        :param data_container: data container
        :param context: execution context

        :return: fitted pipeline
        """
        for i in range(self.epochs):
            p = p.handle_fit(data_container, context)

        return p

    def _initialize_metrics(self, metrics):
        """
        Initialize metrics results dict for train, and validation using the metrics function dict.

        :param metrics: metrics function dict

        :return:
        """
        self.metrics_results_train = {}
        self.metrics_results_validation = {}

        for m in metrics:
            self.metrics_results_train[m] = []
            self.metrics_results_validation[m] = []

    def get_main_metric_name(self) -> str:
        """
        Get main metric name.

        :return:
        """
        return self.callbacks[0].name


class AutoML(ForceHandleOnlyMixin, BaseStep):
    """
    A step to execute any Automatic Machine Learning Algorithms.

    Example usage :

    .. code-block:: python

        auto_ml = AutoML(
            pipeline,
            n_trials=n_iter,
            validation_split_function=validation_splitter(0.2),
            hyperparams_optimizer=RandomSearchHyperparameterSelectionStrategy(),
            scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
            callbacks=[
                MetricCallback('mse', metric_function=mean_squared_error, higher_score_is_better=False)
            ],
            refit_trial=True,
            print_metrics=False,
            cache_folder_when_no_handle=str(tmpdir)
        )

        auto_ml = auto_ml.fit(data_inputs, expected_outputs)


    .. seealso::
        :class:`~neuraxle.base.BaseStep`,
        :class:`~neuraxle.base.ForceHandleOnlyMixin`,
        :class:`Trainer`,
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`~neuraxle.metaopt.trial.Trials`,
        :class:`HyperparamsRepository`,
        :class:`InMemoryHyperparamsRepository`,
        :class:`HyperparamsJSONRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(
            self,
            pipeline: BaseStep,
            validation_splitter: 'BaseValidationSplitter',
            refit_trial: bool,
            scoring_callback: ScoringCallback,
            hyperparams_optimizer: BaseHyperparameterSelectionStrategy = None,
            hyperparams_repository: HyperparamsRepository = None,
            n_trials: int = 10,
            epochs: int = 1,
            callbacks: List[BaseCallback] = None,
            refit_scoring_function: Callable = None,
            print_func: Callable = None,
            cache_folder_when_no_handle=None
    ):
        BaseStep.__init__(self)
        ForceHandleOnlyMixin.__init__(self, cache_folder=cache_folder_when_no_handle)

        self.validation_split_function: BaseValidationSplitter = validation_splitter

        if print_func is None:
            print_func = print

        if hyperparams_optimizer is None:
            hyperparams_optimizer = RandomSearchHyperparameterSelectionStrategy()
        self.hyperparameter_optimizer: BaseHyperparameterSelectionStrategy = hyperparams_optimizer

        if hyperparams_repository is None:
            hyperparams_repository = HyperparamsJSONRepository(hyperparams_optimizer, cache_folder_when_no_handle)
        else:
            hyperparams_repository.set_strategy(hyperparams_optimizer)

        self.hyperparams_repository: HyperparamsJSONRepository = hyperparams_repository

        self.pipeline: BaseStep = pipeline
        self.print_func: Callable = print_func

        self.n_trial: int = n_trials
        self.hyperparams_repository: HyperparamsRepository = hyperparams_repository

        self.refit_scoring_function: Callable = refit_scoring_function

        if callbacks is None:
            callbacks = []

        callbacks: List[BaseCallback] = [scoring_callback] + callbacks

        self.refit_trial: bool = refit_trial

        self.trainer = Trainer(
            callbacks=callbacks,
            epochs=epochs,
            print_func=self.print_func
        )

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> 'BaseStep':
        """
        Run Auto ML Loop.
        Find the best hyperparams using the hyperparameter optmizer.
        Evaluate the pipeline on each trial using a validation technique.

        :param data_container: data container to fit
        :param context: execution context

        :return: self
        """
        validation_splits = self.validation_split_function.split_data_container(data_container)

        for trial_number in range(self.n_trial):
            try:
                auto_ml_data = AutoMLContainer(
                    trial_number=trial_number,
                    trials=self.hyperparams_repository.load_all_trials(TRIAL_STATUS.SUCCESS),
                    hyperparameter_space=self.pipeline.get_hyperparams_space(),
                    main_scoring_metric_name=self.trainer.get_main_metric_name()
                )

                with self.hyperparams_repository.new_trial(auto_ml_data) as repo_trial:
                    self.print_func('\ntrial {}/{}'.format(trial_number + 1, self.n_trial))

                    repo_trial_split = self._execute_trial(
                        trial_number=trial_number,
                        repo_trial=repo_trial,
                        context=context,
                        validation_splits=validation_splits
                    )
            except (SystemError, SystemExit, EOFError, KeyboardInterrupt) as error:
                track = traceback.format_exc()
                repo_trial.set_failed(error)
                self.print_func(track)
                raise error
            except Exception:
                track = traceback.format_exc()
                self.print_func('failed trial {}'.format(
                    self._get_trial_split_description(repo_trial, repo_trial_split, validation_splits, trial_number)))
                self.print_func(track)
            finally:
                repo_trial.update_final_trial_status()
                self.hyperparams_repository.save_trial(repo_trial)

        best_hyperparams = self.hyperparams_repository.get_best_hyperparams()

        self.print_func(
            'best hyperparams:\n{}'.format(json.dumps(best_hyperparams.to_nested_dict(), sort_keys=True, indent=4)))
        p: BaseStep = self._load_virgin_model(hyperparams=best_hyperparams)

        if self.refit_trial:
            p = self.trainer.refit(
                p=p,
                data_container=data_container,
                context=context
            )

            self.hyperparams_repository.save_best_model(p)

        return self

    def _execute_trial(self, trial_number: int, repo_trial: Trial, context: ExecutionContext,
                       validation_splits: List[Tuple[DataContainer, DataContainer]]):
        for training_data_container, validation_data_container in validation_splits:
            p = copy.deepcopy(self.pipeline)
            p.update_hyperparams(repo_trial.hyperparams)
            repo_trial.set_hyperparams(p.get_hyperparams())

            with repo_trial.new_validation_split(p) as repo_trial_split:
                trial_split_description = self._get_trial_split_description(
                    repo_trial=repo_trial,
                    repo_trial_split=repo_trial_split,
                    validation_splits=validation_splits,
                    trial_number=trial_number
                )

                self.print_func('fitting trial {}'.format(
                    trial_split_description
                ))

                repo_trial_split = self.trainer.fit_trial_split(
                    trial_split=repo_trial_split,
                    train_data_container=training_data_container,
                    validation_data_container=validation_data_container,
                    context=context
                )

                repo_trial_split.set_success()

                self.print_func('success trial {} score: {}'.format(
                    trial_split_description,
                    repo_trial_split.get_validation_score()
                ))

        return repo_trial_split

    def _get_trial_split_description(self, repo_trial, repo_trial_split, validation_splits, trial_number):
        trial_split_description = '{}/{} split {}/{}\nhyperparams: {}\n'.format(
            trial_number + 1,
            self.n_trial,
            repo_trial_split.split_number + 1,
            len(validation_splits),
            json.dumps(repo_trial.hyperparams, sort_keys=True, indent=4)
        )
        return trial_split_description

    def get_best_model(self):
        """
        Get best model using the hyperparams repository.

        :return:
        """
        return self.hyperparams_repository.get_best_model()

    def _load_virgin_best_model(self) -> BaseStep:
        """
        Get the best model from all of the previous trials.

        :return: best model step
        """
        best_hyperparams = self.hyperparams_repository.get_best_hyperparams()
        p: Union[BaseCrossValidationWrapper, BaseStep] = copy.copy(self.pipeline)
        p = p.update_hyperparams(best_hyperparams)

        best_model = p.get_step()
        return copy.deepcopy(best_model)

    def _load_virgin_model(self, hyperparams: HyperparameterSamples) -> BaseStep:
        """
        Load virigin model with the given hyperparams.

        :return: best model step
        """
        return copy.deepcopy(self.pipeline).update_hyperparams(hyperparams)


class AutoMLContainer:
    """
    Data object for auto ml.

    .. seealso::
        :class:`Trainer`,
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`~neuraxle.metaopt.trial.Trials`,
        :class:`HyperparamsRepository`,
        :class:`InMemoryHyperparamsRepository`,
        :class:`HyperparamsJSONRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(
            self,
            trials: 'Trials',
            hyperparameter_space: HyperparameterSpace,
            trial_number: int,
            main_scoring_metric_name: str
    ):
        self.trials = trials
        self.hyperparameter_space = hyperparameter_space
        self.trial_number = trial_number
        self.main_scoring_metric_name = main_scoring_metric_name


class RandomSearchHyperparameterSelectionStrategy(BaseHyperparameterSelectionStrategy):
    """
    AutoML Hyperparameter Optimizer that randomly samples the space of random variables.
    Please refer to :class:`AutoML` for a usage example.

    .. seealso::
        :class:`Trainer`,
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`~neuraxle.metaopt.trial.Trials`,
        :class:`HyperparamsRepository`,
        :class:`InMemoryHyperparamsRepository`,
        :class:`HyperparamsJSONRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(self):
        BaseHyperparameterSelectionStrategy.__init__(self)

    def find_next_best_hyperparams(self, auto_ml_container: 'AutoMLContainer') -> HyperparameterSamples:
        """
        Randomly sample the next hyperparams to try.

        :param auto_ml_container: trials data container
        :return: next best hyperparams
        """
        return auto_ml_container.hyperparameter_space.rvs()


class BaseValidationSplitter(ABC):
    def split_data_container(self, data_container: DataContainer) -> List[Tuple[DataContainer, DataContainer]]:
        """
        Wrap a validation split function with a split data container function.
        A validation split function takes two arguments:  data inputs, and expected outputs.

        :param data_container: data container to split
        :return: a function that returns the pairs of training, and validation data containers for each validation split.
        """
        train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs = self.split(
            data_inputs=data_container.data_inputs,
            expected_outputs=data_container.expected_outputs
        )

        train_data_container = DataContainer(data_inputs=train_data_inputs, expected_outputs=train_expected_outputs)
        validation_data_container = DataContainer(data_inputs=validation_data_inputs,
                                                  expected_outputs=validation_expected_outputs)

        splits = []
        for (train_current_id, train_di, train_eo), (validation_current_id, validation_di, validation_eo) in zip(
                train_data_container, validation_data_container):
            train_data_container_split = DataContainer(
                summary_id=train_current_id,
                data_inputs=train_di,
                expected_outputs=train_eo
            )

            validation_data_container_split = DataContainer(
                summary_id=validation_current_id,
                data_inputs=validation_di,
                expected_outputs=validation_eo
            )

            splits.append((train_data_container_split, validation_data_container_split))

        return splits

    @abstractmethod
    def split(self, data_inputs, expected_outputs=None) -> Tuple[List, List, List, List]:
        pass


class KFoldCrossValidationSplitter(BaseValidationSplitter):
    """
    Create a function that splits data with K-Fold Cross-Validation resampling.

    .. code-block:: python

        # create a kfold cross validation splitter with 2 kfold
        kfold_cross_validation_split(0.20)


    :param k_fold: number of folds.
    :return:
    """

    def __init__(self, k_fold: int):
        BaseValidationSplitter.__init__(self)
        self.k_fold = k_fold

    def split(self, data_inputs, expected_outputs=None) -> Tuple[List, List, List, List]:
        data_inputs_train, data_inputs_val = kfold_cross_validation_split(
            data_inputs=data_inputs,
            k_fold=self.k_fold
        )

        if expected_outputs is not None:
            expected_outputs_train, expected_outputs_val = kfold_cross_validation_split(
                data_inputs=expected_outputs,
                k_fold=self.k_fold
            )

            return data_inputs_train, expected_outputs_train, data_inputs_val, expected_outputs_val

        return data_inputs_train, [None] * len(data_inputs_train), data_inputs_val, [None] * len(data_inputs_val)


def kfold_cross_validation_split(data_inputs, k_fold):
    splitted_train_data_inputs = []
    splitted_validation_inputs = []

    step = len(data_inputs) / float(k_fold)
    for i in range(k_fold):
        a = int(step * i)
        b = int(step * (i + 1))
        if b > len(data_inputs):
            b = len(data_inputs)

        validation = data_inputs[a:b]
        train = np.concatenate((data_inputs[:a], data_inputs[b:]), axis=0)

        splitted_validation_inputs.append(validation)
        splitted_train_data_inputs.append(train)

    return splitted_train_data_inputs, splitted_validation_inputs


class ValidationSplitter(BaseValidationSplitter):
    """
    Create a function that splits data into a training, and a validation set.

    .. code-block:: python

        # create a validation splitter function with 80% train, and 20% validation
        validation_splitter(0.20)


    :param test_size: test size in float
    :return:
    """

    def __init__(self, test_size: float):
        self.test_size = test_size

    def split(self, data_inputs, expected_outputs=None) -> Tuple[List, List, List, List]:
        train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs = validation_split(
            test_size=self.test_size,
            data_inputs=data_inputs,
            expected_outputs=expected_outputs
        )

        return [train_data_inputs], [train_expected_outputs], [validation_data_inputs], [validation_expected_outputs]


def validation_split(test_size: float, data_inputs, expected_outputs=None) -> Tuple[List, List, List, List]:
    """
    Split data inputs, and expected outputs into a training set, and a validation set.

    :param test_size: test size in float
    :param data_inputs: data inputs to split
    :param expected_outputs: expected outputs to split
    :return: train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs
    """
    validation_data_inputs = _validation_split(data_inputs, test_size)
    validation_expected_outputs = None
    if expected_outputs is not None:
        validation_expected_outputs = _validation_split(expected_outputs, test_size)

    train_data_inputs = _train_split(data_inputs, test_size)
    train_expected_outputs = None
    if expected_outputs is not None:
        train_expected_outputs = _train_split(expected_outputs, test_size)

    return train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs


def _train_split(data_inputs, test_size) -> List:
    """
    Split training set.

    :param data_inputs: data inputs to split
    :return: train_data_inputs
    """
    return data_inputs[0:_get_index_split(data_inputs, test_size)]


def _validation_split(data_inputs, test_size) -> List:
    """
    Split validation set.

    :param data_inputs: data inputs to split
    :return: validation_data_inputs
    """
    return data_inputs[_get_index_split(data_inputs, test_size):]


def _get_index_split(data_inputs, test_size):
    return math.floor(len(data_inputs) * (1 - test_size))
