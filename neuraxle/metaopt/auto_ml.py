"""
Neuraxle's AutoML Classes
====================================
Classes used to build any Automatic Machine Learning pipelines.
Hyperparameter selection strategies are used to optimize the hyperparameters of given pipelines.

..
    Copyright 2021, Neuraxio Inc.

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
import gc
import glob
import hashlib
import json
import logging
import math
import multiprocessing
import os
import sys
import time
import traceback
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Union, Any, Optional

import numpy as np
from neuraxle.base import (BaseService, BaseServiceT, BaseStep, ExecutionContext, ExecutionPhase, Flow,
                           ForceHandleMixin, _HasChildrenMixin, MetaService, MetaServiceMixin, TruncableService)
from neuraxle.data_container import DACT as DACT
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace)
from neuraxle.logging.warnings import (warn_deprecated_arg,
                                       warn_deprecated_class)
from neuraxle.metaopt.callbacks import (BaseCallback, CallbackList,
                                        ScoringCallback)
from neuraxle.metaopt.data.vanilla import TrialStatus
from neuraxle.metaopt.observable import _Observable, _Observer
from neuraxle.metaopt.trial import Trial, Trials, TrialSplit
from neuraxle.metaopt.validation import BaseCrossValidationWrapper


class HyperparamsRepository(_Observable[Tuple['HyperparamsRepository', Trial]], ABC):
    """
    Hyperparams repository that saves hyperparams, and scores for every AutoML trial.
    Cache folder can be changed to do different round numbers.

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

    @abstractmethod
    def load_trials(self, status: 'TrialStatus' = None) -> 'Trials':
        """
        Load all hyperparameter trials with their corresponding score.
        Sorted by creation date.
        Filtered by probided status.

        :param status: status to filter trials.
        :return: Trials (hyperparams, scores)
        """
        pass

    def save_trial(self, trial: 'Trial'):
        """
        Save trial, and notify trial observers.

        :param trial: trial to save.
        :return:
        """
        self._save_trial(trial)
        self.notify_next(value=(self, trial))  # notify a tuple of (repo, trial) to observers

    @abstractmethod
    def _save_trial(self, trial: 'Trial'):
        """
        save trial.

        :param trial: trial to save.
        :return:
        """
        pass


class InMemoryHyperparamsRepository(HyperparamsRepository):
    """
    In memory hyperparams repository that can print information about trials.
    Useful for debugging.
    """

    def __init__(self, pre_made_trials: Optional[Trials] = None):
        HyperparamsRepository.__init__(self)
        self.trials: Trials = pre_made_trials if pre_made_trials is not None else Trials()

    def load_trials(self, status: 'TrialStatus' = None) -> 'Trials':
        """
        Load all trials with the given status.

        :param status: trial status
        :return: list of trials
        """
        return self.trials.filter(status)

    def _save_trial(self, trial: 'Trial'):
        """
        Save trial.

        :param trial: trial to save
        :return:
        """
        self.trials.append(trial)


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
            cache_folder: str = None,
            best_retrained_model_folder: str = None
    ):
        HyperparamsRepository.__init__(self)
        cache_folder: str = cache_folder if cache_folder is not None else 'json_repo_cache'
        best_retrained_model_folder: str = (
            best_retrained_model_folder if best_retrained_model_folder is not None else 'json_repo_best_model')
        self.json_path_remove_on_update = None

    def _save_trial(self, trial: 'Trial'):
        """
        Save trial json.

        :param trial: trial to save
        :return:
        """
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        self._remove_previous_trial_state_json()

        trial_path_func = {
            TrialStatus.SUCCESS: self._get_successful_trial_json_file_path,
            TrialStatus.FAILED: self._get_failed_trial_json_file_path,
            TrialStatus.RUNNING: self._get_ongoing_trial_json_file_path,
            TrialStatus.PLANNED: self._get_new_trial_json_file_path
        }
        trial_file_path = trial_path_func[trial.status](trial)

        with open(trial_file_path, 'w+') as outfile:
            json.dump(trial.to_json(), outfile)

        if trial.status in (TrialStatus.SUCCESS, TrialStatus.FAILED):
            self.json_path_remove_on_update = None
        else:
            self.json_path_remove_on_update = trial_file_path

        # Sleeping to have a valid time difference between files when reloading them to sort them by creation time:
        time.sleep(0.1)

    def new_trial(self, auto_ml_container) -> Trial:
        """
        Create new hyperperams trial json file.

        :param auto_ml_container: auto ml container
        :return:
        """
        trial: Trial = HyperparamsRepository.new_trial(self, auto_ml_container)
        self._save_trial(trial)

        return trial

    def load_trials(self, status: 'TrialStatus' = None) -> 'Trials':
        """
        Load all hyperparameter trials with their corresponding score.
        Reads all the saved trial json files, sorted by creation date.

        :param status: (optional) filter to select only trials with this status.
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
                try:
                    trial_json = json.load(f)
                except Exception as err:
                    print('invalid trial json file'.format(base_path))
                    print(traceback.format_exc())
                    continue

            if status is None or trial_json['status'] == status.value:
                trials.append(Trial.from_json(
                    update_trial_function=self.save_trial,
                    trial_json=trial_json,
                    cache_folder=self.cache_folder
                ))

        return trials

    def _get_successful_trial_json_file_path(self, trial: 'Trial') -> str:
        """
        Get the json path for the given successful trial.

        :param trial: trial
        :return: str
        """
        trial_hash = self._get_trial_hash(trial.hyperparams.to_flat_dict())
        return os.path.join(
            self.cache_folder,
            str(float(trial.get_validation_score())).replace('.', ',') + "_" + trial_hash
        ) + '.json'

    def _get_failed_trial_json_file_path(self, trial: 'Trial'):
        """
        Get the json path for the given failed trial.

        :param trial: trial
        :return: str
        """
        trial_hash = self._get_trial_hash(trial.hyperparams.to_flat_dict())
        return os.path.join(self.cache_folder, 'FAILED_' + trial_hash) + '.json'

    def _get_ongoing_trial_json_file_path(self, trial: 'Trial'):
        """
        Get ongoing trial json path.
        """
        hp_dict = trial.hyperparams.to_flat_dict()
        current_hyperparameters_hash = self._get_trial_hash(hp_dict)
        return os.path.join(self.cache_folder, "ONGOING_" + current_hyperparameters_hash) + '.json'

    def _get_new_trial_json_file_path(self, trial: 'Trial'):
        """
        Get new trial json path.
        """
        hp_dict = trial.hyperparams.to_flat_dict()
        current_hyperparameters_hash = self._get_trial_hash(hp_dict)
        return os.path.join(self.cache_folder, "NEW_" + current_hyperparameters_hash) + '.json'

    def _remove_previous_trial_state_json(self):
        if self.json_path_remove_on_update and os.path.exists(self.json_path_remove_on_update):
            os.remove(self.json_path_remove_on_update)

    def subscribe_to_cache_folder_changes(self, refresh_interval_in_seconds: int,
                                          observer: _Observer[Tuple[HyperparamsRepository, Trial]]):
        """
        Every refresh_interval_in_seconds

        :param refresh_interval_in_seconds: number of seconds to wait before sending updates to the observers
        :param observer:
        :return:
        """
        self._observers.add(observer)
        # TODO: start a process that notifies observers anytime a the file of a trial changes
        # possibly use this ? https://github.com/samuelcolvin/watchgod
        # note: this is how you notify observers self.on_next((self, updated_trial))


class BaseHyperparameterOptimizer(ABC):

    def __init__(self, main_metric_name: str = None):
        """
        :param main_metric_name: if None, pick first metric from the metrics callback.
        """
        self.main_metric_name = main_metric_name

    @abstractmethod
    def find_next_best_hyperparams(self, auto_ml_container) -> HyperparameterSamples:
        """
        Find the next best hyperparams using previous trials.

        :param auto_ml_container: trials data container
        :return: next best hyperparams
        """
        raise NotImplementedError()


class AutoMLFlow(Flow):

    def __init__(self, repo: HyperparamsRepository, logger: logging.Logger = None, **kwargs):
        super().__init__(logger=logger)
        self.repo: HyperparamsRepository = repo

    def start_run(self) -> 'TrialFlow':
        """
        Start a new run.
        :return:
        """
        raise NotImplementedError("")

    def update(
        self,
        project_id: str,
        client_id: str,
        run_id: int,
        trial_id: int,
        new_val: Any
    ):
        raise NotImplementedError("")


class TrialFlow(MetaService):
    def __init__(self, wrapped: Flow):
        MetaService.__init__(self, wrapped=wrapped)


class Trainer(BaseService):
    """
    Class used to train a pipeline using various data splits and callbacks for evaluation purposes.

    # TODO: add this `with_val_set` method that would change splitter to PresetValidationSetSplitter(self, val) and override.
    """

    def __init__(
            self,
            validation_splitter: 'BaseValidationSplitter',
            callbacks: List[BaseCallback] = None,
            n_epochs: int = 1,
    ):
        BaseService.__init__(self)
        self.validation_splitter: BaseValidationSplitter = validation_splitter
        self.callbacks: CallbackList = CallbackList(callbacks) if callbacks is not None else []
        self.n_epochs = 1

    def train(
        self,
        pipeline: BaseStep,
        dact: DACT,
        context: ExecutionContext = None,
    ) -> Trial:
        """
        Train pipeline using the validation splitter.
        Track training, and validation metrics for each epoch.
        Note: the present method is just a shortcut to using the `execute_trial` method with less boilerplate code needed. Refer to `execute_trial` for full flexibility

        :param pipeline: pipeline to train on
        :param data_inputs: data inputs
        :param expected_outputs: expected ouptuts to fit on
        :return: executed trial

        """

        splits: List[Tuple[DACT, DACT]] = self.validation_splitter.split_dact(dact, context=context)

        for train_dact, val_dact in splits:
            p = copy.deepcopy(pipeline)

            with context.flow.log_split():
                # TODO: this method returns a context manager that takes care of the rest, and which itself has the flow as an attribute.

                # TODO: failsafe on the mlflow for the clients and projects. Example: no clients and no projects? use the default ones. No split? use a default single split. Log warning if so?

                for i in range(self.epochs):

                    with context.flow.log_epoch(i, self.epochs):
                        # TODO: this method returns a context manager that takes care of the rest, and which itself has the flow as an attribute.

                        p = p.set_train(True)
                        p = p.handle_fit(train_dact.copy(), context.train())

                        y_pred_train = p.handle_predict(train_dact.copy(), context.validation())
                        y_pred_val = p.handle_predict(val_dact.copy(), context.validation())

                        if self.callbacks.call(
                                context=context.copy().set_execution_phase(ExecutionPhase.VALIDATION),
                                input_train=train_dact,
                                pred_train=y_pred_train,
                                input_val=val_dact,
                                pred_val=y_pred_val,
                        ):
                            break
                            # Saves the metrics with flow split exit.

                # TODO: log success in the __exit__ method(s).
                context.flow.log_success()

    def refit(self, p: BaseStep, data_container: DACT, context: ExecutionContext) -> BaseStep:
        """
        Refit the pipeline on the whole dataset (without any validation technique).

        :param p: trial to refit
        :param data_container: data container
        :param context: execution context

        :return: fitted pipeline
        """
        context.set_execution_phase(ExecutionPhase.TRAIN)
        for i in range(self.epochs):
            p = p.handle_fit(data_container, context)
            # TODO: log retraing metrics outside of a split by using the split abstraction again but outside split list?
        return p


class BaseControllerLoop(TruncableService):

    def __init__(
        self,
        trainer: Trainer,
        n_trials: int,
        hyperparameter_optimizer: BaseHyperparameterOptimizer = None,
        continue_loop_on_error: bool = True
    ):
        if hyperparameter_optimizer is None:
            hyperparameter_optimizer = RandomSearch()

        TruncableService.__init__(self, {
            Trainer: trainer,
            BaseHyperparameterOptimizer: hyperparameter_optimizer,
        })
        self.n_trials = n_trials

        self.error_types_to_raise = (
            SystemError, SystemExit, EOFError, KeyboardInterrupt) if continue_loop_on_error else (Exception,)

    def run(self, pipeline, dact: DACT, context: ExecutionContext):
        """
        Run the controller loop.

        :param context: execution context
        :return:
        """
        raise NotImplementedError("")

    def _get_next(self):
        """
        Distributed or parallel for loop like `for i in n_trials`.
        """
        for i in range(self.n_trials):
            yield i

    def loop(self, context: ExecutionContext):
        """
        Loop over all trials.

        :param context: execution context
        :return:
        """
        for i in self._get_next():
            try:
                yield self.trainer.train(i, context)
            except Exception as e:
                if self.continue_loop_on_error:
                    context.logger.error('error {}'.format(e))
                else:
                    raise e


class DefaultLoop(BaseControllerLoop):

    def __init__(
        self,
        trainer: Trainer,
        n_trials: int,
        n_jobs: int = 1,
        hyperparams_optimizer: BaseHyperparameterOptimizer = None,
        continue_loop_on_error: bool = True,
    ):
        super().__init__(
            trainer,
            n_trials,
            hyperparameter_optimizer=hyperparams_optimizer,
            continue_loop_on_error=continue_loop_on_error
        )
        self.n_jobs = n_jobs

    def loop(self, repo_run, p: BaseStep, data_container: DACT, context: ExecutionContext):
        for i in self._get_next():

            # TDA for lock unlock to be all in new_trial().
            repo_run.lock()
            p = self.next_best_prediction_algo.prepare_new_trial(p, repo_run.all_trials())
            hps = p.get_hyperparams()
            with repo_run.new_trial(hps) as trial:
                repo_run.unlock()
                # assert trial.is_started()
                # assert not trial.is_finished()

                trial_result: TrialResult = self.trainer.train(
                    p,
                    data_container,
                    context.with_trial(trial),
                    # this param is for allowing hyperband to easily control the n_epochs in percents...
                )

                # assert trial.repo_run == repo_run
                # if isinstance(trial.repo_run.repo, PostGreSQLHPRepo): assert hasattr(trial.repo_run.repo.db.connection)
                trial.set_finished(p, trial_result)

        _comment = """
            def _next_best_prediction_algo.prepare_new_trial(p: BaseStep, past_trials_with_scores) -> BaseStep:
                trial_p = p.copy()

                space = trial_p.get_hyperparam_space()
                rvs = next_best_prediction_algo.rvs(space, past_trials_with_scores)  # can have NULL scores as well.
                trial_p.set_hyperparams(rvs)
                return trial_p
        """


class AutoML(ForceHandleMixin, _HasChildrenMixin, BaseStep):
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
        :class:`BaseValidationSplitter`,
        :class:`HyperparamsJSONRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(
            self,
            pipeline: BaseStep,
            controller_loop: BaseControllerLoop,  # HyperbandControllerLoop(), ClusteringParallelFor()
            flow: Flow,
            start_new_run: bool = True,  # otherwise, pick last run. TODO: here?
            refit_best_trial: bool = True,
            hyperparams_optimizer: BaseHyperparameterOptimizer = None,
            hyperparams_repository: HyperparamsRepository = None,
    ):
        """
        Notes on multiprocess :
              Usage of a multiprocess-safe hyperparams repository is recommended, although it is, most of the time, not necessary.
              Beware of the behaviour of HyperparamsRepository's observers/subscribers.
              Context instances are not shared between trial but copied. So is the AutoML loop and the DACTs.


        :param pipeline: The pipeline, or BaseStep, which will be use by the AutoMLloop
        :param validation_splitter: A :class:`BaseValidationSplitter` instance to split data between training and validation set.
        :param refit_trial: A boolean indicating whether to perform, after ,  a fit call with
        :param scoring_callback: The scoring callback to use during training
        :param hyperparams_optimizer: a :class:`BaseHyperparameterSelectionStrategy` instance that can be queried for new sets of hyperparameters.
        :param hyperparams_repository: a :class:`HyperparamsRepository` instance to store experiement status and results.
        :param n_trials: The number of different hyperparameters to try.
        :param epochs: The number of epoch to perform for each trial.
        :param callbacks: A list of callbacks to perform after each epoch.
        :param cache_folder_when_no_handle: default cache folder used if auto_ml_loop isn't called through handler functions.
        :param n_jobs: If n_jobs in (None, 1), then automl is executed in a single process, which may spawns on multiple thread. if n_jobs > 1, then n_jobs process are launched, if n_jobs <= -1 then (n_cpus + 1 + n_jobs) process are launched. One trial at a time is executed by process.
        :param continue_loop_on_error:
        """
        BaseStep.__init__(self)
        _HasChildrenMixin.__init__(self)
        ForceHandleMixin.__init__(self)

        self.pipeline: BaseStep = pipeline
        self.hyperparameter_optimizer: BaseHyperparameterOptimizer = hyperparams_optimizer or RandomSearch()
        self.refit_best_trial: bool = refit_best_trial
        self.controller_loop: BaseControllerLoop = controller_loop

    def get_children(self) -> List[BaseServiceT]:
        return [self.pipeline]

    def _fit_data_container(self, data_container: DACT, context: ExecutionContext) -> 'BaseStep':
        """
        Run Auto ML Loop.
        Find the best hyperparams using the hyperparameter optmizer.
        Evaluate the pipeline on each trial using a validation technique.

        :param data_container: data container to fit
        :param context: execution context

        :return: self
        """
        validation_splits = self.validation_splitter.split_dact(
            data_container=data_container,
            context=context
        )

        # Keeping a reference of the main logger
        main_logger = context.logger

        if self.n_jobs in (None, 1):
            # Single Process
            for trial_number in range(self.n_trial):
                self._attempt_trial(trial_number, validation_splits, context)
        else:
            # Multiprocssing
            context.logger.info(f"Number of processors available: {multiprocessing.cpu_count()}")

            if isinstance(self.hyperparams_repository, InMemoryHyperparamsRepository):
                raise ValueError(
                    "Cannot use InMemoryHyperparamsRepository for multiprocessing, use json-based repository.")

            n_jobs = self.n_jobs
            if n_jobs <= -1:
                n_jobs = multiprocessing.cpu_count() + 1 + self.n_jobs

            with multiprocessing.get_context("spawn").Pool(processes=n_jobs) as pool:
                args = [(self, trial_number, validation_splits, context) for trial_number in range(self.n_trial)]
                pool.starmap(AutoML._attempt_trial, args)

        context.set_logger(main_logger)

        best_hyperparams = self.hyperparams_repository.get_best_hyperparams()

        context.logger.info(
            '\nbest hyperparams: {}'.format(json.dumps(best_hyperparams.to_nested_dict(), sort_keys=True, indent=4)))

        # Notify HyperparamsRepository subscribers
        self.hyperparams_repository.notify_complete(value=self.hyperparams_repository)

        if self.refit_trial:
            p: BaseStep = self._load_virgin_model(hyperparams=best_hyperparams)
            p = self.trainer.refit(
                p=p,
                data_container=data_container,
                context=context.set_execution_phase(ExecutionPhase.TRAIN)
            )

            self.hyperparams_repository.save_best_model(p)

        return self

    def _attempt_trial(self, trial_number, validation_splits, context: ExecutionContext):

        try:
            auto_ml_data = AutoMLContainer(
                trial_number=trial_number,
                trials=self.hyperparams_repository.load_trials(TrialStatus.SUCCESS),
                hyperparameter_space=self.pipeline.get_hyperparams_space(),
                main_scoring_metric_name=self.trainer.get_main_metric_name()
            )

            with self.hyperparams_repository.new_trial(auto_ml_data) as repo_trial:
                repo_trial_split = None
                context.set_logger(repo_trial.logger)
                context.logger.info('trial {}/{}'.format(trial_number + 1, self.n_trial))

                repo_trial_split = self.trainer.execute_trial(
                    pipeline=self.pipeline,
                    context=context,
                    repo_trial=repo_trial,
                    validation_splits=validation_splits,
                    n_trial=self.n_trial
                )
        except self.error_types_to_raise as error:
            track = traceback.format_exc()
            repo_trial.set_failed(error)
            context.logger.critical(track)
            raise error
        except Exception:
            track = traceback.format_exc()
            repo_trial_split_number = 0 if repo_trial_split is None else repo_trial_split.split_number + 1
            context.logger.error('failed trial {}'.format(_get_trial_split_description(
                repo_trial=repo_trial,
                repo_trial_split_number=repo_trial_split_number,
                validation_splits=validation_splits,
                trial_number=trial_number,
                n_trial=self.n_trial
            )))
            context.logger.error(track)
        finally:
            repo_trial.update_final_trial_status()
            # Some heavy objects might have stayed in memory for a while during the execution of our trial;
            # It is best to do a full collection as that may free up some ram.
            gc.collect()

    def _fit_transform_data_container(self, data_container: DACT, context: ExecutionContext) -> \
            ('BaseStep', DACT):
        raise NotImplementedError("AutoML does not implement method _fit_transform_data_container. Use method such as "
                                  "fit or handle_fit to train models and then use method such as get_best_model to "
                                  "retrieve the model you wish to use for transform")

    def _transform_data_container(self, data_container: DACT, context: ExecutionContext) -> DACT:
        return self.wrapped.handle_transform(data_container, context)

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

    def get_best_model(self) -> BaseStep:
        """
        Get best model using the hyperparams repository.

        :return:
        """
        return self.hyperparams_repository.get_best_model()

    def get_children(self) -> List[BaseStep]:
        return [self.get_best_model()]

    @property
    def wrapped(self) -> BaseStep:
        return self.get_children()


def _get_trial_split_description(
        repo_trial: Trial,
        repo_trial_split_number: int,
        validation_splits: List[Tuple[DACT, DACT]],
        trial_number: int,
        n_trial: int
):
    trial_split_description = '{}/{} split {}/{}\nhyperparams: {}'.format(
        trial_number + 1,
        n_trial,
        repo_trial_split_number + 1,
        len(validation_splits),
        json.dumps(repo_trial.hyperparams, sort_keys=True, indent=4)
    )
    return trial_split_description


class EasyAutoML(AutoML):
    """
    This is a wrapper to the old version of the AutoML module.
    It is kept for easier backwards compatibility. It also provides a
    nice interface to easily use the AutoML module.

    :param pipeline: pipeline to copy and use for training
    :param validation_splitter: validation splitter to use
    :param refit_trial: whether to refit the best model on the whole dataset after the optimization
    :param scoring_callback: main callback to use for scoring, that is deprecated
    :param hyperparams_optimizer: hyperparams optimizer to use
    :param hyperparams_repository: hyperparams repository to use
    :param n_trials: number of trials to run
    :param epochs: number of epochs to train the model for each val split
    :param callbacks: callbacks to use for training - there can be aditionnal metrics there
    :param refit_scoring_function: scoring function to use for refitting the best model
    :param cache_folder_when_no_handle: folder to use for caching when no handle is provided
    :param n_jobs: number of jobs to use for parallelization, defaults is None for no parallelization
    :param continue_loop_on_error: whether to continue the main optimization loop on error or not
    :return: AutoML objectr ready to use
    """

    def __init__(
        self,
        pipeline: BaseStep,
        validation_splitter: 'BaseValidationSplitter',
        refit_best_trial: bool,
        scoring_callback: ScoringCallback,
        hyperparams_optimizer: BaseHyperparameterOptimizer = None,
        hyperparams_repository: HyperparamsRepository = None,
        n_trials: int = 10,
        epochs: int = 1,
        callbacks: List[BaseCallback] = None,
        cache_folder_when_no_handle=None,
        n_jobs=None,
        continue_loop_on_error=True
    ):
        warn_deprecated_class(self, AutoML)
        trainer = Trainer(
            callbacks=[scoring_callback] + callbacks,
            validation_splitter=validation_splitter,
            n_epochs=epochs,
        )
        controller_loop = DefaultLoop(
            trainer=trainer,
            n_trials=n_trials,
            n_jobs=n_jobs,
            hyperparams_optimizer=hyperparams_optimizer,
            continue_loop_on_error=continue_loop_on_error,
        )
        flow: Flow = AutoMLFlow(repo=hyperparams_repository)
        assert cache_folder_when_no_handle is None  # TODO: remove this.

        AutoML.__init__(
            self,
            pipeline=pipeline,
            controller_loop=controller_loop,
            flow=flow,
            refit_best_trial=refit_best_trial,
            start_new_run=True,
        )


class RandomSearch(BaseHyperparameterOptimizer):
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

    def __init__(self, main_metric_name: str = None):
        BaseHyperparameterOptimizer.__init__(self, main_metric_name=main_metric_name)

    def find_next_best_hyperparams(self, auto_ml_container) -> HyperparameterSamples:
        """
        Randomly sample the next hyperparams to try.

        :param auto_ml_container: trials data container
        :return: next best hyperparams
        """
        return auto_ml_container.hyperparameter_space.rvs()


class BaseValidationSplitter(ABC):
    def split_dact(self, data_container: DACT, context: ExecutionContext) -> List[
            Tuple[DACT, DACT]]:
        """
        Wrap a validation split function with a split data container function.
        A validation split function takes two arguments:  data inputs, and expected outputs.

        :param data_container: data container to split
        :return: a function that returns the pairs of training, and validation data containers for each validation split.
        """
        train_data_inputs, train_expected_outputs, train_ids, validation_data_inputs, validation_expected_outputs, validation_ids = self.split(
            data_inputs=data_container.data_inputs,
            expected_outputs=data_container.expected_outputs,
            context=context
        )

        train_data_container = DACT(data_inputs=train_data_inputs,
                                    ids=train_ids,
                                    expected_outputs=train_expected_outputs)
        validation_data_container = DACT(data_inputs=validation_data_inputs,
                                         ids=validation_ids,
                                         expected_outputs=validation_expected_outputs)

        splits = []
        for (train_id, train_di, train_eo), (validation_id, validation_di, validation_eo) in zip(
                train_data_container, validation_data_container):
            # TODO: use ListDACT instead of DACT
            train_data_container_split = DACT(
                data_inputs=train_di,
                expected_outputs=train_eo
            )

            validation_data_container_split = DACT(
                data_inputs=validation_di,
                expected_outputs=validation_eo
            )

            splits.append((train_data_container_split, validation_data_container_split))

        return splits

    @abstractmethod
    def split(self, data_inputs, ids=None, expected_outputs=None, context: ExecutionContext = None) \
            -> Tuple[List, List, List, List, List, List]:
        """
        Train/Test split data inputs and expected outputs.

        :param data_inputs: data inputs
        :param ids: id associated with each data entry (optional)
        :param expected_outputs: expected outputs (optional)
        :param context: execution context (optional)
        :return: train_data_inputs, train_expected_outputs, train_ids, validation_data_inputs, validation_expected_outputs, validation_ids
        """
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

    def split(self, data_inputs, ids=None, expected_outputs=None, context: ExecutionContext = None) \
            -> Tuple[List, List, List, List, List, List]:
        data_inputs_train, data_inputs_val = kfold_cross_validation_split(
            data_inputs=data_inputs,
            k_fold=self.k_fold
        )

        if ids is not None:
            ids_train, ids_val = kfold_cross_validation_split(
                data_inputs=ids,
                k_fold=self.k_fold
            )
        else:
            ids_train, ids_val = [None] * len(data_inputs_train), [None] * len(data_inputs_val)

        if expected_outputs is not None:
            expected_outputs_train, expected_outputs_val = kfold_cross_validation_split(
                data_inputs=expected_outputs,
                k_fold=self.k_fold
            )
        else:
            expected_outputs_train, expected_outputs_val = [None] * len(data_inputs_train), [None] * len(
                data_inputs_val)

        return data_inputs_train, expected_outputs_train, ids_train, \
            data_inputs_val, expected_outputs_val, ids_val


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

    def split(
        self, data_inputs, ids=None, expected_outputs=None, context: ExecutionContext = None
    ) -> Tuple[List, List, List, List]:
        train_data_inputs, train_expected_outputs, train_ids, validation_data_inputs, validation_expected_outputs, validation_ids = validation_split(
            test_size=self.test_size,
            data_inputs=data_inputs,
            ids=ids,
            expected_outputs=expected_outputs
        )

        return [train_data_inputs], [train_expected_outputs], [train_ids], \
               [validation_data_inputs], [validation_expected_outputs], [validation_ids]


def validation_split(test_size: float, data_inputs, ids=None, expected_outputs=None) \
        -> Tuple[List, List, List, List, List, List]:
    """
    Split data inputs, and expected outputs into a training set, and a validation set.

    :param test_size: test size in float
    :param data_inputs: data inputs to split
    :param ids: ids associated with each data entry
    :param expected_outputs: expected outputs to split
    :return: train_data_inputs, train_expected_outputs, ids_train, validation_data_inputs, validation_expected_outputs, ids_val
    """
    validation_data_inputs = _validation_split(data_inputs, test_size)
    validation_expected_outputs, ids_val = None, None
    if expected_outputs is not None:
        validation_expected_outputs = _validation_split(expected_outputs, test_size)
    if ids is not None:
        ids_val = _validation_split(ids, test_size)

    train_data_inputs = _train_split(data_inputs, test_size)
    train_expected_outputs, ids_train = None, None
    if expected_outputs is not None:
        train_expected_outputs = _train_split(expected_outputs, test_size)
    if ids is not None:
        ids_train = _train_split(ids, test_size)

    return train_data_inputs, train_expected_outputs, ids_train, \
        validation_data_inputs, validation_expected_outputs, ids_val


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
