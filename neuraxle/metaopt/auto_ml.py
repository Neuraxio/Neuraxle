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

from asyncore import loop
import copy
import gc
import json
import multiprocessing
import traceback
from operator import attrgetter
from tracemalloc import start
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
from neuraxle.base import (BaseService, BaseServiceT, BaseStep,
                           ExecutionContext, ExecutionPhase, Flow,
                           ForceHandleMixin, TrialStatus, TruncableService,
                           _HasChildrenMixin)
from neuraxle.data_container import DACT as DACT
from neuraxle.data_container import IDT
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace)
from neuraxle.logging.warnings import (warn_deprecated_arg,
                                       warn_deprecated_class)
from neuraxle.metaopt.callbacks import (ARG_Y_EXPECTED, ARG_Y_PREDICTD,
                                        BaseCallback, CallbackList,
                                        ScoringCallback)
from neuraxle.metaopt.data.aggregates import (Client, ManageableT, Project, Root, Round,
                                              Trial, TrialSplit)
from neuraxle.metaopt.data.vanilla import (DEFAULT_CLIENT, DEFAULT_PROJECT,
                                           AutoMLContext, BaseDataclass,
                                           ClientDataclass,
                                           HyperparamsRepository,
                                           InMemoryHyperparamsRepository,
                                           MetricResultsDataclass,
                                           ProjectDataclass, RecursiveDict,
                                           RootDataclass, RoundDataclass,
                                           ScopedLocation, SubDataclassT,
                                           TrialDataclass, TrialSplitDataclass)
from neuraxle.metaopt.validation import (BaseCrossValidationWrapper,
                                         BaseHyperparameterOptimizer,
                                         BaseValidationSplitter, RandomSearch)


class Trainer(BaseService):
    """
    Class used to train a pipeline using various data splits and callbacks for evaluation purposes.

    # TODO: add this `with_val_set` method that would change splitter to
    # PresetValidationSetSplitter(self, val) and override.
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
        self.n_epochs = n_epochs

    def train(
        self,
        pipeline: BaseStep,
        dact: DACT,
        trial_scope: Trial,
    ):
        """
        Train pipeline using the validation splitter.
        Track training, and validation metrics for each epoch.
        Note: the present method is just a shortcut to using the `execute_trial` method with less boilerplate code needed. Refer to `execute_trial` for full flexibility
        """

        splits: List[Tuple[DACT, DACT]] = self.validation_splitter.split_dact(
            dact, context=trial_scope.context)

        for train_dact, val_dact in splits:
            with trial_scope.new_validation_split() as trial_split_scope:
                self.train_split(pipeline, train_dact, val_dact, trial_split_scope)

    def train_split(
        self,
        pipeline: BaseStep,
        train_dact: DACT,
        val_dact: Optional[DACT],
        trial_split_scope: TrialSplit
    ):
        """
        Train a pipeline split. You probably want to use `self.train` instead, to use the validation splitter.
        If validation DACT is None, the evaluation metrics will not save validation results.
        """
        trial_split_scope: TrialSplit = trial_split_scope.with_n_epochs(self.n_epochs)
        p: BaseStep = pipeline.copy(trial_split_scope.context, deep=True)
        p.set_hyperparams(trial_split_scope.get_hyperparams())

        for _ in range(self.n_epochs):
            e = trial_split_scope.next_epoch()

            # Fit train
            p = p.set_train(True)
            p = p.handle_fit(
                train_dact.copy(),
                trial_split_scope.context.train())

            # Predict train & val
            p = p.set_train(False)
            eval_dact_train = p.handle_predict(
                train_dact.without_eo(),
                trial_split_scope.context.validation())
            eval_dact_train: DACT[IDT, ARG_Y_PREDICTD, ARG_Y_EXPECTED] = eval_dact_train.with_eo(train_dact.eo)

            if val_dact is not None:
                eval_dact_valid = p.handle_predict(
                    val_dact.without_eo(),
                    trial_split_scope.context.validation())
                eval_dact_valid: DACT[IDT, ARG_Y_PREDICTD, ARG_Y_EXPECTED] = eval_dact_valid.with_eo(val_dact.eo)
            else:
                eval_dact_valid = None

            # Log metrics / evaluate
            if self.callbacks.call(
                trial_split_scope,
                eval_dact_train,
                eval_dact_valid,
                e == self.n_epochs
            ):
                break  # Saves stats using the '__exit__' method of managed scoped aggregates.

    def refit(
        self,
        pipeline: BaseStep,
        dact: DACT,
        trial_scope: Trial,
    ) -> BaseStep:
        """
        Refit the pipeline on the whole dataset (without any validation technique).

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
        hp_optimizer: BaseHyperparameterOptimizer = None,
        continue_loop_on_error: bool = True
    ):
        if hp_optimizer is None:
            hp_optimizer = RandomSearch()

        TruncableService.__init__(self, {
            Trainer: trainer,
            BaseHyperparameterOptimizer: hp_optimizer,
        })
        self.n_trials = n_trials
        self.continue_loop_on_error: bool = continue_loop_on_error

    @property
    def trainer(self) -> Trainer:
        return self.get_service(Trainer)

    def run(self, pipeline: BaseStep, dact: DACT, round: Round, refit_best_trial: bool) -> int:
        """
        Run the controller loop.

        :param context: execution context
        :return: the ID of the round that was executed (either created or continued from previous optimization).
        """

        # thread_safe_lock, context = context.thread_safe() ??

        hp_optimizer: BaseHyperparameterOptimizer = self[BaseHyperparameterOptimizer]
        hp_space: HyperparameterSpace = pipeline.get_hyperparams_space()

        round: Round = round.with_optimizer(hp_optimizer, hp_space)
        if self.continue_loop_on_error:
            round.continue_loop_on_error()

        for managed_trial_scope in self.loop(round):
            managed_trial_scope: Trial = managed_trial_scope  # typing helps
            # trial_scope.context.restore_lock(thread_safe_lock) ??
            self.trainer.train(pipeline, dact, managed_trial_scope)

        round_id: int = round.get_id()
        return round_id

    def loop(self, round_scope: Round) -> Iterator[Trial]:
        """
        Loop over all trials.

        :param dact: data container that is not yet splitted
        :param context: execution context
        :return:
        """
        for _ in range(self.n_trials):
            with round_scope.new_rvs_trial() as managed_trial_scope:
                managed_trial_scope: Trial = managed_trial_scope  # typing helps

                if self.continue_loop_on_error:
                    managed_trial_scope.continue_loop_on_error()

                yield managed_trial_scope

    def next_trial(self, round_scope: Round) -> ManageableT[Trial]:
        """
        Get the next trial to be executed.

        :param round_scope: round scope
        :return: the next trial to be executed.
        """
        return round_scope.new_rvs_trial()  # TODO: finish for parallelization

    def refit_best_trial(self, pipeline: BaseStep, dact: DACT, round_scope: Round) -> BaseStep:
        """
        Refit the pipeline on the whole dataset (without any validation technique).
        """
        with round_scope.managed_best_trial() as managed_trial_scope:

            refitted: BaseStep = self.trainer.refit(
                pipeline, dact, managed_trial_scope)

        return refitted


class DefaultLoop(BaseControllerLoop):

    def __init__(
        self,
        trainer: Trainer,
        n_trials: int,
        hp_optimizer: BaseHyperparameterOptimizer = None,
        continue_loop_on_error: bool = True,
        n_jobs: int = 1,
    ):
        super().__init__(
            trainer,
            n_trials,
            hp_optimizer=hp_optimizer,
            continue_loop_on_error=continue_loop_on_error
        )
        self.n_jobs = n_jobs

    def TODO_loop(self, pipeline, dact: DACT, context: ExecutionContext):
        # TODO: what is this method used for?

        if self.n_jobs in (None, 1):
            # Single Process
            # for i in i super().loop...
            for trial_number in range(self.n_trial):
                self._attempt_trial(trial_number, validation_splits, context)

        else:
            # Multiprocssing
            # dispatch task to each process where each fetch an interation of the super.loop??
            # todo: refactor this method to make the loop not necessarily a generator.
            #       this way, it could be possible to call "self.next" in threads to do the same.

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


class AutoML(ForceHandleMixin, _HasChildrenMixin, BaseStep):
    """
    A step to execute Automated Machine Learning (AutoML) algorithms. This step will
    automatically split the data into train and validation splits, and execute an
    hyperparameter optimization on the splits to find the best hyperparameters.

    The step with the chosen good hyperparameters will be refitted to the full
    unsplitted data if desired.
    """

    def __init__(
            self,
            pipeline: BaseStep,
            loop: BaseControllerLoop,
            repo: HyperparamsRepository,
            main_metric_name: str,
            start_new_round: bool = True,
            refit_best_trial: bool = True,
            project_name: str = DEFAULT_PROJECT,
            client_name: str = DEFAULT_CLIENT,
    ):
        """
        .. note::
            Usage of a multiprocess-safe hyperparams repository is recommended,
            although it is, most of the time, not necessary.
            Beware of the behaviour of HyperparamsRepository's observers/subscribers.
            Context instances are not shared between trial but copied.
            So is the AutoML loop and the DACTs.

        :param pipeline: The pipeline, or BaseStep, which will be use by the AutoMLloop
        :param loop: The loop, or BaseControllerLoop, which will be used by the AutoML loop
        :param flow: The flow, or Flow, which will be used by the AutoML loop
        :param refit_best_trial: A boolean indicating whether to perform, after a fit call, a refit on the best trial.
        """
        BaseStep.__init__(self)
        _HasChildrenMixin.__init__(self)
        ForceHandleMixin.__init__(self)

        self.pipeline: BaseStep = pipeline
        self.loop: BaseControllerLoop = loop
        self.repo: HyperparamsRepository = repo

        self.main_metric_name: str = main_metric_name
        self.start_new_round: bool = start_new_round
        self.refit_best_trial: bool = refit_best_trial
        self.project_name: str = project_name
        self.client_name: str = client_name

        self.has_model_been_retrained: bool = False

    def get_children(self) -> List[BaseStep]:
        return [self.pipeline]

    @property
    def wrapped(self) -> BaseStep:
        return self.pipeline

    def _fit_transform_data_container(
        self, data_container: DACT, context: ExecutionContext
    ) -> Tuple['BaseStep', DACT]:
        if not self.refit_best_trial:
            raise ValueError(
                "self.refit_best_trial must be True in this AutoML class to do the transform in 'fit_transform'.")

        self = self._fit_data_container(data_container, context)
        data_container = self._transform_data_container()
        return self, data_container

    def _fit_data_container(self, data_container: DACT, context: ExecutionContext) -> 'BaseStep':
        """
        Run Auto ML Loop.
        Find the best hyperparams using the hyperparameter optmizer.
        Evaluate the pipeline on each trial using a validation technique.

        :param data_container: data container to fit
        :param context: execution context

        :return: self
        """
        automl_context: AutoMLContext = AutoMLContext.from_context(context, repo=self.repo)
        root: Root = Root.from_context(automl_context)

        with root.get_project(self.project_name) as ps:
            ps: Project = ps
            with ps.get_client(self.client_name) as cs:
                cs: Client = cs
                with cs.optim_round(self.start_new_round, self.main_metric_name) as rs:
                    rs: Round = rs

                    self.loop.run(self.pipeline, data_container, rs)

                    if self.refit_best_trial:
                        self.pipeline = self.loop.refit_best_trial(self.pipeline, data_container, rs)
                        self.has_model_been_retrained = True

        return self

    def _transform_data_container(self, data_container: DACT, context: ExecutionContext) -> DACT:
        if not self.has_model_been_retrained:
            raise ValueError('self.refit_best_trial must be True in AutoML class to transform.')

        return self.pipeline.handle_transform(data_container, context)


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
    :return: AutoML object ready to use
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
            hp_optimizer=hyperparams_optimizer,
            continue_loop_on_error=continue_loop_on_error,
            start_new_round=True,
        )
        assert cache_folder_when_no_handle is None  # TODO: remove this.

        AutoML.__init__(
            self,
            pipeline=pipeline,
            loop=controller_loop,
            repo=hyperparams_repository,
            refit_best_trial=refit_best_trial,
        )
