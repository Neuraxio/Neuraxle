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
import json
import logging
import math
import multiprocessing
import os
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, List, Optional, Tuple, Type, Union

import numpy as np
from neuraxle.base import (BaseService, BaseServiceT, BaseStep,
                           ExecutionContext, ExecutionPhase, Flow,
                           ForceHandleMixin, MetaService, MetaServiceMixin,
                           TrialStatus, TruncableService, _HasChildrenMixin,
                           synchroneous_flow_method)
from neuraxle.data_container import DACT as DACT
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace)
from neuraxle.logging.warnings import (warn_deprecated_arg,
                                       warn_deprecated_class)
from neuraxle.metaopt.callbacks import (BaseCallback, CallbackList,
                                        ScoringCallback)
from neuraxle.metaopt.data.trial import (RoundManager, RoundScope,
                                         TrialManager, TrialScope,
                                         TrialSplitManager)
from neuraxle.metaopt.data.vanilla import (BaseDataclass, ClientDataclass,
                                           HyperparamsRepository,
                                           MetricResultsDataclass,
                                           ProjectDataclass, RecursiveDict,
                                           RootMetadata, RoundDataclass,
                                           ScopedLocation, SubDataclassT,
                                           TrialDataclass, TrialSplitDataclass)
from neuraxle.metaopt.validation import (BaseCrossValidationWrapper,
                                         BaseHyperparameterOptimizer,
                                         RandomSearch)


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
        self.n_epochs = n_epochs

    def train(
        self,
        pipeline: BaseStep,
        dact: DACT,
        trial_scope: TrialScope,
    ) -> TrialManager:
        """
        Train pipeline using the validation splitter.
        Track training, and validation metrics for each epoch.
        Note: the present method is just a shortcut to using the `execute_trial` method with less boilerplate code needed. Refer to `execute_trial` for full flexibility

        :param pipeline: pipeline to train on
        :param data_inputs: data inputs
        :param expected_outputs: expected ouptuts to fit on
        :return: executed trial

        """

        splits: List[Tuple[DACT, DACT]] = self.validation_splitter.split_dact(dact, context=trial_scope.context)

        for train_dact, val_dact in splits:
            with trial_scope.new_trial_split(self.n_epochs) as trial_split_scope:
                trial_split_scope: TrialSplitScope = trial_split_scope  # typing helps

                # TODO: No split? use a default single split. Log warning if so?

                p = pipeline.copy(trial_split_scope.context).set_hyperparams(trial_scope.hps)

                for e in range(self.n_epochs):
                    with trial_split_scope.new_epoch(e) as epoch_scope:
                        epoch_scope: EpochScope = epoch_scope  # typing helps

                        p = p.set_train(True)
                        p = p.handle_fit(train_dact.copy(), epoch_scope.context.train())

                        y_pred_train = p.handle_predict(train_dact.copy(), epoch_scope.context.validation())
                        y_pred_val = p.handle_predict(val_dact.copy(), epoch_scope.context.validation())

                        if self.callbacks.call(
                                context=epoch_scope.context.validation(),
                                input_train=train_dact,
                                pred_train=y_pred_train,
                                input_val=val_dact,
                                pred_val=y_pred_val,
                        ):
                            break
                            # Saves the metrics with flow split exit.

                # TODO: log success in the __exit__ method(s).

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

    def run(self, pipeline: BaseStep, dact: DACT, context: ExecutionContext):
        """
        Run the controller loop.

        :param context: execution context
        :return:
        """
        trainer: Trainer = self[Trainer]
        hp_space: HyperparameterSpace = pipeline.get_hyperparams_space()
        # thread_safe_lock, context = context.thread_safe()

        # TODO: failsafe on the mlflow for the clients and projects.
        #           Example: no clients and no projects? use the default ones.
        with RoundScope(context, hp_space) as round_scope:
            round_scope: RoundScope

            for trial_scope in self.loop(round_scope):
                trial_scope: TrialScope = trial_scope  # typing helps
                # trial_scope.context.restore_lock(thread_safe_lock)

                trainer.train(pipeline, dact, trial_scope)

    def loop(self, round_scope: RoundScope) -> Iterator[TrialScope]:
        """
        Loop over all trials.

        :param dact: data container that is not yet splitted
        :param context: execution context
        :return:
        """
        hp_optimizer: BaseHyperparameterOptimizer = self[BaseHyperparameterOptimizer]

        for _ in range(self.n_trials):
            with round_scope.new_hyperparametrized_trial(hp_optimizer, self.continue_loop_on_error) as trial_scope:
                trial_scope: TrialScope = trial_scope  # typing helps

                yield trial_scope


class DefaultLoop(BaseControllerLoop):

    def __init__(
        self,
        trainer: Trainer,
        n_trials: int,
        n_jobs: int = 1,
        hp_optimizer: BaseHyperparameterOptimizer = None,
        continue_loop_on_error: bool = True,
    ):
        super().__init__(
            trainer,
            n_trials,
            hp_optimizer=hp_optimizer,
            continue_loop_on_error=continue_loop_on_error
        )
        self.n_jobs = n_jobs

    def _run(self, pipeline, dact: DACT, context: ExecutionContext):
        # TODO: what is this method used for?

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

        self.pipeline = self.refit_best_trial(self.pipeline, data_container, context)

        return self.pipeline

    def refit_best_trial(self, pipeline: BaseStep, data_container: DACT, context: ExecutionContext) -> BaseStep:
        """
        Refit the pipeline on the whole dataset (without any validation technique).

        :param pipeline: a virgin pipeline to refit.
        :param data_container: data container containing all the data to refit on.
        :param context: execution context containing the flow to operate on and with.
        :return: fitted pipeline
        """
        p: BaseStep = self._load_virgin_model(hyperparams=best_hyperparams)
        p = self.trainer.refit(
            p=p,
            data_container=data_container,
            context=context.set_execution_phase(ExecutionPhase.TRAIN)
        )

        context.flow.log_model(p)

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
        :class:`EasyAutoML`,
        :class:`Trainer`,
        :class:`HyperparamsRepository`,
        :class:`BaseControllerLoop`,
        :class:`BaseValidationSplitter`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`HyperparameterSpace`
        :class:`BaseScoringCallback`,
        :class:`Flow`,
    """

    def __init__(
            self,
            pipeline: BaseStep,
            loop: BaseControllerLoop,  # HyperbandControllerLoop(), ClusteringParallelFor()
            flow: Flow,
            start_new_round: bool = True,  # otherwise, pick last run. TODO: here?
            refit_best_trial: bool = True,
    ):
        """
        Notes on multiprocess :
              Usage of a multiprocess-safe hyperparams repository is recommended, although it is, most of the time, not necessary.
              Beware of the behaviour of HyperparamsRepository's observers/subscribers.
              Context instances are not shared between trial but copied. So is the AutoML loop and the DACTs.

        :param pipeline: The pipeline, or BaseStep, which will be use by the AutoMLloop
        :param loop: The loop, or BaseControllerLoop, which will be used by the AutoML loop
        :param flow: The flow, or Flow, which will be used by the AutoML loop
        :param start_new_round: If True, a new run (round) will be started. Otherwise, the last run will be used.
        :param refit_best_trial: A boolean indicating whether to perform, after a fit call, a refit on the best trial.
        """
        BaseStep.__init__(self)
        _HasChildrenMixin.__init__(self)
        ForceHandleMixin.__init__(self)

        self.pipeline: BaseStep = pipeline
        self.loop: BaseControllerLoop = loop
        self.flow: Flow = flow
        self.start_new_run: bool = start_new_round
        self.refit_best_trial: bool = refit_best_trial

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
        context.copy().register_service(Flow, self.flow)

        self.loop.run(self.pipeline, data_container, context)

        if self.refit_best_trial:
            self.pipeline = self.loop.refit_best_trial(self.pipeline, data_container, context)

        return self

    def _fit_transform_data_container(
        self, data_container: DACT, context: ExecutionContext
    ) -> ('BaseStep', DACT):
        raise NotImplementedError("AutoML does not implement method _fit_transform_data_container. Use method such as "
                                  "fit or handle_fit to train models and then use method such as get_best_model to "
                                  "retrieve the model you wish to use for transform")

    def _transform_data_container(self, data_container: DACT, context: ExecutionContext) -> DACT:
        return self.pipeline.handle_transform(data_container, context)

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
        raise NotImplementedError("TODO")

    def get_children(self) -> List[BaseStep]:
        return [self.get_best_model()]

    @property
    def wrapped(self) -> BaseStep:
        return self.get_children()


def _get_trial_split_description(
        repo_trial: TrialManager,
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
        )
        flow: Flow = AutoMLFlow(repo=hyperparams_repository)
        assert cache_folder_when_no_handle is None  # TODO: remove this.

        AutoML.__init__(
            self,
            pipeline=pipeline,
            loop=controller_loop,
            flow=flow,
            refit_best_trial=refit_best_trial,
            start_new_round=True,
        )
