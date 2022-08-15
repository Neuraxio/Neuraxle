"""
Neuraxle's AutoML Classes
====================================
Classes used to build any Automatic Machine Learning pipelines.
Hyperparameter selection strategies are used to optimize the hyperparameters of given pipelines.

..
    Copyright 2022, Neuraxio Inc.

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

import gc
from copy import copy
from typing import ContextManager, Iterator, List, Optional, Tuple

from neuraxle.base import (CX, BaseService, BaseStep, BaseStepT, ExecutionContext, ExecutionMode, ForceHandleMixin,
                           TruncableService, _HasChildrenMixin)
from neuraxle.data_container import ARG_X_INPUTTED, IDT
from neuraxle.data_container import DataContainer as DACT
from neuraxle.data_container import TrainDACT
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.callbacks import (ARG_Y_EXPECTED, ARG_Y_PREDICTD, BaseCallback, CallbackList, MetricCallback,
                                        ScoringCallback)
from neuraxle.metaopt.context import AutoMLContext
from neuraxle.metaopt.data.aggregates import Client, Project, Root, Round, Trial, TrialSplit
from neuraxle.metaopt.data.reporting import RoundReport
from neuraxle.metaopt.data.vanilla import DEFAULT_CLIENT, DEFAULT_PROJECT, RoundDataclass, ScopedLocation
from neuraxle.metaopt.optimizer import BaseHyperparameterOptimizer, GridExplorationSampler, RandomSearchSampler
from neuraxle.metaopt.repositories.repo import HyperparamsRepository
from neuraxle.metaopt.validation import BaseValidationSplitter, ValidationSplitter


class Trainer(BaseService):
    """
    Class used to train a pipeline using various data splits and callbacks for evaluation purposes.
    It loops on splits that the splitter yields, and on epochs as well, to train and validate the pipeline
    with the given metrics and other callbacks.

    If the predicted expected output of a pipeline's prediction dact is not empty, then it will be used
    in the metrics instead of using the fed dacts' expected output. This is to allow for the use of
    autoregressive models, where the expected output is not known at the time of sending it to the model at train-time, but is known for the least at validation time as per the validation splitter.
    """
    # TODO: add this `with_val_set` method that would change splitter to
    #       PresetValidationSetSplitter(self, val) and override..?

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
        return_trained_pipelines: bool = False
    ) -> Optional[List[BaseStep]]:
        """
        Train pipeline using the validation splitter.
        Track training, and validation metrics for each epoch.
        Note: the present method is just a shortcut to using the `execute_trial` method with less boilerplate code needed. Refer to `execute_trial` for full flexibility
        """
        trained_pipelines: List[BaseStep] = []

        splits: List[Tuple[DACT, DACT]] = self.validation_splitter.split_dact(
            dact, context=trial_scope.context)

        for train_dact, val_dact in splits:
            with trial_scope.new_validation_split() as trial_split_scope:
                p = self.train_split(pipeline, train_dact, val_dact, trial_split_scope)
                if return_trained_pipelines:
                    trained_pipelines.append(p)
                else:
                    del p
                    gc.collect()

        return trained_pipelines

    def train_split(
        self,
        pipeline: BaseStep,
        train_dact: DACT,
        val_dact: Optional[DACT],
        trial_split_scope: TrialSplit
    ) -> BaseStep:
        """
        Train a pipeline split. You probably want to use `self.train` instead, to use the validation splitter.
        If validation DACT is None, the evaluation metrics will not save validation results.

        It is to be noted that here, if the data container, after a prediction at train and validation time,
        has an empty expected output (of .expected_outputs of length 0 or that is None), then the
        trainer will pick the expected output of the pre-predicted data container that was fed as an input.
        """
        trial_split_scope: TrialSplit = trial_split_scope.with_n_epochs(self.n_epochs)
        p: BaseStep = pipeline._copy(trial_split_scope.context, deep=True)
        p.set_hyperparams(trial_split_scope.get_hyperparams())
        context: AutoMLContext = trial_split_scope.context

        for _ in range(self.n_epochs):
            e = trial_split_scope.next_epoch()

            # Fit train
            p = p.set_train(True)
            context = context.train()
            p = p.handle_fit(
                train_dact.copy(),
                context
            )

            # Predict train & val
            p = p.set_train(False)
            context = context.validation()
            eval_dact_train = p.handle_predict(
                train_dact.without_eo(),
                context
            )
            eval_dact_train: DACT[IDT, ARG_Y_PREDICTD, ARG_Y_EXPECTED] = eval_dact_train
            _has_empty_eo = eval_dact_train.expected_outputs is None or (hasattr(
                eval_dact_train.expected_outputs, "__len__") and len(eval_dact_train.expected_outputs) == 0)
            if _has_empty_eo or self.validation_splitter.force_fixed_metric_expected_outputs is True:
                eval_dact_train = eval_dact_train.with_eo(train_dact.expected_outputs)

            if val_dact is not None:
                context = context.validation()
                eval_dact_valid = p.handle_predict(
                    val_dact.without_eo(),
                    context
                )
                eval_dact_valid: DACT[IDT, ARG_Y_PREDICTD, ARG_Y_EXPECTED] = eval_dact_valid
                _has_empty_eo = eval_dact_valid.expected_outputs is None or (hasattr(
                    eval_dact_valid.expected_outputs, "__len__") and len(eval_dact_valid.expected_outputs) == 0)
                if _has_empty_eo or self.validation_splitter.force_fixed_metric_expected_outputs is True:
                    eval_dact_valid = eval_dact_valid.with_eo(val_dact.expected_outputs)
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

        return p

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

        with trial_scope.retrain_split() as trial_split_scope:
            trial_split_scope: TrialSplit = trial_split_scope

            return self.train_split(
                pipeline,
                dact,
                None,
                trial_split_scope
            )


class BaseControllerLoop(TruncableService):

    def __init__(
        self,
        trainer: Trainer,
        n_trials: int,
        hp_optimizer: BaseHyperparameterOptimizer = None,
        continue_loop_on_error: bool = True
    ):
        if hp_optimizer is None:
            hp_optimizer = RandomSearchSampler()

        TruncableService.__init__(self, {
            Trainer: trainer,
            BaseHyperparameterOptimizer: hp_optimizer,
        })
        self.n_trials = n_trials
        self.continue_loop_on_error: bool = continue_loop_on_error

    @property
    def trainer(self) -> Trainer:
        return self.get_service(Trainer)

    def run(self, pipeline: BaseStep, dact: DACT, round_scope: Round):
        """
        Run the controller loop.

        :param context: execution context
        :return: the ID of the round that was executed (either created or continued from previous optimization).
        """

        # thread_safe_lock, ..., context = context.thread_safe() ??

        hp_optimizer: BaseHyperparameterOptimizer = self[BaseHyperparameterOptimizer]
        hp_space: HyperparameterSpace = pipeline.get_hyperparams_space()

        round_scope: Round = round_scope.with_optimizer(hp_optimizer, hp_space)

        for managed_trial_scope in self.loop(round_scope):
            managed_trial_scope: Trial = managed_trial_scope  # typing helps
            # trial_scope.context.restore_lock(thread_safe_lock) ??
            self.trainer.train(pipeline, dact, managed_trial_scope)

    def loop(self, round_scope: Round) -> Iterator[Trial]:
        """
        Loop over all trials.

        :param dact: data container that is not yet splitted
        :param context: execution context
        :return:
        """
        for _ in range(self.n_trials):
            with self.next_trial(round_scope) as managed_trial_scope:
                managed_trial_scope: Trial = managed_trial_scope  # typing helps

                yield managed_trial_scope

    def next_trial(self, round_scope: Round) -> ContextManager[Trial]:
        """
        Get the next trial to be executed.

        :param round_scope: round scope
        :return: the next trial to be executed.
        """
        return round_scope.new_rvs_trial(self.continue_loop_on_error)  # TODO: parallelization with this method?

    def refit_best_trial(self, pipeline: BaseStep, dact: DACT, round_scope: Round) -> BaseStep:
        """
        Refit the pipeline on the whole dataset (without any validation technique).
        """
        with round_scope.refitting_best_trial() as managed_trial_scope:

            refitted: BaseStep = self.trainer.refit(
                pipeline, dact, managed_trial_scope)

        return refitted

    def for_refit_only(self) -> 'BaseControllerLoop':
        """
        Create a controller loop configured with zero iterations
        so as to only make the "refit_best_trial" possible.
        """
        self_copy = copy(self)
        self_copy.n_trials = 0
        return self_copy


class DefaultLoop(BaseControllerLoop):

    def __init__(
        self,
        trainer: Trainer,
        n_trials: int,
        hp_optimizer: BaseHyperparameterOptimizer = None,
        continue_loop_on_error: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__(
            trainer,
            n_trials,
            hp_optimizer=hp_optimizer,
            continue_loop_on_error=continue_loop_on_error
        )
        self.n_jobs = n_jobs


class ControlledAutoML(ForceHandleMixin, _HasChildrenMixin[BaseStepT], BaseStep):
    """
    A step to execute Automated Machine Learning (AutoML) algorithms. This step will
    automatically split the data into train and validation splits, and execute an
    hyperparameter optimization on the splits to find the best hyperparameters.

    The :class:`BaseControllerLoop` is useful to possibly split the execution into multiple
    threads, or even multiple machines to decide how to execute the loop.

    The :class:`Trainer` is responsible for training the pipeline on the train and validation
    splits, as per the data split provided by the splitter, and the predicted data containers.

    It is to be noted that if the data container, after a prediction at train and validation time,
    has an empty expected output (of .expected_outputs of length 0 or that is None), then the
    trainer will pick the expected output of the pre-predicted data container that was fed as an input.

    The step with the chosen best hyperparameters will be optionnally refitted to the full
    unsplitted data (pre-split data) if desired, and will be useable using :func:`refit_best_trial`.
    """

    def __init__(
            self,
            pipeline: BaseStepT,
            loop: BaseControllerLoop,
            main_metric_name: str,
            repo: HyperparamsRepository = None,
            start_new_round: bool = True,
            refit_best_trial: bool = True,
            project_name: str = DEFAULT_PROJECT,
            client_name: str = DEFAULT_CLIENT,
    ):
        """
        .. note::
            Usage of a multiprocess-safe hyperparams repository is recommended,
            although it is, most of the time, not necessary.
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

        self.pipeline: BaseStepT = pipeline
        self.loop: BaseControllerLoop = loop
        self.repo: HyperparamsRepository = repo

        self.main_metric_name: str = main_metric_name
        self.start_new_round: bool = start_new_round
        self.refit_best_trial: bool = refit_best_trial
        self.project_name: str = project_name
        self.client_name: str = client_name
        self._round_number: Optional[int] = None

        self.has_model_been_retrained: bool = False

    def get_children(self) -> List[BaseStep]:
        return [self.pipeline]

    @property
    def wrapped(self) -> BaseStep:
        return self.pipeline

    def _fit_transform_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple['BaseStep', DACT]:
        if not self.refit_best_trial:
            raise ValueError(
                "self.refit_best_trial must be True in this AutoML class to do the transform in 'fit_transform'.")

        self = self._fit_data_container(data_container, context)
        data_container = self._transform_data_container(data_container, context)
        return self, data_container

    def to_force_refit_best_trial(self) -> 'ControlledAutoML':
        self_copy = copy(self)
        self_copy.refit_best_trial = True
        self_copy.has_model_been_retrained = False
        self_copy.start_new_round = False
        self_copy.loop = self_copy.loop.for_refit_only()
        return self_copy

    def _fit_data_container(self, data_container: DACT, context: CX) -> 'BaseStep':
        """
        Run Auto ML Loop.
        Find the best hyperparams using the hyperparameter optmizer.
        Evaluate the pipeline on each trial using a validation technique.

        :param data_container: data container to fit
        :param context: execution context

        :return: self
        """
        automl_context: AutoMLContext = self.get_automl_context(context, with_loc=False)
        root: Root = Root.from_context(automl_context, is_deep=False)

        with root.get_project(self.project_name) as ps:
            ps: Project = ps
            with ps.get_client(self.client_name) as cs:
                cs: Client = cs
                with cs.optim_round(self.start_new_round, self.main_metric_name) as rs:
                    rs: Round = rs
                    self._round_number = rs.round_number

                    self.loop.run(self.pipeline, data_container, rs)

                    if self.refit_best_trial:
                        self.pipeline = self.loop.refit_best_trial(self.pipeline, data_container, rs)
                        self.has_model_been_retrained = True

        return self

    def get_automl_context(self, context: ExecutionContext, with_loc=True) -> AutoMLContext:
        cx = AutoMLContext.from_context(context, repo=self.repo)
        if self.repo is None:
            self.repo = cx.repo
        if with_loc and (self._round_number is not None or not self.start_new_round):
            loc = ScopedLocation(self.project_name, self.client_name, self.round_number)
            cx = cx.with_loc(loc)
        return cx

    def _encapsulate_data(
        self, data_inputs: ARG_X_INPUTTED, expected_outputs: ARG_Y_EXPECTED, execution_mode: ExecutionMode
    ) -> Tuple[CX, TrainDACT]:
        """
        This method is overriden from :class:`ForceHandleMixin` to encapsulate
        the repository in a AutoMLContext instead of in a regular CX in case the
        handler methods were not called but the repository was passed at construction.
        """
        data_container = TrainDACT(data_inputs=data_inputs, expected_outputs=expected_outputs)
        context = CX(execution_mode=execution_mode)
        context = self.get_automl_context(context)
        return context, data_container

    def get_best_model(self) -> BaseStep:
        """
        Get the best model if it has been refit, otherwise raises an assertion error.
        """
        self._assert(
            self.has_model_been_retrained,
            "The model has not been retrained, so it cannot be used to get the best model."
        )
        return self.pipeline

    @property
    def round_number(self) -> Optional[int]:
        if self._round_number is None:
            if self.start_new_round:
                raise ValueError("AutoML loop has not been run yet, cannot ask for report.")
            else:
                return len(self.repo.load(ScopedLocation(self.project_name, self.client_name))) - 1
        return self._round_number

    @property
    def report(self) -> RoundReport:
        dc: RoundDataclass = self.repo.load(ScopedLocation(
            self.project_name, self.client_name, self.round_number), deep=True)
        return RoundReport(dc)

    def _transform_data_container(self, data_container: DACT, context: CX) -> DACT:
        if not self.has_model_been_retrained:
            raise ValueError(
                'self.refit_best_trial must be True in AutoML class '
                'to transform, and the AutoMl should have been fitted.')

        return self.pipeline.handle_transform(data_container, context)


class AutoML(ControlledAutoML):
    """
    This class provides a nice interface to easily use the
    ControlledAutoML class and the metaopt module in general.

    It is a wrapper around the :class:`ControlledAutoML` class, which is a wrapper around
    the AutoML loop contained in the :class:`BaseControllerLoop` class
    and the :class:`Trainer` class all contained in this module.

    :param pipeline: pipeline to copy and use for training
    :param validation_splitter: validation splitter to use
    :param refit_best_trial: whether to refit the best model on the whole dataset after the optimization
    :param scoring_callback: main callback to use for scoring, that is deprecated
    :param hyperparams_optimizer: hyperparams optimizer to use
    :param hyperparams_repository: hyperparams repository to use
    :param n_trials: number of trials to run
    :param epochs: number of epochs to train the model for each val split
    :param callbacks: callbacks to use for training - there can be aditionnal metrics there
    :param n_jobs: number of jobs to use for parallelization, defaults is None for no parallelization
    :param continue_loop_on_error: whether to continue the main optimization loop on error or not
    :return: AutoML object ready to use with fit and transform.
    """

    def __init__(
        self,
        pipeline: BaseStep,
        validation_splitter: 'BaseValidationSplitter' = None,
        hyperparams_optimizer: BaseHyperparameterOptimizer = None,
        scoring_callback: ScoringCallback = None,
        callbacks: List[BaseCallback] = None,
        hyperparams_repository: HyperparamsRepository = None,
        n_trials: int = None,
        refit_best_trial: bool = True,
        start_new_round=True,
        epochs: int = 1,
        n_jobs=1,
        continue_loop_on_error=True
    ):
        # parse or guess args:
        validation_splitter = validation_splitter or ValidationSplitter(0.2)
        hyperparams_optimizer = hyperparams_optimizer or GridExplorationSampler()

        callbacks = list(callbacks) if callbacks is not None else []
        if scoring_callback is not None:
            callbacks = [scoring_callback] + callbacks
        if len(callbacks) == 0:
            raise ValueError("At least one callback must be provided.")
        if not isinstance(callbacks[0], MetricCallback):
            raise ValueError("The first callback is the scoring callback and it must be a MetricCallback.")

        if n_trials is None or n_trials < 1:
            n_trials: int = GridExplorationSampler.estimate_ideal_n_trials(pipeline.get_hyperparams_space())

        # init subservices:
        trainer = Trainer(
            callbacks=callbacks,
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

        # init base class:
        ControlledAutoML.__init__(
            self,
            pipeline=pipeline,
            loop=controller_loop,
            repo=hyperparams_repository,
            main_metric_name=callbacks[0].name,
            start_new_round=start_new_round,
            refit_best_trial=refit_best_trial,
        )
