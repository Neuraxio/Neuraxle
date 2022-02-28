import logging
import os
import shutil
from typing import Set

import numpy as np
import pytest
from neuraxle.base import BaseStep
from neuraxle.base import ExecutionContext as CX
from neuraxle.base import HandleOnlyMixin, Identity, TrialStatus
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.distributions import FixedHyperparameter
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.logging.logging import NEURAXLE_LOGGER_NAME, NeuraxleLogger
from neuraxle.metaopt.auto_ml import AutoML
from neuraxle.metaopt.callbacks import ScoringCallback
from neuraxle.metaopt.data.json_repo import HyperparamsJSONRepository
from neuraxle.metaopt.data.vanilla import (DEFAULT_PROJECT, AutoMLContext,
                                           ProjectDataclass, ScopedLocation)
from neuraxle.metaopt.validation import ValidationSplitter, RandomSearchSampler
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import AddN, MultiplyByN, NumpyReshape
from sklearn.metrics import mean_squared_error


class FitTransformCounterLoggingStep(HandleOnlyMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        HandleOnlyMixin.__init__(self)
        self.logging_call_counter = 0

    def _fit_data_container(self, data_container: DACT, context: CX) -> BaseStep:
        self._log(context, "fit")
        return self

    def _transform_data_container(self, data_container: DACT, context: CX) -> DACT:
        self._log(context, "transform")
        return data_container

    def _fit_transform_data_container(self, data_container: DACT, context: CX) -> DACT:
        self._log(context, "fit_transform")
        return data_container

    def _log(self, context, name):
        context.logger.warning(f"{name} call - logging call # {self.logging_call_counter}")
        self.logging_call_counter += 1


def test_context_logger_log_file(tmpdir):
    # Setup
    file_path = os.path.join(tmpdir, "test.log")
    if os.path.exists(file_path):
        os.remove(file_path)
    try:

        # Given
        cx = CX(tmpdir)
        cx.logger.with_file_handler(file_path)
        pipeline = Pipeline([
            MultiplyByN(2).set_hyperparams_space(HyperparameterSpace({
                'multiply_by': FixedHyperparameter(2)
            })),
            NumpyReshape(new_shape=(-1, 1)),
            FitTransformCounterLoggingStep()
        ])

        # When
        dact = DACT(
            data_inputs=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        )
        pipeline.handle_fit(dact, cx)

        # Then
        log = ''.join(cx.logger.read_log_file())
        assert "fit call - logging call # 0" in log

    # Teardown
    finally:
        cx.logger.without_file_handler()
        os.remove(file_path)


class TestTrialLogger:

    @pytest.mark.skip(reason="TODO: fix the on disk repo and this logging test.")
    def test_logger_automl(self, tmpdir):
        # Given
        context = CX(tmpdir)
        self.tmpdir = str(tmpdir)
        hp_repository = HyperparamsJSONRepository(cache_folder=self.tmpdir)
        n_epochs = 2
        n_trials = 4
        auto_ml = AutoML(
            pipeline=Pipeline([
                MultiplyByN(2).set_hyperparams_space(HyperparameterSpace({
                    'multiply_by': FixedHyperparameter(2)
                })),
                NumpyReshape(new_shape=(-1, 1)),
                FitTransformCounterLoggingStep()
            ]),
            hyperparams_optimizer=RandomSearchSampler(),
            validation_splitter=ValidationSplitter(0.20),
            scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
            n_trials=n_trials,
            refit_best_trial=True,
            epochs=n_epochs,
            hyperparams_repository=hp_repository,
            continue_loop_on_error=False
        )

        # When
        data_container = DACT(
            data_inputs=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            expected_outputs=np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        )
        auto_ml.handle_fit(data_container, context)

        # Then
        file_paths = [os.path.join(hp_repository.cache_folder, f"trial_{i}.log") for i in range(n_trials)]
        assert len(file_paths) == n_trials

        for f in file_paths:
            # TODO: cx.copy().with_sublocation(...).logger.read_log_file()
            assert os.path.exists(f)

        for f in file_paths:
            with open(f, 'r') as f:
                log = f.readlines()
                assert len(log) == 36


def test_automl_context_has_loc():
    cx = AutoMLContext.from_context()

    assert cx.loc is not None


def test_automl_context_pushes_loc_attr():
    cx = AutoMLContext.from_context()

    c1 = cx.push_attr(ProjectDataclass(DEFAULT_PROJECT))

    assert cx.loc == ScopedLocation()
    assert c1.loc == ScopedLocation(DEFAULT_PROJECT)


def test_automl_context_loc_pushes_identifier():
    cx = AutoMLContext.from_context()

    c1 = cx.push_attr(ProjectDataclass(DEFAULT_PROJECT))

    assert cx.get_identifier() == NEURAXLE_LOGGER_NAME
    assert c1.get_identifier() == f"{NEURAXLE_LOGGER_NAME}.{DEFAULT_PROJECT}"


def test_root_neuraxle_logger_logs_to_string():
    nxl: NeuraxleLogger = NeuraxleLogger.from_identifier(CX().get_identifier())

    nxl.info("This is a test.")

    assert "This is a test." in nxl.get_scoped_string_history()


def test_automl_neuraxle_logger_logs_to_repo_file():
    cx: AutoMLContext = AutoMLContext.from_context()

    try:
        cx.add_scoped_logger_file_handler()

        cx.flow.log_status(TrialStatus.RUNNING)
    finally:
        cx.free_scoped_logger_file_handler()
    cx.flow.log_end(TrialStatus.ABORTED)

    assert os.path.exists(cx.repo.get_scoped_logger_path(cx.loc))
    f = cx.read_scoped_logger_file()
    assert str(TrialStatus.RUNNING) in f
    assert str(TrialStatus.ABORTED) not in f


def test_sub_root_neuraxle_loggers_logs_to_string():
    str_r = "Testing root."
    str_a = "Testing a."
    str_b = "Testing b."
    cx = CX()
    nxl_r: NeuraxleLogger = NeuraxleLogger.from_identifier(
        cx.get_identifier())
    nxl_a: NeuraxleLogger = NeuraxleLogger.from_identifier(
        cx.push(Identity(name="a")).get_identifier())
    nxl_b: NeuraxleLogger = NeuraxleLogger.from_identifier(
        cx.push(Identity(name="b")).get_identifier())

    nxl_r.info(str_r)
    nxl_a.info(str_a)
    nxl_b.info(str_b)

    assert str_r in nxl_r.get_scoped_string_history()
    assert str_r not in nxl_a.get_scoped_string_history()
    assert str_r not in nxl_b.get_scoped_string_history()

    assert str_a in nxl_r.get_scoped_string_history()
    assert str_a in nxl_a.get_scoped_string_history()
    assert str_a not in nxl_b.get_scoped_string_history()

    assert str_b in nxl_r.get_scoped_string_history()
    assert str_b not in nxl_a.get_scoped_string_history()
    assert str_b in nxl_b.get_scoped_string_history()


def test_auto_ml_context_services_names():
    cx = AutoMLContext.from_context()

    names: Set[str] = set(cx.getattr("name").to_flat_dict().values())
    assert names == set([
        'AutoMLContext',
        'Flow',
        'VanillaHyperparamsRepository',
        'ScopedLocation',
        'ContextLock'
    ])


def test_automl_context_repo_service_config():
    cx = AutoMLContext.from_context()

    cx.repo.set_config({"some_key": "some_value"})

    assert dict(cx.get_config().to_flat_dict()) == {
        'VanillaHyperparamsRepository__some_key': 'some_value'
    }
    assert cx.has_service("HyperparamsRepository")
    assert cx.get_service("HyperparamsRepository").__class__.__name__ == "VanillaHyperparamsRepository"
