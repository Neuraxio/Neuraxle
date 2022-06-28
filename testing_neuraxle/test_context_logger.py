import logging
import logging.config
import logging.handlers
import os
from multiprocessing import Process, Queue
from typing import List, Set

import numpy as np
from neuraxle.base import ExecutionContext as CX
from neuraxle.base import Identity, TrialStatus
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.distributions import FixedHyperparameter
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.logging.logging import (NEURAXLE_LOGGER_NAME, NeuraxleLogger,
                                      ParallelLoggingConsumerThread)
from neuraxle.metaopt.auto_ml import AutoML
from neuraxle.metaopt.callbacks import ScoringCallback
from neuraxle.metaopt.data.json_repo import HyperparamsOnDiskRepository
from neuraxle.metaopt.data.vanilla import (DEFAULT_CLIENT, DEFAULT_PROJECT,
                                           AutoMLContext, ProjectDataclass,
                                           ScopedLocation, TrialSplitDataclass)
from neuraxle.metaopt.validation import RandomSearchSampler, ValidationSplitter
from neuraxle.pipeline import Pipeline
from neuraxle.steps.misc import FitTransformCounterLoggingStep
from neuraxle.steps.numpy import MultiplyByN, NumpyReshape
from sklearn.metrics import mean_squared_error


def test_root_neuraxle_logger_has_name_and_identifier():
    cx = CX()
    some_message = "some message."

    nxl: NeuraxleLogger = cx.logger
    nxl.info(some_message)

    assert some_message in nxl.get_root_string_history()
    assert nxl.name == NEURAXLE_LOGGER_NAME


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
        assert "fit call - logging call #0" in log

    # Teardown
    finally:
        cx.logger.without_file_handler()
        os.remove(file_path)


class TestTrialLogger:

    def test_logger_automl(self, tmpdir):
        # Given
        context = CX(tmpdir)
        self.tmpdir = str(tmpdir)
        hp_repository = HyperparamsOnDiskRepository(cache_folder=self.tmpdir)
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
            scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False, name='MSE'),
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
        file_paths = [os.path.join(
            hp_repository.cache_folder, f"_{DEFAULT_PROJECT}", f"_{DEFAULT_CLIENT}", "_0", f"_{i}", "_0", "log.txt"
        ) for i in range(n_trials)]
        assert len(file_paths) == n_trials

        for f in file_paths:
            assert os.path.exists(f)

        for f in file_paths:
            with open(f, 'r') as f:
                log = f.readlines()
                assert len(log) == 18, log


def test_automl_context_has_loc():

    cx = AutoMLContext.from_context()

    assert cx.loc == ScopedLocation()


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


def test_automl_neuraxle_logger_logs_to_repo_file(tmpdir):
    cx: AutoMLContext = AutoMLContext.from_context(repo=HyperparamsOnDiskRepository(cache_folder=tmpdir))

    cx.flow.log_status(TrialStatus.RUNNING)
    cx.flow.log_end(TrialStatus.ABORTED)

    log_file_path_at_loc = cx.repo.get_scoped_logger_path(cx.loc)
    assert os.path.exists(log_file_path_at_loc)
    log1 = cx.read_scoped_log()
    with open(log_file_path_at_loc, 'r') as _file:
        log2 = _file.read()
    assert log1 == log2
    assert str(TrialStatus.RUNNING) in log1
    assert str(TrialStatus.ABORTED) in log1


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


def test_logger_logs_error_stack_trace():
    cx = CX()
    expected_error_stack_trace = (
        "raise ValueError('This is an error.')  # THIS COMMENT IS ALSO LOGGED"
    )

    try:
        raise ValueError('This is an error.')  # THIS COMMENT IS ALSO LOGGED
    except ValueError as e:
        cx.flow.log_error(e)

    assert expected_error_stack_trace in cx.logger.get_root_string_history()


def test_scoped_logger_can_shorten_log_messages():
    trial_scope = ScopedLocation.default_full()[:TrialSplitDataclass]
    cx = AutoMLContext.from_context(loc=trial_scope)

    cx.flow.log_status(TrialStatus.RUNNING)
    cx.flow.log_end(TrialStatus.ABORTED)
    short_logs_l: List[str] = cx.logger.get_short_scoped_logs()
    short_logs = "\n".join(short_logs_l)
    long_logs = cx.logger.get_scoped_string_history()

    assert str(TrialStatus.RUNNING.value) in long_logs
    assert str(TrialStatus.ABORTED.value) in long_logs
    assert "[" in long_logs
    assert "]" in long_logs
    assert "INFO" in long_logs
    assert str(TrialStatus.RUNNING.value) in short_logs
    assert str(TrialStatus.ABORTED.value) in short_logs
    assert "[" not in short_logs
    assert "]" not in short_logs
    assert "INFO" not in short_logs


class SomeParallelLogginWorkers:
    FIRST_LOG_MESSAGE = 'some message logged by worker process'
    SECOND_LOG_MESSAGE = 'Producer - fit_transform call - logging call #0'

    def __init__(self, logging_queue: Queue, n_process: int):
        self.logging_queue: Queue = logging_queue
        self.n_process: int = n_process
        self.workers: List[Process] = []

    def start(self):
        for i in range(self.n_process):
            proc = Process(
                target=self.logger_producer_thread,
                name=f"worker_{i}",
                args=(self.logging_queue,)
            )
            self.workers.append(proc)
            proc.start()

    @staticmethod
    def logger_producer_thread(logging_queue: Queue):
        queue_handler = logging.handlers.QueueHandler(logging_queue)
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        root.addHandler(queue_handler)

        logger = CX().logger
        logger.log(logging.ERROR, SomeParallelLogginWorkers.FIRST_LOG_MESSAGE)

        dact = DACT(di=range(10))
        step, out = FitTransformCounterLoggingStep().set_name("Producer").handle_fit_transform(dact, CX())
        return

    def join(self):
        for worker in self.workers:
            worker.join()


def test_neuraxle_logger_can_operate_in_parallel():
    # TODO: test with disk files as well?
    logging_queue = Queue()
    n_process = 5
    logger_thread = ParallelLoggingConsumerThread(logging_queue)
    logger_thread.start()
    workers = SomeParallelLogginWorkers(logging_queue, n_process)
    workers.start()

    pass  # main thread could be used here as well to be useful for other things.

    workers.join()
    logger_thread.join()
    assert '[neuraxle.Producer]' in CX().logger.get_root_string_history()
    parallel_process_start_counter = 0
    parallel_transform_counter = 0
    for logged_line in CX().logger:
        parallel_process_start_counter += int(SomeParallelLogginWorkers.FIRST_LOG_MESSAGE in logged_line)
        parallel_transform_counter += int(SomeParallelLogginWorkers.SECOND_LOG_MESSAGE in logged_line)
    assert parallel_process_start_counter == n_process
    assert parallel_transform_counter == n_process
