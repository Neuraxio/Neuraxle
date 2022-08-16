import os
from typing import Callable, Optional

import numpy as np
import pytest
from neuraxle.base import BaseStep
from neuraxle.base import ExecutionContext as CX
from neuraxle.base import Identity, NonFittableMixin
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.distributions import Boolean, Choice, LogUniform, RandInt
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import AutoML, RandomSearchSampler
from neuraxle.metaopt.callbacks import EarlyStoppingCallback, MetricCallback, ScoringCallback
from neuraxle.metaopt.context import AutoMLContext
from neuraxle.metaopt.data.vanilla import ScopedLocation
from neuraxle.metaopt.repositories.repo import HyperparamsRepository, VanillaHyperparamsRepository
from neuraxle.metaopt.validation import ValidationSplitter
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import NumpyRavel
from neuraxle.steps.output_handlers import OutputTransformerWrapper
from neuraxle.steps.sklearn import SKLearnWrapper
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from testing_neuraxle.metaopt.test_automl_repositories import CX_WITH_REPO_CTORS, TmpDir


def _create_data_source():
    data_inputs = np.random.random((25, 50)).astype(np.float32)
    expected_outputs = (np.random.random((25,)) > 0.5).astype(np.int32)
    return data_inputs, expected_outputs


class SetNoneEO(Identity):

    def __init__(self):
        Identity.__init__(self)

    def _will_process(self, dact: DACT, cx: CX):
        dact, cx = Identity._will_process(self, dact, cx)
        dact = dact.with_eo(None)
        return dact, cx


class FailingStep(NonFittableMixin, BaseStep):

    def __init__(self):
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)

    def _will_process(self, dact: DACT, cx: CX):
        raise ValueError("This error should be found in the logs of the test.")
        return dact, cx


def _create_pipeline(has_failing_step=False):
    return Pipeline([
        StandardScaler(),
        OutputTransformerWrapper(NumpyRavel()),
        SKLearnWrapper(
            LogisticRegression(),
            HyperparameterSpace({
                'C': LogUniform(0.01, 10.0),
                'fit_intercept': Boolean(),
                'penalty': Choice(['none', 'l2']),
                'max_iter': RandInt(20, 200)
            })
        ),
        FailingStep() if has_failing_step else Identity(),
        SetNoneEO(),
    ])


@pytest.mark.parametrize('cx_repo_ctor', CX_WITH_REPO_CTORS)
@pytest.mark.parametrize('has_failing_step', [False, True])
def test_automl_api_entry_point(tmpdir, cx_repo_ctor: Callable[[Optional[TmpDir]], AutoMLContext], has_failing_step: bool):
    data_inputs, expected_outputs = _create_data_source()
    dact = DACT(data_inputs=data_inputs, expected_outputs=expected_outputs)
    pipeline = _create_pipeline(has_failing_step=has_failing_step)
    # TODO: # HyperbandControllerLoop(), ClusteringParallelFor() ?

    a: AutoML = AutoML(
        pipeline=pipeline,
        validation_splitter=ValidationSplitter(0.20),
        hyperparams_optimizer=RandomSearchSampler(),
        hyperparams_repository=VanillaHyperparamsRepository(cache_folder=os.path.join(tmpdir, "hp")),
        scoring_callback=ScoringCallback(mean_squared_error),
        callbacks=[
            MetricCallback('accuracy', metric_function=accuracy_score, higher_score_is_better=False),
            EarlyStoppingCallback(max_epochs_without_improvement=3)
        ],
        continue_loop_on_error=True,
        n_trials=4,
        epochs=5,
        refit_best_trial=False,
    )
    cx: CX = cx_repo_ctor()
    repo: HyperparamsRepository = cx.repo

    a = a.handle_fit(dact, cx)

    if has_failing_step:
        assert 'ValueError("This error should be found in the logs of the test.")' in repo.get_log_from_logging_handler(
            cx.logger, ScopedLocation())
    else:
        a, _out = a.to_force_refit_best_trial().handle_fit_transform(dact, cx)
        assert _out is not None
