import typing

import numpy as np
import pytest
from neuraxle.base import (BaseService, BaseStep, ExecutionContext, Flow,
                           HandleOnlyMixin, Identity, MetaStep)
from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.distributions import RandInt, Uniform
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace)
from neuraxle.metaopt.auto_ml import (AutoML, AutoMLFlow, DefaultLoop,
                                      InMemoryHyperparamsRepository,
                                      RandomSearch, Trainer,
                                      ValidationSplitter)
from neuraxle.metaopt.callbacks import MetricCallback
from neuraxle.pipeline import Pipeline
from neuraxle.steps.data import DataShuffler
from neuraxle.steps.flow import TrainOnlyWrapper
from neuraxle.steps.numpy import AddN
from neuraxle.steps.sklearn import SKLearnWrapper
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, SGDClassifier, SGDRegressor
from sklearn.metrics import median_absolute_error


class StepThatAssertsContextIsSpecified(HandleOnlyMixin, BaseStep):
    def __init__(self, expected_context: ExecutionContext):
        BaseStep.__init__(self)
        HandleOnlyMixin.__init__(self)
        self.context = expected_context

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> 'BaseTransformer':
        # todo: context.flow.get_logs() == same ? Looks like pytest with logger class.
        self._assert_equals(self.context, context, 'Context is not the expected one.', context)
        return super()._fit_data_container(data_container, context)


def test_automl_context_is_correctly_specified_into_trial_with_full_automl_scenario(tmpdir):
    # This is a large test
    dact = DataContainer(di=list(range(10)), eo=list(range(10, 20)))
    cx = ExecutionContext(root=tmpdir)
    expected_deep_cx = ExecutionContext(root=tmpdir)
    assertion_step = StepThatAssertsContextIsSpecified(expected_context=expected_deep_cx)
    automl = AutoML(
        pipeline=Pipeline([
            TrainOnlyWrapper(DataShuffler()),
            AddN().with_hp_range(range(8, 12)),
            assertion_step
        ]),
        controller_loop=DefaultLoop(
            trainer=Trainer(
                validation_splitter=ValidationSplitter(test_size=0.2),
                n_epochs=1,
                callbacks=[MetricCallback('MAE', median_absolute_error, True)]
            ),
            hyperparams_optimizer=RandomSearch('MAE'),
            n_trials=80,
            n_jobs=10,
        ),
        flow=AutoMLFlow(
            repo=InMemoryHyperparamsRepository(),
            project_id="default_project",
            client_id="default_client",
        ),
        start_new_run=True,
        refit_best_trial=True,
    )
    automl = automl.handle_fit(dact, cx)

    predicted = automl.handle_predict(dact.without_eo(), cx)

    assert np.array_equal(predicted.di, np.array(range(10, 20)))


def test_two_automl_in_parallel_can_contribute_to_the_same_hp_repository():
    # This is a large test
    pass


def test_automl_flow_logs_the_data_of_the_status_and_metrics_and_introspection():
    pass


def test_on_disk_repo_is_structured_accordingly():
    pass


def test_automl_will_use_logger_for_each_trial():
    pass


def test_automl_default_loop_does_the_right_number_of_trials():
    pass


def test_automl_rounds_does_the_right_number_of_trials():
    pass


def test_trial_split_collects_good_metadata():
    pass
