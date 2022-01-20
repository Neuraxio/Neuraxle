import numpy as np
import pytest
from neuraxle.base import (BaseService, BaseStep, BaseTransformer,
                           ExecutionContext, Flow, HandleOnlyMixin, Identity,
                           MetaStep)
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.distributions import RandInt, Uniform
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace)
from neuraxle.metaopt.auto_ml import AutoML, DefaultLoop, RandomSearch, Trainer
from neuraxle.metaopt.callbacks import (CallbackList, EarlyStoppingCallback,
                                        MetricCallback)
from neuraxle.metaopt.data.aggregates import (Client, Project, Root, Round,
                                              Trial, TrialSplit)
from neuraxle.metaopt.data.vanilla import (DEFAULT_CLIENT, DEFAULT_PROJECT,
                                           AutoMLContext, BaseDataclass,
                                           BaseHyperparameterOptimizer,
                                           ClientDataclass,
                                           MetricResultsDataclass,
                                           ProjectDataclass, RootDataclass,
                                           RoundDataclass, ScopedLocation,
                                           TrialDataclass, TrialSplitDataclass,
                                           VanillaHyperparamsRepository,
                                           dataclass_2_id_attr)
from neuraxle.metaopt.validation import (GridExplorationSampler,
                                         ValidationSplitter)
from neuraxle.pipeline import Pipeline
from neuraxle.steps.data import DataShuffler
from neuraxle.steps.flow import TrainOnlyWrapper
from neuraxle.steps.misc import AssertFalseStep
from neuraxle.steps.numpy import AddN, MultiplyByN
from sklearn.metrics import median_absolute_error


class StepThatAssertsContextIsSpecified(Identity):
    def __init__(self, expected_loc: ScopedLocation):
        BaseStep.__init__(self)
        HandleOnlyMixin.__init__(self)
        self.expected_loc = expected_loc

    def _did_process(self, data_container: DACT, context: ExecutionContext) -> DACT:
        context: AutoMLContext = context  # typing annotation for IDE
        self._assert_equals(
            self.expected_loc, context.loc,
            f'Context is not at the expected location. '
            f'Expected {self.expected_loc}, got {context.loc}.',
            context)
        self._assert_equals(
            context.loc in context.repo.root, True,
            "Context should have the dataclass, but it doesn't", context)
        return data_container


def test_automl_context_is_correctly_specified_into_trial_with_full_automl_scenario(tmpdir):
    # This is a large test
    dact = DACT(di=list(range(10)), eo=list(range(10, 20)))
    cx = ExecutionContext(root=tmpdir)
    expected_deep_cx_loc = ScopedLocation.default(0, 0, 0)
    assertion_step = StepThatAssertsContextIsSpecified(expected_loc=expected_deep_cx_loc)
    automl = _create_automl_test_loop(tmpdir, assertion_step)
    automl = automl.handle_fit(dact, cx)

    predicted = automl.handle_predict(dact.without_eo(), cx)

    assert np.array_equal(predicted.di, np.array(range(10, 20)))


def test_automl_step_can_interrupt_on_fail_with_full_automl_scenario(tmpdir):
    # This is a large test
    dact = DACT(di=list(range(10)), eo=list(range(10, 20)))
    cx = ExecutionContext(root=tmpdir)
    assertion_step = AssertFalseStep()
    automl = _create_automl_test_loop(tmpdir, assertion_step)

    with pytest.raises(AssertionError):
        automl.handle_fit(dact, cx)


def _create_automl_test_loop(tmpdir, assertion_step: BaseStep):
    automl = AutoML(
        pipeline=Pipeline([
            TrainOnlyWrapper(DataShuffler()),
            AddN().with_hp_range(range(8, 12)),
            assertion_step
        ]),
        loop=DefaultLoop(
            trainer=Trainer(
                validation_splitter=ValidationSplitter(test_size=0.2),
                n_epochs=4,
                callbacks=[MetricCallback('MAE', median_absolute_error, True)],
            ),
            main_metric_name='MAE',
            hp_optimizer=RandomSearch(),
            n_trials=5,
            start_new_round=True,
            continue_loop_on_error=False,
            n_jobs=2,
        ),
        repo=VanillaHyperparamsRepository(tmpdir),
        refit_best_trial=True,
    )

    return automl


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
