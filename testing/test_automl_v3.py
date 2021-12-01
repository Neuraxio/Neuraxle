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
from neuraxle.metaopt.callbacks import CallbackList, MetricCallback
from neuraxle.metaopt.data.trial import (ClientScope, EpochScope, ProjectScope,
                                         RoundScope, TrialScope,
                                         TrialSplitScope)
from neuraxle.metaopt.data.vanilla import (AutoMLContext, AutoMLFlow,
                                           BaseDataclass,
                                           BaseHyperparameterOptimizer,
                                           ClientDataclass,
                                           MetricResultsDataclass,
                                           ProjectDataclass, RoundDataclass,
                                           ScopedLocation, TrialDataclass,
                                           TrialSplitDataclass,
                                           VanillaHyperparamsRepository,
                                           dataclass_2_attr)
from neuraxle.metaopt.validation import (GridExplorationSampler,
                                         ValidationSplitter)
from neuraxle.pipeline import Pipeline
from neuraxle.steps.data import DataShuffler
from neuraxle.steps.flow import TrainOnlyWrapper
from neuraxle.steps.numpy import AddN, MultiplyByN
from sklearn.metrics import median_absolute_error


class StepThatAssertsContextIsSpecified(HandleOnlyMixin, BaseStep):
    def __init__(self, expected_context: ExecutionContext):
        BaseStep.__init__(self)
        HandleOnlyMixin.__init__(self)
        self.context = expected_context

    def _fit_data_container(self, data_container: DACT, context: ExecutionContext) -> BaseTransformer:
        # todo: context.flow.get_logs() == same ? Looks like pytest with logger class.
        self._assert_equals(self.context, context, 'Context is not the expected one.', context)
        return super()._fit_data_container(data_container, context)


def test_automl_context_is_correctly_specified_into_trial_with_full_automl_scenario(tmpdir):
    # This is a large test
    dact = DACT(di=list(range(10)), eo=list(range(10, 20)))
    cx = ExecutionContext(root=tmpdir)
    expected_deep_cx = ExecutionContext(root=tmpdir)
    assertion_step = StepThatAssertsContextIsSpecified(expected_context=expected_deep_cx)
    automl = AutoML(
        pipeline=Pipeline([
            TrainOnlyWrapper(DataShuffler()),
            AddN().with_hp_range(range(8, 12)),
            assertion_step
        ]),
        loop=DefaultLoop(
            trainer=Trainer(
                validation_splitter=ValidationSplitter(test_size=0.2),
                n_epochs=1,
                callbacks=[MetricCallback('MAE', median_absolute_error, True)]
            ),
            hp_optimizer=RandomSearch(main_metric_name='MAE'),
            start_new_round=True,
            n_trials=80,
            n_jobs=10,
        ),
        repo=VanillaHyperparamsRepository(tmpdir),
        refit_best_trial=True,
    )
    automl = automl.handle_fit(dact, cx)

    predicted = automl.handle_predict(dact.without_eo(), cx)

    assert np.array_equal(predicted.di, np.array(range(10, 20)))


def test_scoped_cascade_does_the_right_logging(tmpdir):
    dact_train_x: DACT = DACT(ids=range(0, 10), di=range(0, 10))
    dact_train_y: DACT = DACT(ids=range(0, 10), di=range(100, 110))
    dact_valid_x: DACT = DACT(ids=range(10, 20), di=range(10, 20))
    dact_valid_y: DACT = DACT(ids=range(10, 20), di=range(110, 120))
    context = AutoMLContext.from_context(ExecutionContext(), VanillaHyperparamsRepository(tmpdir))
    hp_optimizer: BaseHyperparameterOptimizer = GridExplorationSampler(main_metric_name='MAE', expected_n_trials=1)
    n_epochs = 3
    callbacks = CallbackList([MetricCallback('MAE', median_absolute_error, False)])
    expected_scope = ScopedLocation(
        project_name='some_test_project',
        client_name='some_test_client',
        round_number=1,
        trial_number=1,
        split_number=1,
        metric_name='MAE',
    )
    p = Pipeline([
        MultiplyByN().with_hp_range(range(1, 3)),
        AddN().with_hp_range(range(99, 103)),
    ])

    with ProjectScope(context, expected_scope.project_name) as ps:
        ps: ProjectScope = ps
        with ps.new_client(expected_scope.client_name) as cs:
            cs: ClientScope = cs
            with cs.optim_round(p.get_hyperparams_space()) as rs:
                rs: RoundScope = rs
                with rs.new_hyperparametrized_trial(hp_optimizer=hp_optimizer, continue_loop_on_error=False) as ts:
                    ts: TrialScope = ts
                    with ts.new_trial_split() as tss:
                        tss: TrialSplitScope = tss

                        for e in range(n_epochs):
                            with tss.new_epoch() as es:
                                es: EpochScope = es

                                if callbacks.call(
                                    es.context.validation(),
                                    e, n_epochs,
                                    dact_train_x, dact_train_y,
                                    dact_valid_x, dact_valid_y,
                                ):
                                    break

    for dataclass_type in dataclass_2_attr.keys():
        dc: BaseDataclass = context.repo.get(expected_scope[ProjectDataclass])
        assert isinstance(dc, dataclass_2_attr)
        assert dc.get_id() == expected_scope[dataclass_type]

        if dataclass_type == MetricResultsDataclass:
            assert len(dc.get_sublocation()) == n_epochs
        else:
            assert len(dc.get_sublocation()) == 1

    # dc: MetricResultsDataclass = context.repo.get(ScopedLocation())
    assert len(dc.validation_values) == n_epochs


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
