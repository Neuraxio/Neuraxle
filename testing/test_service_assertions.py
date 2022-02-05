import os
import shutil
from abc import ABC, abstractmethod

import numpy as np
import pytest
from neuraxle.base import (BaseService, BaseStep, BaseTransformerT,
                           CX, ForceHandleIdentity, Identity,
                           NonFittableMixin, StepWithContext)
from neuraxle.data_container import DataContainer as DACT
from neuraxle.data_container import DataContainer as DACT
from neuraxle.metaopt.auto_ml import AutoML, RandomSearchSampler
from neuraxle.metaopt.callbacks import ScoringCallback
from neuraxle.metaopt.data.json_repo import HyperparamsJSONRepository
from neuraxle.metaopt.validation import ValidationSplitter
from neuraxle.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


class SomeBaseService(BaseService):
    @abstractmethod
    def service_method(self, data):
        pass


class SomeService(SomeBaseService):
    def service_method(self, data):
        self.data = data


class AnotherService(BaseService):
    def __init__(self):
        super().__init__()

    def _setup(self, context: 'CX' = None) -> BaseTransformerT:
        return BaseService._setup(self, context=context)


class SomeStep(ForceHandleIdentity):
    def _will_process(self, data_container: DACT, context: CX):
        data_container, context = super()._will_process(data_container, context)
        service = context.get_service(SomeBaseService)
        service.service_method(data_container.data_inputs)
        return data_container, context


class RegisterServiceDynamically(ForceHandleIdentity):
    def _will_process(self, data_container: DACT, context: CX):
        context.register_service(SomeBaseService, SomeService())
        return data_container, context


class SomeStepWithFailedAssertion(NonFittableMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)

    # Create a step that will self._assert false and raise an exception and ensure it raised.
    def transform(self, data_inputs):
        for di in data_inputs:
            self._assert(di, f"{di}")
        return None


def test_context_assertions_raises_when_it_throws(tmpdir):
    assertion_was_raised = False

    try:
        SomeStepWithFailedAssertion().transform([True, True, False])
    except AssertionError:
        assertion_was_raised = True

    assert assertion_was_raised


def test_context_assertions_passes_when_it_passes(tmpdir):
    assertion_was_raised = False

    try:
        SomeStepWithFailedAssertion().transform([True, True, True])
    except AssertionError:
        assertion_was_raised = True

    assert not assertion_was_raised


def test_with_context_should_inject_dependencies_properly(tmpdir):
    data_inputs = np.array([0, 1, 2, 3])
    context = CX(root=tmpdir)
    service = SomeService()
    context.set_service_locator({SomeBaseService: service})
    p = Pipeline([
        SomeStep().assert_has_services(SomeBaseService),
        SomeStep().assert_has_services_at_execution(SomeBaseService)
    ]).with_context(context=context)

    p.transform(data_inputs=data_inputs)

    assert np.array_equal(service.data, data_inputs)


def test_with_context_should_fail_at_init_when_services_are_missing(tmpdir):
    context = CX(root=tmpdir)
    p = Pipeline([
        SomeStep().assert_has_services(SomeBaseService)
    ]).with_context(context=context)

    data_inputs = np.array([0, 1, 2, 3])
    with pytest.raises(AssertionError) as exception_info:
        p.transform(data_inputs=data_inputs)

    assert 'Expected context to have service of type SomeBaseService' in exception_info.value.args[0]


def test_localassert_should_assert_dependencies_properly_at_exec(tmpdir):
    data_inputs = np.array([0, 1, 2, 3])
    context = CX(root=tmpdir)
    RegisterServiceDynamically().handle_transform(DACT(data_inputs), context)
    p = Pipeline([Pipeline([
        SomeStep().assert_has_services_at_execution(SomeBaseService)
    ])]).with_context(context=context)

    p.transform(data_inputs)
    service = p.context.get_service(SomeBaseService)
    assert np.array_equal(service.data, data_inputs)


def test_localassert_should_fail_when_services_are_missing_at_exec(tmpdir):
    context = CX(root=tmpdir)
    p = Pipeline([
        SomeStep().assert_has_services_at_execution(SomeBaseService)
    ]).with_context(context=context)
    data_inputs = np.array([0, 1, 2, 3])

    with pytest.raises(AssertionError) as exception_info:
        p.transform(data_inputs=data_inputs)

    assert 'Expected context to have service of type SomeBaseService' in exception_info.value.args[0]


def _make_autoML_loop(tmpdir, p: Pipeline):
    hp_repository = HyperparamsJSONRepository(cache_folder=tmpdir)
    n_epochs = 1
    return AutoML(
        pipeline=p,
        hyperparams_optimizer=RandomSearchSampler(),
        validation_splitter=ValidationSplitter(0.20),
        scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
        n_trials=1,
        refit_best_trial=True,
        epochs=n_epochs,
        hyperparams_repository=hp_repository,
        cache_folder_when_no_handle=str(tmpdir),
        continue_loop_on_error=False
    )


class TestServiceAssertion:

    def _setup(self, tmpdir):
        self.tmpdir = str(tmpdir)
        self.tmpdir_hp = self.tmpdir+"_hp"

    def teardown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        shutil.rmtree(self.tmpdir + "_hp", ignore_errors=True)

    def test_step_with_context_saver_only_saves_wrapped(self, tmpdir):
        self._setup(tmpdir)
        pipeline_name = 'testname'
        context = CX(tmpdir).set_service_locator({SomeBaseService: SomeService()})
        p = Pipeline([
            SomeStep().assert_has_services(SomeBaseService)
        ]).set_name(pipeline_name).with_context(context=context)

        p.save(context, full_dump=True)

        p: Pipeline = CX(tmpdir).load(os.path.join(pipeline_name))
        assert isinstance(p, Pipeline)

    def test_auto_ml_should_inject_dependencies_properly(self, tmpdir):
        self._setup(tmpdir)
        data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        expected_outputs = data_inputs * 2
        p = Pipeline([
            SomeStep().assert_has_services(SomeBaseService),
            SomeStep().assert_has_services_at_execution(SomeBaseService)
        ])
        context = CX(root=self.tmpdir)
        service = SomeService()
        context.set_service_locator({SomeBaseService: service})

        auto_ml: AutoML = _make_autoML_loop(self.tmpdir_hp, p)
        auto_ml: StepWithContext = auto_ml.with_context(context=context)
        assert isinstance(auto_ml, StepWithContext)
        auto_ml.fit(data_inputs, expected_outputs)

        assert np.array_equal(service.data, data_inputs)

    def test_auto_ml_should_fail_at_init_when_services_are_missing(self, tmpdir):
        self._setup(tmpdir)
        data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        expected_outputs = data_inputs * 2
        p = Pipeline([
            RegisterServiceDynamically(),
            SomeStep().assert_has_services(SomeBaseService),
        ])

        context = CX(root=self.tmpdir)

        auto_ml: AutoML = _make_autoML_loop(self.tmpdir_hp, p)
        auto_ml: StepWithContext = auto_ml.with_context(context=context)
        assert isinstance(auto_ml, StepWithContext)

        with pytest.raises(AssertionError) as exception_info:
            auto_ml.fit(data_inputs, expected_outputs)

        assert 'SomeBaseService dependency missing' in exception_info.value.args[0]

    def test_auto_ml_should_fail_at_exec_when_services_are_missing(self, tmpdir):
        self._setup(tmpdir)
        data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        expected_outputs = data_inputs * 2
        p = Pipeline([
            SomeStep().assert_has_services_at_execution(SomeBaseService),
        ])
        context = CX(root=self.tmpdir)

        auto_ml: AutoML = _make_autoML_loop(self.tmpdir_hp, p)
        auto_ml: StepWithContext = auto_ml.with_context(context=context)
        assert isinstance(auto_ml, StepWithContext)

        with pytest.raises(AssertionError) as exception_info:
            auto_ml.fit(data_inputs, expected_outputs)
        assert 'SomeBaseService dependency missing' in exception_info.value.args[0]

    def test_auto_ml_should_assert_dependecies_properly_at_exec(self, tmpdir):
        self._setup(tmpdir)
        data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        expected_outputs = data_inputs * 2
        p = Pipeline([
            RegisterServiceDynamically(),
            SomeStep().assert_has_services_at_execution(SomeBaseService),
        ])
        context = CX(root=self.tmpdir)

        auto_ml: AutoML = _make_autoML_loop(self.tmpdir_hp, p)
        auto_ml: StepWithContext = auto_ml.with_context(context=context)
        assert isinstance(auto_ml, StepWithContext)
        auto_ml.fit(data_inputs, expected_outputs)

        service = context.get_service(SomeBaseService)
        assert np.array_equal(service.data, data_inputs)


def test_context_can_access_services_from_apply_method(tmpdir):
    data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_outputs = data_inputs * 2

    cx = CX(tmpdir, services={
        SomeBaseService: SomeService(),
        AnotherService: AnotherService()
    })
    p = Pipeline([
        Identity().assert_has_services(AnotherService),
        SomeStep().assert_has_services_at_execution(SomeBaseService),
    ]).with_context(context=cx)

    for i in cx.services.values():
        assert isinstance(i, BaseService)
        assert i.get_config().to_flat_dict() == dict()

    p.fit(data_inputs, expected_outputs)

    service = cx.get_service(SomeBaseService)
    assert np.array_equal(service.data, data_inputs)

    for i in cx.services.values():
        assert i.get_config().to_flat_dict() == dict()
