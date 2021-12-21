from typing import Any, Tuple

import pytest
from neuraxle.base import (BaseService, BaseStep, ExecutionContext, Identity,
                           MetaStep, NamedStepsList, _HasChildrenMixin)
from neuraxle.hyperparams.space import RecursiveDict
from neuraxle.pipeline import Pipeline

from testing.test_pipeline import SomeStep


class SomePipeline(Pipeline):
    def __init__(self, steps: NamedStepsList):
        Pipeline.__init__(self, steps)
        self.teared_down = False

    def teardown(self) -> 'BaseStep':
        self.teared_down = True
        return Pipeline.teardown(self)


class SomeException(BaseStep):
    def fit_transform(self, data_inputs, expected_outputs=None) -> Tuple['BaseStep', Any]:
        raise Exception()

    def fit(self, data_inputs, expected_outputs=None) -> Tuple['BaseStep', Any]:
        raise Exception()

    def transform(self, data_inputs, expected_outputs=None) -> Tuple['BaseStep', Any]:
        raise Exception()


class SomeStepSetup(SomeStep):
    def __init__(self):
        SomeStep.__init__(self)
        self.called_with = None


def test_fit_transform_should_setup_pipeline_and_steps():
    step_setup = SomeStepSetup()
    p = SomePipeline([
        step_setup
    ])

    assert not p.is_initialized
    assert not step_setup.is_initialized

    p.fit_transform([1], [1])

    assert p.is_initialized
    assert step_setup.is_initialized


def test_transform_should_setup_pipeline_and_steps():
    step_setup = SomeStepSetup()
    p = SomePipeline([
        step_setup
    ])
    assert not p.is_initialized
    assert not step_setup.is_initialized

    p.transform([1])

    assert p.is_initialized
    assert step_setup.is_initialized


def test_fit_should_setup_pipeline_and_steps():
    step_setup = SomeStepSetup()
    p = SomePipeline([
        step_setup
    ])
    assert not p.is_initialized
    assert not step_setup.is_initialized

    p.fit([1], [1])

    assert p.is_initialized
    assert step_setup.is_initialized


class SomeService(BaseService):
    pass


@pytest.mark.parametrize('base_service', [
    Identity(),
    MetaStep(Identity()),
    SomePipeline([SomeStepSetup()])
])
def test_that_steps_are_setuppeable(base_service: BaseService, tmpdir):
    assert not base_service.is_initialized
    _verify_subservices_initialization(base_service, False)
    base_service.setup(ExecutionContext(tmpdir))
    _verify_subservices_initialization(base_service, True)
    base_service.teardown()
    _verify_subservices_initialization(base_service, False)


def _verify_subservices_initialization(sub_service, is_initialized: bool):
    assert sub_service.is_initialized == is_initialized
    if isinstance(sub_service, _HasChildrenMixin):
        for child in sub_service.get_children():
            _verify_subservices_initialization(child, is_initialized)


@pytest.mark.parametrize('base_service', [
    Identity(),
    MetaStep(Identity()),
    SomePipeline([SomeStepSetup()]),
    ExecutionContext(),
    ExecutionContext().set_service_locator({
        Identity: Identity(),
        SomeService: SomeService()
    }),
    ExecutionContext().set_service_locator({
        Pipeline: Pipeline([SomeStepSetup()])
    })
])
def test_that_steps_are_applyable_with_name(base_service: BaseService, tmpdir):

    names = base_service.apply(lambda self: RecursiveDict({"name": self.get_name()}))

    _verify_subservices_names(base_service, names)


def _verify_subservices_names(sub_service, sub_service_name: RecursiveDict):
    assert sub_service.name == sub_service_name["name"], f"Not equal: {sub_service.name} != {sub_service_name['name']}."
    if isinstance(sub_service, _HasChildrenMixin):
        for child in sub_service.get_children():
            _verify_subservices_names(child, sub_service_name[child.name])
