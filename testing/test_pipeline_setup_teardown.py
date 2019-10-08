from typing import Any

import pytest

from neuraxle.base import NamedTupleList, BaseStep
from neuraxle.pipeline import Pipeline
from testing.test_pipeline import SomeStep


class SomePipeline(Pipeline):
    def __init__(self, steps: NamedTupleList):
        super().__init__(steps)
        self.teared_down = False

    def teardown(self) -> 'BaseStep':
        self.teared_down = True
        return self


class SomeException(BaseStep):
    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BaseStep', Any):
        raise Exception()

    def fit(self, data_inputs, expected_outputs=None) -> ('BaseStep', Any):
        raise Exception()

    def transform(self, data_inputs, expected_outputs=None) -> ('BaseStep', Any):
        raise Exception()


class SomeStepSetup(SomeStep):
    def __init__(self):
        super().__init__()
        self.called_with = None

    def setup(self, step_path: str, setup_arguments: dict) -> 'BaseStep':
        self.called_with = [step_path, setup_arguments]


def test_fit_transform_should_setup_pipeline_and_steps():
    step_setup = SomeStepSetup()
    p = SomePipeline([
        step_setup
    ])

    p.fit_transform([1], [1])

    assert p.is_initialized
    assert step_setup.called_with == ['SomePipeline/SomeStepSetup', {}]


def test_transform_should_setup_pipeline_and_steps():
    step_setup = SomeStepSetup()
    p = SomePipeline([
        step_setup
    ])

    p.transform([1])

    assert p.is_initialized
    assert step_setup.called_with == ['SomePipeline/SomeStepSetup', {}]


def test_fit_should_setup_pipeline_and_steps():
    step_setup = SomeStepSetup()
    p = SomePipeline([
        step_setup
    ])

    p.fit([1], [1])

    assert p.is_initialized
