from typing import Any

from neuraxle.hyperparams.distributions import Boolean
from neuraxle.hyperparams.space import HyperparameterSpace, HyperparameterSamples
from neuraxle.steps.util import StepClonerForEachDataInput
from testing.test_pipeline import SomeStep

HYPE_SPACE = HyperparameterSpace({
    "a__test": Boolean()
})

HYPE_SAMPLE = HyperparameterSamples({
    "a__test": True
})


def test_step_cloner_should_set_wrapped_step_hyperparams():
    step_cloner = StepClonerForEachDataInput(SomeStep())

    step_cloner.set_hyperparams(HYPE_SAMPLE)

    assert step_cloner.step.get_hyperparams() == HYPE_SAMPLE


def test_step_cloner_should_set_wrapped_step_hyperparams_space():
    step_cloner = StepClonerForEachDataInput(SomeStep())

    step_cloner.set_hyperparams_space(HYPE_SPACE)

    assert step_cloner.step.get_hyperparams_space() == HYPE_SPACE


class SomeStepInverseTransform(SomeStep):
    def fit_transform(self, data_inputs, expected_outputs=None):
        return self, 'fit_transform'

    def inverse_transform(self, processed_outputs):
        return 'inverse_transform'


def test_step_cloner_should_fit_transform():
    some_step = SomeStepInverseTransform()
    step_cloner = StepClonerForEachDataInput(some_step)

    step_cloner, processed_outputs = step_cloner.fit_transform([0])

    assert isinstance(step_cloner.steps[0], SomeStepInverseTransform)
    assert processed_outputs == ['fit_transform']


def test_step_cloner_should_inverse_transform():
    step_cloner = StepClonerForEachDataInput(SomeStepInverseTransform())

    step_cloner, processed_outputs = step_cloner.fit_transform([0])
    step_cloner = step_cloner.reverse()
    processed_outputs = step_cloner.inverse_transform(processed_outputs)

    assert processed_outputs == ['inverse_transform']
