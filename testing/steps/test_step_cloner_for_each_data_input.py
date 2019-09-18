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
