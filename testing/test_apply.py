import numpy as np

from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline
from neuraxle.steps.misc import TapeCallbackFunction
from neuraxle.steps.numpy import MultiplyByN
from neuraxle.steps.output_handlers import OutputTransformerWrapper

DATA_INPUTS = np.array([1])
EXPECTED_OUTPUTS = np.array([1])

tape_transform = TapeCallbackFunction()
tape_fit = TapeCallbackFunction()


def test_apply_on_pipeline_should_call_method_on_each_steps():
    pipeline = Pipeline([MultiplyByN(1), MultiplyByN(1)])

    pipeline.apply('set_hyperparams', HyperparameterSamples({'multiply_by': 2}))

    assert pipeline.get_hyperparams()['multiply_by'] == 2
    assert pipeline['MultiplyByN'].get_hyperparams()['multiply_by'] == 2
    assert pipeline['MultiplyByN1'].get_hyperparams()['multiply_by'] == 2


def test_apply_method_on_pipeline_should_call_method_on_each_steps():
    pipeline = Pipeline([MultiplyByN(1), MultiplyByN(1)])

    pipeline.apply_method(
        lambda step: step.set_hyperparams(HyperparameterSamples({'multiply_by': 2}))
    )

    assert pipeline.get_hyperparams()['multiply_by'] == 2
    assert pipeline['MultiplyByN'].get_hyperparams()['multiply_by'] == 2
    assert pipeline['MultiplyByN1'].get_hyperparams()['multiply_by'] == 2


def test_apply_on_pipeline_with_meta_step_should_call_method_on_each_steps():
    pipeline = Pipeline([OutputTransformerWrapper(MultiplyByN(1)), MultiplyByN(1)])

    pipeline.apply('set_hyperparams', HyperparameterSamples({'multiply_by': 2}))

    assert pipeline.get_hyperparams()['multiply_by'] == 2
    assert pipeline['OutputTransformerWrapper'].wrapped.get_hyperparams()['multiply_by'] == 2
    assert pipeline['MultiplyByN'].get_hyperparams()['multiply_by'] == 2


def test_apply_method_on_pipeline_with_meta_step_should_call_method_on_each_steps():
    pipeline = Pipeline([OutputTransformerWrapper(MultiplyByN(1)), MultiplyByN(1)])

    pipeline.apply_method(
        lambda step: step.set_hyperparams(HyperparameterSamples({'multiply_by': 2}))
    )

    assert pipeline.get_hyperparams()['multiply_by'] == 2
    assert pipeline['OutputTransformerWrapper'].wrapped.get_hyperparams()['multiply_by'] == 2
    assert pipeline['MultiplyByN'].get_hyperparams()['multiply_by'] == 2
