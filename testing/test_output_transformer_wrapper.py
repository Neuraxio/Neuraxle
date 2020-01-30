import numpy as np

from neuraxle.base import ExecutionContext
from neuraxle.data_container import DataContainer
from neuraxle.pipeline import Pipeline
from neuraxle.steps.misc import FitCallbackStep, TapeCallbackFunction
from neuraxle.steps.numpy import MultiplyByN
from neuraxle.steps.output_handlers import OutputTransformerWrapper


def test_output_transformer_wrapper_should_fit_with_data_inputs_and_expected_outputs_as_data_inputs():
    tape = TapeCallbackFunction()
    p = Pipeline([
        OutputTransformerWrapper(FitCallbackStep(tape))
    ])
    data_inputs, expected_outputs = _create_data_source((10, 10))

    p.fit(data_inputs, expected_outputs)

    assert np.array_equal(tape.data[0][0], expected_outputs)
    for i in range(10):
        assert tape.data[0][1][i] is None


def test_output_transformer_wrapper_should_fit_transform_with_data_inputs_and_expected_outputs():
    tape = TapeCallbackFunction()
    p = Pipeline([
        OutputTransformerWrapper(Pipeline([MultiplyByN(2), FitCallbackStep(tape)]))
    ])
    data_inputs, expected_outputs = _create_data_source((10, 10))

    p, data_container = p.handle_fit_transform(DataContainer(
        data_inputs=data_inputs,
        expected_outputs=expected_outputs
    ), ExecutionContext())

    assert np.array_equal(data_container.data_inputs, data_inputs)
    assert np.array_equal(data_container.expected_outputs, expected_outputs * 2)
    assert np.array_equal(tape.data[0][0], expected_outputs * 2)
    for i in range(10):
        assert tape.data[0][1][i] is None


def test_output_transformer_wrapper_should_transform_with_data_inputs_and_expected_outputs():
    p = Pipeline([
        OutputTransformerWrapper(MultiplyByN(2))
    ])
    data_inputs, expected_outputs = _create_data_source((10, 10))

    data_container = p.handle_transform(DataContainer(
        data_inputs=data_inputs,
        expected_outputs=expected_outputs
    ), ExecutionContext())

    assert np.array_equal(data_container.data_inputs, data_inputs)
    assert np.array_equal(data_container.expected_outputs, expected_outputs * 2)


def _create_data_source(shape):
    data_inputs = np.random.random(shape).astype(np.float32)
    expected_outputs = np.random.random(shape).astype(np.float32)
    return data_inputs, expected_outputs
