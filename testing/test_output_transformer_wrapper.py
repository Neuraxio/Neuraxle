import numpy as np
import pytest

from neuraxle.base import ExecutionContext, BaseTransformer
from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline
from neuraxle.steps.misc import FitCallbackStep, TapeCallbackFunction
from neuraxle.steps.numpy import MultiplyByN
from neuraxle.steps.output_handlers import OutputTransformerWrapper, InputAndOutputTransformerWrapper


class MultiplyByNInputAndOutput(BaseTransformer):
    def __init__(self, multiply_by=1):
        super().__init__(hyperparams=HyperparameterSamples({'multiply_by': multiply_by}))

    def transform(self, data_inputs):
        data_inputs, expected_outputs = data_inputs

        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        if not isinstance(expected_outputs, np.ndarray):
            expected_outputs = np.array(expected_outputs)

        return data_inputs * self.hyperparams['multiply_by'], expected_outputs * self.hyperparams['multiply_by']

    def inverse_transform(self, data_inputs):
        data_inputs, expected_outputs = data_inputs

        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        if not isinstance(expected_outputs, np.ndarray):
            expected_outputs = np.array(expected_outputs)

        return data_inputs / self.hyperparams['multiply_by'], expected_outputs / self.hyperparams['multiply_by']


def test_output_transformer_wrapper_should_fit_with_data_inputs_and_expected_outputs_as_data_inputs():
    tape = TapeCallbackFunction()
    p = OutputTransformerWrapper(FitCallbackStep(tape))
    data_inputs, expected_outputs = _create_data_source((10, 10))

    p.fit(data_inputs, expected_outputs)

    assert np.array_equal(tape.data[0][0], expected_outputs)
    for i in range(10):
        assert tape.data[0][1][i] is None


def test_output_transformer_wrapper_should_fit_transform_with_data_inputs_and_expected_outputs():
    tape = TapeCallbackFunction()
    p = OutputTransformerWrapper(Pipeline([MultiplyByN(2), FitCallbackStep(tape)]))
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
    p = OutputTransformerWrapper(MultiplyByN(2))
    data_inputs, expected_outputs = _create_data_source((10, 10))

    data_container = p.handle_transform(DataContainer(
        data_inputs=data_inputs,
        expected_outputs=expected_outputs
    ), ExecutionContext())

    assert np.array_equal(data_container.data_inputs, data_inputs)
    assert np.array_equal(data_container.expected_outputs, expected_outputs * 2)


def test_input_and_output_transformer_wrapper_should_fit_with_data_inputs_and_expected_outputs_as_data_inputs():
    tape = TapeCallbackFunction()
    p = InputAndOutputTransformerWrapper(FitCallbackStep(tape))
    data_inputs, expected_outputs = _create_data_source((10, 10))

    p.fit(data_inputs, expected_outputs)

    assert np.array_equal(tape.data[0][0][0], data_inputs)
    assert np.array_equal(tape.data[0][0][1], expected_outputs)


def test_input_and_output_transformer_wrapper_should_fit_transform_with_data_inputs_and_expected_outputs():
    tape = TapeCallbackFunction()
    p = InputAndOutputTransformerWrapper(Pipeline([MultiplyByNInputAndOutput(2), FitCallbackStep(tape)]))
    data_inputs, expected_outputs = _create_data_source((10, 10))

    p, data_container = p.handle_fit_transform(DataContainer(
        data_inputs=data_inputs,
        expected_outputs=expected_outputs
    ), ExecutionContext())

    assert np.array_equal(data_container.data_inputs, data_inputs * 2)
    assert np.array_equal(data_container.expected_outputs, expected_outputs * 2)
    assert np.array_equal(tape.data[0][0][0], data_inputs * 2)
    assert np.array_equal(tape.data[0][0][1], expected_outputs * 2)


def test_input_and_output_transformer_wrapper_should_transform_with_data_inputs_and_expected_outputs():
    p = InputAndOutputTransformerWrapper(MultiplyByNInputAndOutput(2))
    data_inputs, expected_outputs = _create_data_source((10, 10))

    data_container = p.handle_transform(DataContainer(
        data_inputs=data_inputs,
        expected_outputs=expected_outputs
    ), ExecutionContext())

    assert np.array_equal(data_container.data_inputs, data_inputs * 2)
    assert np.array_equal(data_container.expected_outputs, expected_outputs * 2)


class ChangeLenDataInputs(BaseTransformer):
    def __init__(self):
        super().__init__()

    def transform(self, data_inputs):
        data_inputs, expected_outputs = data_inputs
        return data_inputs[0:int(len(data_inputs) / 2)], expected_outputs


class ChangeLenDataInputsAndExpectedOutputs(BaseTransformer):
    def __init__(self):
        super().__init__()

    def transform(self, data_inputs):
        data_inputs, expected_outputs = data_inputs
        return data_inputs[0:int(len(data_inputs) / 2)], expected_outputs


def test_input_and_output_transformer_wrapper_should_not_return_a_different_amount_of_data_inputs_and_expected_outputs():
    with pytest.raises(AssertionError):
        p = InputAndOutputTransformerWrapper(ChangeLenDataInputs())
        data_inputs, expected_outputs = _create_data_source((10, 10))

        p.handle_transform(DataContainer(
            data_inputs=data_inputs,
            expected_outputs=expected_outputs
        ), ExecutionContext())


def test_input_and_output_transformer_wrapper_should_raise_an_assertion_error_if_current_ids_have_not_been_resampled_correctly():
    with pytest.raises(AssertionError):
        p = InputAndOutputTransformerWrapper(ChangeLenDataInputsAndExpectedOutputs())
        data_inputs, expected_outputs = _create_data_source((10, 10))

        p.handle_transform(DataContainer(
            data_inputs=data_inputs,
            expected_outputs=expected_outputs
        ), ExecutionContext())


def _create_data_source(shape):
    data_inputs = np.random.random(shape).astype(np.float32)
    expected_outputs = np.random.random(shape).astype(np.float32)
    return data_inputs, expected_outputs
