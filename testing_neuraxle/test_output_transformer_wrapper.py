import numpy as np
import pytest
from neuraxle.base import BaseTransformer
from neuraxle.base import ExecutionContext as CX
from neuraxle.base import ForceHandleMixin
from neuraxle.data_container import EOT
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline
from neuraxle.steps.misc import FitCallbackStep, TapeCallbackFunction
from neuraxle.steps.numpy import MultiplyByN
from neuraxle.steps.output_handlers import IdsInputAndOutputTransformerWrapper, OutputTransformerWrapper


class MultiplyByNInputAndOutput(BaseTransformer):
    def __init__(self, multiply_by=1):
        super().__init__(hyperparams=HyperparameterSamples({'multiply_by': multiply_by}))

    def transform(self, data_inputs):
        ids, data_inputs, expected_outputs = data_inputs

        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        if not isinstance(expected_outputs, np.ndarray):
            expected_outputs = np.array(expected_outputs)

        return ids, data_inputs * self.hyperparams['multiply_by'], expected_outputs * self.hyperparams['multiply_by']

    def inverse_transform(self, processed_outputs):
        ids, processed_outputs, expected_outputs = processed_outputs

        if not isinstance(processed_outputs, np.ndarray):
            processed_outputs = np.array(processed_outputs)

        if not isinstance(expected_outputs, np.ndarray):
            expected_outputs = np.array(expected_outputs)

        return ids, processed_outputs / self.hyperparams['multiply_by'], expected_outputs / self.hyperparams['multiply_by']


def test_output_transformer_wrapper_should_fit_with_data_inputs_and_expected_outputs_as_data_inputs():
    tape = TapeCallbackFunction()
    p = OutputTransformerWrapper(FitCallbackStep(tape))
    data_inputs, expected_outputs = _create_data_source((10, 10))

    p.fit(data_inputs, expected_outputs)

    assert np.array_equal(tape.data[0][0], expected_outputs)
    for i in range(10):
        _first_eot_seen: EOT = tape.data[0][1]
        assert _first_eot_seen is None


def test_output_transformer_wrapper_should_fit_transform_with_data_inputs_and_expected_outputs():
    tape = TapeCallbackFunction()
    p = OutputTransformerWrapper(Pipeline([MultiplyByN(2), FitCallbackStep(tape)]))
    data_inputs, expected_outputs = _create_data_source((10, 10))

    p, data_container = p.handle_fit_transform(DACT(
        data_inputs=data_inputs,
        expected_outputs=expected_outputs
    ), CX())

    assert np.array_equal(data_container.data_inputs, data_inputs)
    assert np.array_equal(data_container.expected_outputs, expected_outputs * 2)
    assert np.array_equal(tape.data[0][0], expected_outputs * 2)
    for i in range(10):
        _first_eot_seen: EOT = tape.data[0][1]
        assert _first_eot_seen is None


def test_output_transformer_wrapper_should_transform_with_data_inputs_and_expected_outputs():
    p = OutputTransformerWrapper(MultiplyByN(2))
    data_inputs, expected_outputs = _create_data_source((10, 10))

    data_container = p.handle_transform(DACT(
        data_inputs=data_inputs,
        expected_outputs=expected_outputs
    ), CX())

    assert np.array_equal(data_container.data_inputs, data_inputs)
    assert np.array_equal(data_container.expected_outputs, expected_outputs * 2)


def test_input_and_output_transformer_wrapper_should_fit_with_data_inputs_and_expected_outputs_as_data_inputs():
    tape = TapeCallbackFunction()
    p = IdsInputAndOutputTransformerWrapper(FitCallbackStep(tape))
    ids, data_inputs, expected_outputs = _create_data_source((10, 10), with_ids=True)

    p.handle_fit(DACT(ids=ids, di=data_inputs, eo=expected_outputs), CX())

    assert np.array_equal(tape.data[0][0][0], ids)
    assert np.array_equal(tape.data[0][0][1], data_inputs)
    assert np.array_equal(tape.data[0][0][2], expected_outputs)


def test_input_and_output_transformer_wrapper_should_fit_transform_with_data_inputs_and_expected_outputs():
    tape = TapeCallbackFunction()
    p = IdsInputAndOutputTransformerWrapper(Pipeline([MultiplyByNInputAndOutput(2), FitCallbackStep(tape)]))
    ids, data_inputs, expected_outputs = _create_data_source((10, 10), with_ids=True)

    p, data_container = p.handle_fit_transform(DACT(
        ids=ids,
        data_inputs=data_inputs,
        expected_outputs=expected_outputs
    ), CX())

    assert np.array_equal(data_container.data_inputs, data_inputs * 2)
    assert np.array_equal(data_container.expected_outputs, expected_outputs * 2)
    assert np.array_equal(tape.data[0][0][0], ids)
    assert np.array_equal(tape.data[0][0][1], data_inputs * 2)
    assert np.array_equal(tape.data[0][0][2], expected_outputs * 2)


def test_input_and_output_transformer_wrapper_should_transform_with_data_inputs_and_expected_outputs():
    p = IdsInputAndOutputTransformerWrapper(MultiplyByNInputAndOutput(2))
    data_inputs, expected_outputs = _create_data_source((10, 10))

    data_container = p.handle_transform(DACT(
        data_inputs=data_inputs,
        expected_outputs=expected_outputs
    ), CX())

    assert np.array_equal(data_container.data_inputs, data_inputs * 2)
    assert np.array_equal(data_container.expected_outputs, expected_outputs * 2)


class ChangeLenDataInputs(BaseTransformer):
    """
    This should raise an error because it does not return the same length for data_inputs and expected_outputs
    """

    def __init__(self):
        super().__init__()

    def transform(self, data_inputs):
        ids, data_inputs, expected_outputs = data_inputs
        return ids, data_inputs[0:int(len(data_inputs) / 2)], expected_outputs


class ChangeLenDataInputsAndExpectedOutputsWithoutIds(BaseTransformer):
    """
    This should raise an error because ids are not changed to fit the new length of data_inputs and expected_outputs
    """

    def __init__(self):
        BaseTransformer.__init__(self)

    def transform(self, data_inputs):
        ids, data_inputs, expected_outputs = data_inputs
        _clip = int(len(data_inputs) / 2)
        return ids, data_inputs[:_clip], expected_outputs[:_clip]


class DoubleData(ForceHandleMixin, BaseTransformer):
    """
    This should double the data given in entry. Expects to be wrapped in an InputAndOutputTransformerWrapper.
    """

    def __init__(self):
        BaseTransformer.__init__(self)
        ForceHandleMixin.__init__(self)

    def _transform_data_container(self, data_container: DACT, context: CX) -> DACT:
        ids, di, eo = data_container.data_inputs
        return DACT(
            data_inputs=(
                ids * 2 if ids is not None else None,
                di[0].tolist() * 2,
                eo[0].tolist() * 2),
            ids=list(data_container.ids)
        )


def test_input_and_output_transformer_wrapper_should_not_return_a_different_amount_of_data_inputs_and_expected_outputs():
    with pytest.raises(AssertionError):
        p = IdsInputAndOutputTransformerWrapper(ChangeLenDataInputs())
        ids, data_inputs, expected_outputs = _create_data_source((10, 10), with_ids=True)

        p.handle_transform(DACT(
            ids=ids,
            data_inputs=data_inputs,
            expected_outputs=expected_outputs
        ), CX())


def test_input_and_output_transformer_wrapper_should_raise_an_assertion_error_if_ids_have_not_been_resampled_correctly():
    with pytest.raises(AssertionError):
        p = IdsInputAndOutputTransformerWrapper(ChangeLenDataInputsAndExpectedOutputsWithoutIds())
        ids, data_inputs, expected_outputs = _create_data_source((10, 10), with_ids=True)

        p.handle_transform(DACT(
            ids=ids,
            data_inputs=data_inputs,
            expected_outputs=expected_outputs
        ), CX())


def test_data_doubler():
    p = IdsInputAndOutputTransformerWrapper(DoubleData())
    ids, data_inputs, expected_outputs = _create_data_source((10, 10), with_ids=True)

    out = p.handle_transform(DACT(
        ids=ids,
        data_inputs=data_inputs,
        expected_outputs=expected_outputs
    ), CX())

    doubled_length = len(out.data_inputs)
    assert doubled_length == 2 * len(data_inputs)
    assert doubled_length == len(out.expected_outputs)
    assert doubled_length == len(out.ids)


def _create_data_source(shape, with_ids=False):
    data_inputs = np.random.random(shape).astype(np.float32)
    expected_outputs = np.random.random(shape).astype(np.float32)
    if with_ids:
        ids = list(range(len(data_inputs)))
        return ids, data_inputs, expected_outputs
    else:
        return data_inputs, expected_outputs
