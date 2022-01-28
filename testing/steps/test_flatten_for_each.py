import numpy as np

from neuraxle.base import ExecutionContext
from neuraxle.data_container import DataContainer as DACT
from neuraxle.pipeline import Pipeline
from neuraxle.steps.loop import FlattenForEach
from neuraxle.steps.numpy import MultiplyByN
from neuraxle.steps.output_handlers import OutputTransformerWrapper

DATA_SHAPE = (3, 4)
FLAT_DATA_SHAPE = (3 * 4, )


def test_flatten_for_each_unflatten_should_transform_data_inputs():
    p = FlattenForEach(MultiplyByN(2), then_unflatten=True)
    data_inputs, _ = _create_random_of_shape(DATA_SHAPE)

    outputs = p.transform(data_inputs)

    assert np.array(outputs).shape == DATA_SHAPE
    assert np.array_equal(outputs, data_inputs * 2)


def test_flatten_for_each_should_transform_data_inputs():
    p = FlattenForEach(MultiplyByN(2), then_unflatten=False)
    data_inputs, _ = _create_random_of_shape(DATA_SHAPE)

    outputs = p.transform(data_inputs)

    assert np.array(outputs).shape == FLAT_DATA_SHAPE
    assert np.array_equal(outputs.flatten(), data_inputs.flatten() * 2)


def test_flatten_for_each_should_transform_data_inputs_and_expected_outputs():
    p = FlattenForEach(Pipeline([
        MultiplyByN(2),
        OutputTransformerWrapper(MultiplyByN(3))
    ]))
    # TODO: should use a tape here and ensure that the MultiplyByN received a flat 12 shape only once and not 3*4 things
    data_inputs, expected_outputs = _create_random_of_shape(DATA_SHAPE)

    p, outputs = p.handle_fit_transform(
        DACT(data_inputs=data_inputs, expected_outputs=expected_outputs), ExecutionContext())

    assert np.array(outputs.data_inputs).shape == DATA_SHAPE
    assert np.array_equal(outputs.data_inputs, data_inputs * 2)
    assert np.array(outputs.expected_outputs).shape == DATA_SHAPE
    assert np.array_equal(outputs.expected_outputs, expected_outputs * 3)


def _create_random_of_shape(shape):
    data_inputs = np.random.random(shape).astype(np.float32)
    expected_outputs = np.random.random(shape).astype(np.float32)
    return data_inputs, expected_outputs
