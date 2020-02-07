import numpy as np

from neuraxle.base import ExecutionContext
from neuraxle.data_container import DataContainer
from neuraxle.pipeline import Pipeline
from neuraxle.steps.loop import FlattenForEach
from neuraxle.steps.numpy import MultiplyByN
from neuraxle.steps.output_handlers import OutputTransformerWrapper


def test_flatten_for_each_should_transform_data_inputs():
    p = Pipeline([
        FlattenForEach(MultiplyByN(2))
    ])
    data_shape = (10, 10)
    data_inputs, expected_outputs = _create_data_source(data_shape)

    outputs = p.transform(data_inputs)

    assert np.array(outputs).shape == data_shape
    assert np.array_equal(outputs, data_inputs * 2)


def test_flatten_for_each_should_transform_data_inputs_and_expected_outputs():
    p = Pipeline([
        FlattenForEach(Pipeline([
            MultiplyByN(2),
            OutputTransformerWrapper(MultiplyByN(2))
        ]))
    ])
    data_shape = (10, 10)
    data_inputs, expected_outputs = _create_data_source(data_shape)

    p, outputs = p.handle_fit_transform(DataContainer(data_inputs=data_inputs, expected_outputs=expected_outputs), ExecutionContext())

    assert np.array(outputs.data_inputs).shape == data_shape
    assert np.array_equal(outputs.data_inputs, data_inputs * 2)
    assert np.array(outputs.expected_outputs).shape == data_shape
    assert np.array_equal(outputs.expected_outputs, expected_outputs * 2)


def _create_data_source(shape):
    data_inputs = np.random.random(shape).astype(np.float32)
    expected_outputs = np.random.random(shape).astype(np.float32)
    return data_inputs, expected_outputs
