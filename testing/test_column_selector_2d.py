import numpy as np
import pytest

from neuraxle.steps.column_transformer import ColumnSelector2D, ColumnsSelectorND


def test_column_selector_2d_should_select_range():
    step = ColumnSelector2D(range(0, 10))
    data_inputs, expected_outputs = _create_data_source((20, 20))

    outputs = step.transform(data_inputs)

    assert np.array_equal(outputs, data_inputs[..., :10])


def test_column_selector_2d_should_select_int():
    step = ColumnSelector2D(10)
    data_inputs, expected_outputs = _create_data_source((20, 20))

    outputs = step.transform(data_inputs)

    expected_data_inputs = np.expand_dims(data_inputs[..., 10], axis=-1)
    assert np.array_equal(outputs, expected_data_inputs)


def test_column_selector_2d_should_select_slice():
    step = ColumnSelector2D(slice(0, 10, 1))
    data_inputs, expected_outputs = _create_data_source((20, 20))

    outputs = step.transform(data_inputs)

    assert np.array_equal(outputs, data_inputs[..., :10])


def test_column_selector_2d_should_select_list_of_indexes():
    step = ColumnSelector2D([0, 1, 2])
    data_inputs, expected_outputs = _create_data_source((20, 20))

    outputs = step.transform(data_inputs)

    assert np.array_equal(outputs, data_inputs[..., :3])
    pass


def test_column_selector_2d_should_throw_exception_on_unsupported_type():
    step = ColumnSelector2D('unsupported')
    data_inputs, expected_outputs = _create_data_source((20, 20))

    with pytest.raises(ValueError):
        step.transform(data_inputs)


def test_column_selector_nd_should_transform_with_column_selector_2d():
    step = ColumnsSelectorND(0)
    data_inputs, expected_outputs = _create_data_source((20, 20))

    outputs = step.transform(data_inputs)

    assert np.array_equal(outputs, np.expand_dims(data_inputs[..., 0], axis=-1))


def _create_data_source(shape):
    data_inputs = np.random.random(shape).astype(np.float32)
    expected_outputs = np.random.random(shape).astype(np.float32)
    return data_inputs, expected_outputs
