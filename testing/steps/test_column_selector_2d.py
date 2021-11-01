import numpy as np
import pytest

from neuraxle.steps.column_transformer import ColumnSelector2D, ColumnsSelectorND, NumpyColumnSelector2D


@pytest.mark.parametrize('column_selector_2d_class', [ColumnSelector2D, NumpyColumnSelector2D])
def test_column_selector_2d_should_select_range(column_selector_2d_class):
    step = column_selector_2d_class(range(0, 10))
    data_inputs, expected_outputs = _create_data_source((20, 20))

    outputs = step.transform(data_inputs)

    assert np.array_equal(outputs, data_inputs[..., :10])


@pytest.mark.parametrize('column_selector_2d_class', [ColumnSelector2D, NumpyColumnSelector2D])
def test_column_selector_2d_should_select_int(column_selector_2d_class):
    step = column_selector_2d_class(10)
    data_inputs, expected_outputs = _create_data_source((20, 20))

    outputs = step.transform(data_inputs)

    expected_data_inputs = np.expand_dims(data_inputs[..., 10], axis=-1)
    assert np.array_equal(outputs, expected_data_inputs)


@pytest.mark.parametrize('column_selector_2d_class', [ColumnSelector2D, NumpyColumnSelector2D])
def test_column_selector_2d_should_select_slice(column_selector_2d_class):
    step = column_selector_2d_class(slice(0, 10, 1))
    data_inputs, expected_outputs = _create_data_source((20, 20))

    outputs = step.transform(data_inputs)

    assert np.array_equal(outputs, data_inputs[..., :10])


@pytest.mark.parametrize('column_selector_2d_class', [ColumnSelector2D, NumpyColumnSelector2D])
def test_column_selector_2d_should_select_list_of_indexes(column_selector_2d_class):
    step = column_selector_2d_class([0, 1, 2])
    data_inputs, expected_outputs = _create_data_source((20, 20))

    outputs = step.transform(data_inputs)

    assert np.array_equal(outputs, data_inputs[..., :3])


def test_column_selector_nd_should_transform_with_column_selector_2d():
    step = ColumnsSelectorND(0, n_dimension=2)
    data_inputs, expected_outputs = _create_data_source((20, 20))

    outputs = step.transform(data_inputs)

    assert np.array_equal(outputs, np.expand_dims(data_inputs[..., 0], axis=-1))


def _create_data_source(shape):
    data_inputs = np.random.random(shape).astype(np.float32)
    expected_outputs = np.random.random(shape).astype(np.float32)
    return data_inputs, expected_outputs
