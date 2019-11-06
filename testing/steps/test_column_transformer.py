import copy
from typing import Any

import numpy as np
import pytest
from neuraxle.base import BaseStep
from neuraxle.steps.column_transformer import ColumnTransformer


class MultiplyBy2(BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        self.fitted_data = []

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        self._add_fitted_data(data_inputs, expected_outputs)
        return self

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BaseStep', Any):
        self._add_fitted_data(data_inputs, expected_outputs)
        return self, self.transform(data_inputs)

    def _add_fitted_data(self, data_inputs, expected_outputs):
        self.fitted_data.append((copy.deepcopy(data_inputs), copy.deepcopy(expected_outputs)))

    def transform(self, data_inputs):
        return data_inputs * 2


class ColumnChooserTestCase:
    def __init__(
            self,
            data_inputs,
            expected_processed_outputs,
            column_transformer_tuple_list,
            n_dimension,
            expected_step_key=None,
            expected_fitted_data=None,
            expected_outputs=None
    ):
        self.n_dimension = n_dimension
        self.expected_step_key = expected_step_key
        self.data_inputs = data_inputs
        self.expected_outputs = expected_outputs
        self.expected_fitted_data = expected_fitted_data
        self.expected_processed_outputs = expected_processed_outputs
        self.column_transformer_tuple_list = column_transformer_tuple_list


test_case_index_int = ColumnChooserTestCase(
    data_inputs=np.array([
        [0, 1, 2, 3],
        [10, 11, 12, 13],
        [20, 21, 22, 23]
    ]),
    expected_outputs=np.array([
        [0, 1, 2, 3],
        [10, 11, 12, 13],
        [20, 21, 22, 23]
    ]),
    expected_processed_outputs=np.array([
        [2],
        [22],
        [42]
    ]),
    expected_fitted_data=[(
        [[1], [11], [21]],
        [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]]
    )],
    expected_step_key='1_MultiplyBy2',
    column_transformer_tuple_list=[
        (1, MultiplyBy2())
    ],
    n_dimension=3
)

test_case_index_start_end = ColumnChooserTestCase(
    data_inputs=np.array([
        [0, 1, 2, 3],
        [10, 11, 12, 13],
        [20, 21, 22, 23]
    ]),
    expected_outputs=np.array([
        [0, 1, 2, 3],
        [10, 11, 12, 13],
        [20, 21, 22, 23]
    ]),
    expected_processed_outputs=np.array([
        [0, 2],
        [20, 22],
        [40, 42]
    ]),
    expected_fitted_data=[(
        [[0, 1], [10, 11], [20, 21]],
        [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]]
    )],
    expected_step_key='slice(0, 2, None)_MultiplyBy2',
    column_transformer_tuple_list=[
        (slice(0, 2), MultiplyBy2())
    ],
    n_dimension=3
)

test_case_index_start = ColumnChooserTestCase(
    data_inputs=np.array([
        [0, 1, 2, 3],
        [10, 11, 12, 13],
        [20, 21, 22, 23]
    ]),
    expected_outputs=np.array([
        [0, 1, 2, 3],
        [10, 11, 12, 13],
        [20, 21, 22, 23]
    ]),
    expected_processed_outputs=np.array([
        [2, 4, 6],
        [22, 24, 26],
        [42, 44, 46]
    ]),
    expected_fitted_data=[(
        [[1, 2, 3], [11, 12, 13], [21, 22, 23]],
        [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]]
    )],
    expected_step_key='slice(1, None, None)_MultiplyBy2',
    column_transformer_tuple_list=[
        (slice(1, None), MultiplyBy2())
    ],
    n_dimension=3
)

test_case_index_end = ColumnChooserTestCase(
    data_inputs=np.array([
        [0, 1, 2, 3],
        [10, 11, 12, 13],
        [20, 21, 22, 23]
    ]),
    expected_outputs=np.array([
        [0, 1, 2, 3],
        [10, 11, 12, 13],
        [20, 21, 22, 23]
    ]),
    expected_processed_outputs=np.array([
        [0, 2],
        [20, 22],
        [40, 42]
    ]),
    expected_fitted_data=[(
        [[0, 1], [10, 11], [20, 21]],
        [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]]
    )],
    expected_step_key='slice(None, 2, None)_MultiplyBy2',
    column_transformer_tuple_list=[
        (slice(None, 2), MultiplyBy2())
    ],
    n_dimension=3
)

test_case_index_last = ColumnChooserTestCase(
    data_inputs=np.array([
        [0, 1, 2, 3],
        [10, 11, 12, 13],
        [20, 21, 22, 23]
    ]),
    expected_outputs=np.array([
        [0, 1, 2, 3],
        [10, 11, 12, 13],
        [20, 21, 22, 23]
    ]),
    expected_processed_outputs=np.array([
        [0, 2, 4],
        [20, 22, 24],
        [40, 42, 44]
    ]),
    expected_fitted_data=[
        (
            [[0, 1, 2], [10, 11, 12], [20, 21, 22]],
            [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]]
        )
    ],
    expected_step_key='slice(None, -1, None)_MultiplyBy2',
    column_transformer_tuple_list=[
        (slice(None, -1), MultiplyBy2())
    ],
    n_dimension=3
)

test_case_list_of_columns = ColumnChooserTestCase(
    data_inputs=np.array([
        [0, 1, 2, 3],
        [10, 11, 12, 13],
        [20, 21, 22, 23]
    ]),
    expected_outputs=np.array([
        [0, 1, 2, 3],
        [10, 11, 12, 13],
        [20, 21, 22, 23]
    ]),
    expected_processed_outputs=np.array([
        [0, 4],
        [20, 24],
        [40, 44]
    ]),
    expected_fitted_data=[(
        [[0, 2], [10, 12], [20, 22]],
        [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]]
    )],
    expected_step_key='[0, 2]_MultiplyBy2',
    column_transformer_tuple_list=[
        ([0, 2], MultiplyBy2())
    ],
    n_dimension=3
)

@pytest.mark.parametrize("test_case", [
    copy.deepcopy(test_case_index_int),
    copy.deepcopy(test_case_index_start_end),
    copy.deepcopy(test_case_index_start),
    copy.deepcopy(test_case_index_end),
    copy.deepcopy(test_case_index_last),
    copy.deepcopy(test_case_list_of_columns),
])
def test_column_transformer_transform_should_support_indexes(test_case: ColumnChooserTestCase):
    data_inputs = test_case.data_inputs
    column_transformer = ColumnTransformer(test_case.column_transformer_tuple_list)

    outputs = column_transformer.transform(data_inputs)

    assert np.array_equal(outputs, test_case.expected_processed_outputs)


@pytest.mark.parametrize("test_case", [
    copy.deepcopy(test_case_index_int),
    copy.deepcopy(test_case_index_start_end),
    copy.deepcopy(test_case_index_start),
    copy.deepcopy(test_case_index_end),
    copy.deepcopy(test_case_index_last),
    copy.deepcopy(test_case_list_of_columns)
])
def test_column_transformer_fit_transform_should_support_indexes(test_case: ColumnChooserTestCase):
    data_inputs = test_case.data_inputs
    expected_outputs = test_case.expected_outputs
    column_transformer = ColumnTransformer(test_case.column_transformer_tuple_list)

    column_transformer, outputs = column_transformer.fit_transform(data_inputs, expected_outputs)

    assert np.array_equal(outputs, test_case.expected_processed_outputs)
    actual_fitted_data = column_transformer[test_case.expected_step_key]['MultiplyBy2'].fitted_data
    expected_fitted_data = test_case.expected_fitted_data
    assert_data_fitted_properly(actual_fitted_data, expected_fitted_data)


@pytest.mark.parametrize("test_case", [
    copy.deepcopy(test_case_index_int),
    copy.deepcopy(test_case_index_start_end),
    copy.deepcopy(test_case_index_start),
    copy.deepcopy(test_case_index_end),
    copy.deepcopy(test_case_index_last),
    copy.deepcopy(test_case_list_of_columns)
])
def test_column_transformer_fit_should_support_indexes(test_case: ColumnChooserTestCase):
    data_inputs = test_case.data_inputs
    column_transformer = ColumnTransformer(test_case.column_transformer_tuple_list)

    column_transformer = column_transformer.fit(data_inputs, test_case.expected_outputs)

    actual_fitted_data = column_transformer[test_case.expected_step_key]['MultiplyBy2'].fitted_data
    expected_fitted_data = test_case.expected_fitted_data
    assert_data_fitted_properly(actual_fitted_data, expected_fitted_data)


def test_column_transformer_fit_should_support_multiple_tuples():
    # Given
    test_case = ColumnChooserTestCase(
        data_inputs=np.array([
            [1, 1, 2, 3],
            [10, 11, 12, 13],
            [20, 21, 22, 23]
        ]),
        expected_outputs=np.array([
            [0, 1, 2, 3],
            [10, 11, 12, 13],
            [20, 21, 22, 23]
        ]),
        expected_processed_outputs=np.array([
            [2, 2, 2, 3],
            [20, 22, 12, 13],
            [40, 42, 44, 46]
        ]),
        column_transformer_tuple_list=[
            (slice(0, 2), MultiplyBy2()),
            (2, MultiplyBy2())
        ],
        n_dimension=3
    )
    data_inputs = test_case.data_inputs
    column_transformer = ColumnTransformer(test_case.column_transformer_tuple_list)

    # When
    column_transformer = column_transformer.fit(data_inputs, test_case.expected_outputs)

    # Then
    actual_fitted_data = column_transformer['2_MultiplyBy2']['MultiplyBy2'].fitted_data
    expected_fitted_data = [([[2], [12], [22]], [[0, 1, 2, 3],[10, 11, 12, 13], [20, 21, 22, 23]])]
    assert_data_fitted_properly(actual_fitted_data, expected_fitted_data)

    actual_fitted_data = column_transformer['slice(0, 2, None)_MultiplyBy2']['MultiplyBy2'].fitted_data
    expected_fitted_data = [([[1, 1], [10, 11], [20, 21]], [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]])]
    assert_data_fitted_properly(actual_fitted_data, expected_fitted_data)


def test_column_transformer_fit_transform_should_support_multiple_tuples():
    # Given
    test_case = ColumnChooserTestCase(
        data_inputs=np.array([
            [1, 1, 2, 3],
            [10, 11, 12, 13],
            [20, 21, 22, 23]
        ]),
        expected_outputs=np.array([
            [0, 1, 2, 3],
            [10, 11, 12, 13],
            [20, 21, 22, 23]
        ]),
        expected_processed_outputs=np.array([
            [2, 2, 4],
            [20, 22, 24],
            [40, 42, 44]
        ]),
        column_transformer_tuple_list=[
            (slice(0, 2), MultiplyBy2()),
            (2, MultiplyBy2())
        ],
        n_dimension=3
    )
    data_inputs = test_case.data_inputs
    column_transformer = ColumnTransformer(test_case.column_transformer_tuple_list)

    # When
    column_transformer, outputs = column_transformer.fit_transform(data_inputs, test_case.expected_outputs)

    # Then
    assert np.array_equal(test_case.expected_processed_outputs, outputs)
    actual_fitted_data = column_transformer['2_MultiplyBy2']['MultiplyBy2'].fitted_data
    expected_fitted_data = [([[2], [12], [22]], [[0, 1, 2, 3],[10, 11, 12, 13], [20, 21, 22, 23]])]
    assert_data_fitted_properly(actual_fitted_data, expected_fitted_data)

    actual_fitted_data = column_transformer['slice(0, 2, None)_MultiplyBy2']['MultiplyBy2'].fitted_data
    expected_fitted_data = [([[1, 1], [10, 11], [20, 21]], [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]])]
    assert_data_fitted_properly(actual_fitted_data, expected_fitted_data)


def test_column_transformer_transform_should_support_multiple_tuples():
    # Given
    test_case = ColumnChooserTestCase(
        data_inputs=np.array([
            [1, 1, 2, 3],
            [10, 11, 12, 13],
            [20, 21, 22, 23]
        ]),
        expected_outputs=np.array([
            [0, 1, 2, 3],
            [10, 11, 12, 13],
            [20, 21, 22, 23]
        ]),
        expected_processed_outputs=np.array([
            [2, 2, 4],
            [20, 22, 24],
            [40, 42, 44]
        ]),
        column_transformer_tuple_list=[
            (slice(0, 2), MultiplyBy2()),
            (2, MultiplyBy2())
        ],
        n_dimension=3
    )
    data_inputs = test_case.data_inputs
    column_transformer = ColumnTransformer(test_case.column_transformer_tuple_list)

    # When
    outputs = column_transformer.transform(data_inputs)

    # Then
    assert np.array_equal(test_case.expected_processed_outputs, outputs)


def assert_data_fitted_properly(actual_fitted_data, expected_fitted_data):
    for actual_fitted, expected_fitted in zip(actual_fitted_data, expected_fitted_data):
        assert np.array_equal(actual_fitted[0], expected_fitted[0])
        assert np.array_equal(actual_fitted[1], expected_fitted[1])
