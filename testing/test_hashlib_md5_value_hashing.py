import numpy as np
from neuraxle.steps.output_handlers import InputAndOutputTransformerMixin, InputAndOutputTransformerWrapper

from neuraxle.data_container import DataContainer

from neuraxle.base import BaseTransformer, HashlibMd5ValueHasher, ExecutionContext


class WindowTimeSeries(InputAndOutputTransformerMixin, BaseTransformer):
    def __init__(self):
        BaseTransformer.__init__(self,hashers=[HashlibMd5ValueHasher()])
        InputAndOutputTransformerMixin.__init__(self)

    def transform(self, data_inputs):
        di, eo = data_inputs
        new_di, new_eo = np.array_split(np.array(di), 2)
        return new_di, new_eo


class WindowTimeSeriesForOutputTransformerWrapper(BaseTransformer):
    def __init__(self):
        BaseTransformer.__init__(self)

    def transform(self, data_inputs):
        di, eo = data_inputs
        new_di, new_eo = np.array_split(np.array(di), 2)
        return new_di, new_eo


def test_transform_input_and_output_transformer_mixin_with_hashlib_md5_value_hasher():
    data_container: DataContainer = WindowTimeSeries().handle_transform(
        data_container=DataContainer(
            data_inputs=np.array(list(range(10))),
            expected_outputs=np.array(list(range(10)))
        ),
        context=ExecutionContext()
    )

    assert np.array_equal(data_container.data_inputs, np.array(list(range(0, 5))))
    assert np.array_equal(data_container.expected_outputs, np.array(list(range(5, 10))))


def test_transform_input_and_output_transformer_wrapper_with_hashlib_md5_value_hasher():
    step = InputAndOutputTransformerWrapper(WindowTimeSeriesForOutputTransformerWrapper()) \
        .set_hashers([HashlibMd5ValueHasher()])

    data_container = step.handle_transform(
        data_container=DataContainer(
            data_inputs=np.array(list(range(10))),
            expected_outputs=np.array(list(range(10)))
        ),
        context=ExecutionContext()
    )

    assert np.array_equal(data_container.data_inputs, np.array(list(range(0, 5))))
    assert np.array_equal(data_container.expected_outputs, np.array(list(range(5, 10))))


def test_fit_transform_input_and_output_transformer_mixin_with_hashlib_md5_value_hasher():
    step, data_container = WindowTimeSeries().handle_fit_transform(
        data_container=DataContainer(
            data_inputs=np.array(list(range(10))),
            expected_outputs=np.array(list(range(10)))
        ),
        context=ExecutionContext()
    )

    assert np.array_equal(data_container.data_inputs, np.array(list(range(0, 5))))
    assert np.array_equal(data_container.expected_outputs, np.array(list(range(5, 10))))


def test_fit_transform_input_and_output_transformer_wrapper_with_hashlib_md5_value_hasher():
    step = InputAndOutputTransformerWrapper(WindowTimeSeriesForOutputTransformerWrapper()) \
        .set_hashers([HashlibMd5ValueHasher()])

    step, data_container = step.handle_fit_transform(
        data_container=DataContainer(
            data_inputs=np.array(list(range(10))),
            expected_outputs=np.array(list(range(10)))
        ),
        context=ExecutionContext()
    )

    assert np.array_equal(data_container.data_inputs, np.array(list(range(0, 5))))
    assert np.array_equal(data_container.expected_outputs, np.array(list(range(5, 10))))
