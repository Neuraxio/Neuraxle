import numpy as np
from neuraxle.base import BaseStep, DataContainer, NonFittableMixin, ExecutionContext, ExecutionMode
from neuraxle.steps.flow import ReversiblePreprocessingWrapper


class MultiplyBy2(NonFittableMixin, BaseStep):
    def transform(self, data_inputs):
        return data_inputs * 2

    def inverse_transform(self, processed_outputs):
        return processed_outputs / 2


class Add10(NonFittableMixin, BaseStep):
    def transform(self, data_inputs):
        return data_inputs + 10

    def inverse_transform(self, processed_outputs):
        return processed_outputs - 10


def test_reversible_preprocessing_wrapper():
    step = ReversiblePreprocessingWrapper(
        preprocessing_step=MultiplyBy2(),
        postprocessing_step=Add10()
    )

    outputs = step.handle_transform(
        DataContainer(current_ids=range(5), data_inputs=np.array(range(5)), expected_outputs=None),
        ExecutionContext.create_from_root(step, ExecutionMode.TRANSFORM, '/')
    )

    assert np.array_equal(outputs.data_inputs, np.array([5.0, 6.0, 7.0, 8.0, 9.0]))
