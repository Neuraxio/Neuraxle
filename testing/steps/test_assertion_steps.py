from unittest import TestCase
import numpy as np
import pytest


from neuraxle.base import AssertExpectedOutputIsNone, BaseStep, ExecutionContext, ExecutionMode, ExecutionPhase, HandleOnlyMixin, NonFittableMixin
from neuraxle.data_container import DataContainer as DACT
from neuraxle.pipeline import Pipeline


class SomeAssertionStep(NonFittableMixin, HandleOnlyMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        HandleOnlyMixin.__init__(self)

    def _transform_data_container(self, data_container: DACT, context: ExecutionContext) -> DACT:
        _, data_inputs, expected_outputs = data_container.tolist().unpack()
        if expected_outputs is not None:
            self._assert_equals(data_inputs, expected_outputs, "Assertion failed", context)
        return data_inputs


class TestAssertionMethodInSteps(TestCase):

    def test_assertion_step_logs_and_raises_with_pipeline(self):
        data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        expected_outputs = data_inputs * 2
        dact = DACT(data_inputs, None, expected_outputs)
        p = Pipeline([SomeAssertionStep()])

        with self.assertLogs() as captured:
            with pytest.raises(AssertionError):
                p.handle_fit_transform(dact, context=ExecutionContext())

            self.assertIn("Assertion failed", captured.output[0])

    def test_assertion_step_just_logs_with_pipeline_in_prod(self):
        data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        expected_outputs = data_inputs * 2
        dact = DACT(data_inputs, None, expected_outputs)
        p = Pipeline([SomeAssertionStep()])
        context = ExecutionContext(execution_phase=ExecutionPhase.PROD)
        try:
            p = p.handle_fit(dact, context=context)
        except AssertionError:
            pass

        with self.assertLogs() as captured:
            p.handle_predict(dact, context=context)

            # assert that the log still at least contains the expected message:
            self.assertIn("Assertion failed", captured.output[0])


def test_expectedoutputnull_raise_exception_when_notnull(tmpdir):
    data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_outputs = data_inputs * 2

    p = Pipeline([AssertExpectedOutputIsNone()])

    with pytest.raises(AssertionError) as error_info:
        p.fit_transform(data_inputs, expected_outputs)


def test_expectedoutputnull_is_fine_when_null(tmpdir):
    data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_outputs = None

    p = Pipeline([AssertExpectedOutputIsNone()])
    p.fit_transform(data_inputs, expected_outputs)
