import numpy as np
import pytest

from neuraxle.base import ExecutionMode
from neuraxle.pipeline import Pipeline
from neuraxle.steps.flow import ReversiblePreprocessingWrapper
from neuraxle.steps.misc import TapeCallbackFunction, CallbackWrapper
from neuraxle.steps.numpy import MultiplyByN, AddN
from testing_neuraxle.steps.neuraxle_test_case import NeuraxleTestCase

DATA_INPUTS = np.array(range(5))
EXPECTED_OUTPUTS = np.array(range(5, 10))
EXPECTED_PROCESSED_OUTPUTS = np.array([5.0, 6.0, 7.0, 8.0, 9.0])

tape_transform_preprocessing = TapeCallbackFunction()
tape_fit_preprocessing = TapeCallbackFunction()
tape_transform_postprocessing = TapeCallbackFunction()
tape_fit_postprocessing = TapeCallbackFunction()
tape_inverse_transform_preprocessing = TapeCallbackFunction()


@pytest.mark.parametrize('test_case', [
    NeuraxleTestCase(
        pipeline=Pipeline([
            ReversiblePreprocessingWrapper(
                preprocessing_step=CallbackWrapper(MultiplyByN(2), tape_transform_preprocessing, tape_fit_postprocessing, tape_inverse_transform_preprocessing),
                postprocessing_step=CallbackWrapper(AddN(10), tape_transform_postprocessing, tape_fit_postprocessing)
            )]
        ),
        callbacks=[tape_transform_preprocessing, tape_fit_preprocessing, tape_transform_postprocessing, tape_fit_postprocessing, tape_inverse_transform_preprocessing],
        expected_callbacks_data=[
            [DATA_INPUTS],
            [],
            [DATA_INPUTS * 2],
            [],
            [(DATA_INPUTS * 2) + 10]
        ],
        data_inputs=DATA_INPUTS,
        expected_processed_outputs=EXPECTED_PROCESSED_OUTPUTS,
        execution_mode=ExecutionMode.TRANSFORM
    ),
    NeuraxleTestCase(
        pipeline=Pipeline([
            ReversiblePreprocessingWrapper(
                preprocessing_step=CallbackWrapper(MultiplyByN(2), tape_transform_preprocessing, tape_fit_preprocessing, tape_inverse_transform_preprocessing),
                postprocessing_step=CallbackWrapper(AddN(10), tape_transform_postprocessing, tape_fit_postprocessing)
            )]
        ),
        callbacks=[tape_transform_preprocessing, tape_fit_preprocessing, tape_transform_postprocessing, tape_fit_postprocessing, tape_inverse_transform_preprocessing],
        expected_callbacks_data=[
            [DATA_INPUTS],
            [(DATA_INPUTS, EXPECTED_OUTPUTS)],
            [DATA_INPUTS * 2],
            [(DATA_INPUTS * 2, EXPECTED_OUTPUTS)],
            [(DATA_INPUTS * 2) + 10]
        ],
        data_inputs=DATA_INPUTS,
        expected_outputs=EXPECTED_OUTPUTS,
        expected_processed_outputs=EXPECTED_PROCESSED_OUTPUTS,
        execution_mode=ExecutionMode.FIT_TRANSFORM
    ),
    NeuraxleTestCase(
        pipeline=Pipeline([
            ReversiblePreprocessingWrapper(
                preprocessing_step=CallbackWrapper(MultiplyByN(2), tape_transform_preprocessing, tape_fit_preprocessing, tape_inverse_transform_preprocessing),
                postprocessing_step=CallbackWrapper(AddN(10), tape_transform_postprocessing, tape_fit_postprocessing)
            )]
        ),
        callbacks=[tape_transform_preprocessing, tape_fit_preprocessing, tape_transform_postprocessing, tape_fit_postprocessing, tape_inverse_transform_preprocessing],
        expected_callbacks_data=[
            [DATA_INPUTS],
            [(DATA_INPUTS, EXPECTED_OUTPUTS)],
            [],
            [(DATA_INPUTS * 2, EXPECTED_OUTPUTS)],
            []
        ],
        data_inputs=DATA_INPUTS,
        expected_outputs=EXPECTED_OUTPUTS,
        execution_mode=ExecutionMode.FIT
    )
])
def test_reversible_preprocessing_wrapper(test_case):
    processed_outputs = test_case.execute()

    test_case.assert_expected_processed_outputs(processed_outputs)
    test_case.assert_callback_data_is_as_expected()
