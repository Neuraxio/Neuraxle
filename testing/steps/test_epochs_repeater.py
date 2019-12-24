import numpy as np
import pytest

from neuraxle.base import ExecutionMode
from neuraxle.pipeline import Pipeline
from neuraxle.steps.data import EpochRepeater
from neuraxle.steps.misc import TapeCallbackFunction, FitTransformCallbackStep
from testing.steps.neuraxle_test_case import NeuraxleTestCase

DATA_INPUTS = np.array(range(10))
EXPECTED_OUTPUTS = np.array(range(10, 20))

callback_fit = TapeCallbackFunction()
callback_transform = TapeCallbackFunction()
EPOCHS = 2


@pytest.mark.parametrize("test_case", [
    NeuraxleTestCase(
        pipeline=Pipeline([
            EpochRepeater(FitTransformCallbackStep(callback_transform, callback_fit), epochs=EPOCHS)
        ]),
        callbacks=[callback_fit, callback_transform],
        expected_callbacks_data=[
            [(DATA_INPUTS, EXPECTED_OUTPUTS), (DATA_INPUTS, EXPECTED_OUTPUTS)],
            [DATA_INPUTS]
        ],
        data_inputs=DATA_INPUTS,
        expected_outputs=EXPECTED_OUTPUTS,
        expected_processed_outputs=DATA_INPUTS,
        execution_mode=ExecutionMode.FIT_TRANSFORM
    ),
    NeuraxleTestCase(
        pipeline=Pipeline([
            EpochRepeater(FitTransformCallbackStep(callback_transform, callback_fit), epochs=EPOCHS)
        ]),
        callbacks=[callback_fit, callback_transform],
        expected_callbacks_data=[
            [],
            [DATA_INPUTS]
        ],
        data_inputs=DATA_INPUTS,
        expected_outputs=EXPECTED_OUTPUTS,
        expected_processed_outputs=DATA_INPUTS,
        execution_mode=ExecutionMode.TRANSFORM
    ),
    NeuraxleTestCase(
        pipeline=Pipeline([
            EpochRepeater(FitTransformCallbackStep(callback_transform, callback_fit), epochs=EPOCHS, fit_only=False)
        ]),
        callbacks=[callback_fit, callback_transform],
        expected_callbacks_data=[
            [],
            [DATA_INPUTS]
        ],
        data_inputs=DATA_INPUTS,
        expected_outputs=EXPECTED_OUTPUTS,
        expected_processed_outputs=DATA_INPUTS,
        execution_mode=ExecutionMode.TRANSFORM
    ),
    NeuraxleTestCase(
        pipeline=Pipeline([
            EpochRepeater(FitTransformCallbackStep(callback_transform, callback_fit), epochs=EPOCHS)
        ]),
        callbacks=[callback_fit, callback_transform],
        expected_callbacks_data=[
            [(DATA_INPUTS, EXPECTED_OUTPUTS), (DATA_INPUTS, EXPECTED_OUTPUTS)],
            []
        ],
        data_inputs=DATA_INPUTS,
        expected_outputs=EXPECTED_OUTPUTS,
        execution_mode=ExecutionMode.FIT
    )
])
def test_epoch_repeater(test_case):
    processed_outputs = test_case.execute()

    test_case.assert_expected_processed_outputs(processed_outputs)
    test_case.assert_callback_data_is_as_expected()
