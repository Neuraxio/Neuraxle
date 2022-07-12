import numpy as np
import pytest

from neuraxle.base import ExecutionMode
from neuraxle.pipeline import Pipeline
from neuraxle.steps.flow import TrainOrTestOnlyWrapper, TestOnlyWrapper
from neuraxle.steps.misc import TapeCallbackFunction, FitTransformCallbackStep
from testing_neuraxle.steps.neuraxle_test_case import NeuraxleTestCase

DATA_INPUTS = np.array([1])
EXPECTED_OUTPUTS = np.array([1])

tape_transform = TapeCallbackFunction()
tape_fit = TapeCallbackFunction()


@pytest.mark.parametrize('test_case', [
    NeuraxleTestCase(
        pipeline=TrainOrTestOnlyWrapper(FitTransformCallbackStep(tape_transform, tape_fit, transform_function=lambda di: di * 2)),
        more_arguments={'set_train': True},
        callbacks=[tape_transform, tape_fit],
        expected_callbacks_data=[[DATA_INPUTS]],
        data_inputs=DATA_INPUTS,
        expected_processed_outputs=DATA_INPUTS * 2,
        execution_mode=ExecutionMode.TRANSFORM
    ),
    NeuraxleTestCase(
        pipeline=TrainOrTestOnlyWrapper(FitTransformCallbackStep(tape_transform, tape_fit, transform_function=lambda di: di * 2)),
        more_arguments={'set_train': True},
        callbacks=[tape_transform, tape_fit],
        data_inputs=DATA_INPUTS,
        expected_outputs=EXPECTED_OUTPUTS,
        expected_callbacks_data=[[DATA_INPUTS], [(DATA_INPUTS, EXPECTED_OUTPUTS)]],
        expected_processed_outputs=DATA_INPUTS * 2,
        execution_mode=ExecutionMode.FIT_TRANSFORM
    ),
    NeuraxleTestCase(
        pipeline=TrainOrTestOnlyWrapper(FitTransformCallbackStep(tape_transform, tape_fit, transform_function=lambda di: di * 2)),
        more_arguments={'set_train': True},
        callbacks=[tape_transform, tape_fit],
        data_inputs=DATA_INPUTS,
        expected_outputs=EXPECTED_OUTPUTS,
        expected_callbacks_data=[[], [(DATA_INPUTS, EXPECTED_OUTPUTS)]],
        execution_mode=ExecutionMode.FIT
    ),
    NeuraxleTestCase(
        pipeline=TrainOrTestOnlyWrapper(FitTransformCallbackStep(tape_transform, tape_fit, transform_function=lambda di: di * 2)),
        more_arguments={'set_train': False},
        callbacks=[tape_transform, tape_fit],
        expected_callbacks_data=[[], []],
        data_inputs=DATA_INPUTS,
        expected_processed_outputs=DATA_INPUTS,
        execution_mode=ExecutionMode.TRANSFORM
    ),
    NeuraxleTestCase(
        pipeline=TrainOrTestOnlyWrapper(FitTransformCallbackStep(tape_transform, tape_fit, transform_function=lambda di: di * 2)),
        more_arguments={'set_train': False},
        callbacks=[tape_transform, tape_fit],
        data_inputs=DATA_INPUTS,
        expected_outputs=EXPECTED_OUTPUTS,
        expected_callbacks_data=[[], []],
        expected_processed_outputs=DATA_INPUTS,
        execution_mode=ExecutionMode.FIT_TRANSFORM
    ),
    NeuraxleTestCase(
        pipeline=TrainOrTestOnlyWrapper(FitTransformCallbackStep(tape_transform, tape_fit, transform_function=lambda di: di * 2)),
        more_arguments={'set_train': False},
        callbacks=[tape_transform, tape_fit],
        data_inputs=DATA_INPUTS,
        expected_outputs=EXPECTED_OUTPUTS,
        expected_callbacks_data=[[], []],
        execution_mode=ExecutionMode.FIT
    ),
    NeuraxleTestCase(
        pipeline=TrainOrTestOnlyWrapper(FitTransformCallbackStep(tape_transform, tape_fit, transform_function=lambda di: di * 2), is_train_only=False),
        more_arguments={'set_train': False},
        callbacks=[tape_transform, tape_fit],
        expected_callbacks_data=[[DATA_INPUTS]],
        data_inputs=DATA_INPUTS,
        expected_processed_outputs=DATA_INPUTS * 2,
        execution_mode=ExecutionMode.TRANSFORM
    ),
    NeuraxleTestCase(
        pipeline=TrainOrTestOnlyWrapper(FitTransformCallbackStep(tape_transform, tape_fit, transform_function=lambda di: di * 2), is_train_only=False),
        more_arguments={'set_train': False},
        callbacks=[tape_transform, tape_fit],
        data_inputs=DATA_INPUTS,
        expected_outputs=EXPECTED_OUTPUTS,
        expected_callbacks_data=[[DATA_INPUTS], [(DATA_INPUTS, EXPECTED_OUTPUTS)]],
        expected_processed_outputs=DATA_INPUTS * 2,
        execution_mode=ExecutionMode.FIT_TRANSFORM
    ),
    NeuraxleTestCase(
        pipeline=TrainOrTestOnlyWrapper(FitTransformCallbackStep(tape_transform, tape_fit, transform_function=lambda di: di * 2), is_train_only=False),
        more_arguments={'set_train': False},
        callbacks=[tape_transform, tape_fit],
        data_inputs=DATA_INPUTS,
        expected_outputs=EXPECTED_OUTPUTS,
        expected_callbacks_data=[[], [(DATA_INPUTS, EXPECTED_OUTPUTS)]],
        execution_mode=ExecutionMode.FIT
    ),
    NeuraxleTestCase(
        pipeline=TrainOrTestOnlyWrapper(FitTransformCallbackStep(tape_transform, tape_fit, transform_function=lambda di: di * 2), is_train_only=False),
        more_arguments={'set_train': True},
        callbacks=[tape_transform, tape_fit],
        expected_callbacks_data=[[], []],
        data_inputs=DATA_INPUTS,
        expected_processed_outputs=DATA_INPUTS,
        execution_mode=ExecutionMode.TRANSFORM
    ),
    NeuraxleTestCase(
        pipeline=TrainOrTestOnlyWrapper(FitTransformCallbackStep(tape_transform, tape_fit, transform_function=lambda di: di * 2), is_train_only=False),
        more_arguments={'set_train': True},
        callbacks=[tape_transform, tape_fit],
        data_inputs=DATA_INPUTS,
        expected_outputs=EXPECTED_OUTPUTS,
        expected_callbacks_data=[[], []],
        expected_processed_outputs=DATA_INPUTS,
        execution_mode=ExecutionMode.FIT_TRANSFORM
    ),
    NeuraxleTestCase(
        pipeline=TrainOrTestOnlyWrapper(FitTransformCallbackStep(tape_transform, tape_fit, transform_function=lambda di: di * 2), is_train_only=False),
        more_arguments={'set_train': True},
        callbacks=[tape_transform, tape_fit],
        data_inputs=DATA_INPUTS,
        expected_outputs=EXPECTED_OUTPUTS,
        expected_callbacks_data=[[], []],
        execution_mode=ExecutionMode.FIT
    ),
    NeuraxleTestCase(
        pipeline=TestOnlyWrapper(FitTransformCallbackStep(tape_transform, tape_fit, transform_function=lambda di: di * 2)),
        more_arguments={'set_train': False},
        callbacks=[tape_transform, tape_fit],
        data_inputs=DATA_INPUTS,
        expected_outputs=EXPECTED_OUTPUTS,
        expected_callbacks_data=[[], [(DATA_INPUTS, EXPECTED_OUTPUTS)]],
        execution_mode=ExecutionMode.FIT
    )
])
def test_train_only_wrapper(test_case: NeuraxleTestCase):
    test_case.pipeline.set_train(test_case.more_arguments['set_train'])

    processed_outputs = test_case.execute()

    test_case.assert_expected_processed_outputs(processed_outputs)
    test_case.assert_callback_data_is_as_expected()
