import numpy as np

from neuraxle.base import ExecutionMode, ExecutionContext, DEFAULT_CACHE_FOLDER
from neuraxle.data_container import DataContainer
from neuraxle.metaopt.random import ValidationSplitWrapper, RandomSearch
from neuraxle.steps.misc import FitTransformCallbackStep, TapeCallbackFunction


def test_validation_split_wrapper_should_split_data():
    transform_callback = TapeCallbackFunction()
    fit_callback = TapeCallbackFunction()
    random_search = RandomSearch(ValidationSplitWrapper(
        FitTransformCallbackStep(
            transform_callback_function=transform_callback,
            fit_callback_function=fit_callback,
            transform_function=lambda di: di * 2
        ),
        test_size=0.1
    ))
    data_inputs = np.random.randint(1, 100, (100, 5))
    expected_outputs = np.random.randint(1, 100, (100, 5))

    random_search, outputs = random_search.fit_transform(data_inputs, expected_outputs)

    assert np.array_equal(outputs, data_inputs * 2)

    # should fit on train split
    assert np.array_equal(fit_callback.data[0][0], data_inputs[0:90])
    assert np.array_equal(fit_callback.data[0][1], expected_outputs[0:90])

    # should transform on test split
    assert np.array_equal(transform_callback.data[0], data_inputs[0:90])
    assert np.array_equal(transform_callback.data[1], data_inputs[:10])

    # should transform on all data at the end
    assert np.array_equal(transform_callback.data[2], data_inputs)


def test_validation_split_wrapper_handle_methods_should_split_data():
    transform_callback = TapeCallbackFunction()
    fit_callback = TapeCallbackFunction()
    validation_split_wrapper = ValidationSplitWrapper(
        FitTransformCallbackStep(
            transform_callback_function=transform_callback,
            fit_callback_function=fit_callback,
            transform_function=lambda di: di * 2
        ),
        test_size=0.1
    )
    data_inputs = np.random.randint(1, 100, (100, 5))
    expected_outputs = np.random.randint(1, 100, (100, 5))

    validation_split_wrapper, outputs = validation_split_wrapper.handle_fit_transform(
        DataContainer(list(range(len(data_inputs))), data_inputs, expected_outputs),
        ExecutionContext.create_from_root(validation_split_wrapper, ExecutionMode.FIT_TRANSFORM, DEFAULT_CACHE_FOLDER)
    )

    assert np.array_equal(outputs.data_inputs, data_inputs * 2)

    # should fit on train split
    assert np.array_equal(fit_callback.data[0][0], data_inputs[0:90])
    assert np.array_equal(fit_callback.data[0][1], expected_outputs[0:90])

    # should transform on test split
    assert np.array_equal(transform_callback.data[0], data_inputs[0:90])
    assert np.array_equal(transform_callback.data[1], data_inputs[:10])

    # should transform on all data at the end
    assert np.array_equal(transform_callback.data[2], data_inputs)
