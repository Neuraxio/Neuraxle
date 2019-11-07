import numpy as np

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
    assert np.array_equal(transform_callback.data[0], data_inputs[:10])

    # should transform on all data at the end
    assert np.array_equal(transform_callback.data[1], data_inputs)
