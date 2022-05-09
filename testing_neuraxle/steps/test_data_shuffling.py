import numpy as np

from neuraxle.pipeline import Pipeline
from neuraxle.steps.data import DataShuffler
from neuraxle.steps.misc import TapeCallbackFunction, FitTransformCallbackStep


def test_data_shuffling_should_shuffle_data_inputs_and_expected_outputs():
    callback_fit = TapeCallbackFunction()
    callback_transform = TapeCallbackFunction()
    data_shuffler = Pipeline([
        DataShuffler(seed=42, increment_seed_after_each_fit=True),
        FitTransformCallbackStep(callback_transform, callback_fit)
    ])
    data_inputs = np.array(range(10))
    expected_outputs = np.array(range(10, 20))

    outputs = data_shuffler.fit_transform(data_inputs, expected_outputs)

    assert not np.array_equal(outputs, data_inputs)
    assert not np.array_equal(callback_fit.data[0][0], data_inputs)
    assert not np.array_equal(callback_fit.data[0][1], expected_outputs)
    assert not np.array_equal(callback_transform.data, data_inputs)
