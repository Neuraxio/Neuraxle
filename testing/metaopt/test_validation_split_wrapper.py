import numpy as np
from neuraxle.base import ExecutionContext as CX
from neuraxle.data_container import DataContainer as DACT
from neuraxle.metaopt.auto_ml import AutoML
from neuraxle.metaopt.callbacks import MetricCallback
from neuraxle.metaopt.validation import ValidationSplitter
from neuraxle.steps.misc import FitTransformCallbackStep, TapeCallbackFunction
from sklearn.metrics import mean_squared_error


def test_validation_splitter_handler_methods_should_split_data(tmpdir):
    transform_callback = TapeCallbackFunction()
    fit_callback = TapeCallbackFunction()
    validation_split_wrapper = AutoML(
        pipeline=FitTransformCallbackStep(
            transform_callback_function=transform_callback,
            fit_callback_function=fit_callback,
            transform_function=lambda di: di * 2
        ),
        validation_splitter=ValidationSplitter(validation_size=0.1),
        scoring_callback=MetricCallback("MSE", mean_squared_error, False),
        n_trials=18,
    ).with_context(CX(tmpdir))

    data_inputs = np.random.randint(low=1, high=100, size=(100, 5))
    expected_outputs = np.random.randint(low=1, high=100, size=(100, 5))

    validation_split_wrapper, outputs = validation_split_wrapper.handle_fit_transform(
        DACT(di=data_inputs, eo=expected_outputs), CX(tmpdir))

    assert np.array_equal(outputs.data_inputs, data_inputs * 2)

    # should fit on train split
    assert np.array_equal(fit_callback.data[0][0], data_inputs[0:90])
    assert np.array_equal(fit_callback.data[0][1], expected_outputs[0:90])

    # should transform on test split
    assert np.array_equal(transform_callback.data[0], data_inputs[0:90])
    assert np.array_equal(transform_callback.data[1], data_inputs[90:])

    # should transform on all data at the end
    assert np.array_equal(transform_callback.data[2], data_inputs)

    assert validation_split_wrapper.scores_train is not None
    assert validation_split_wrapper.scores_validation is not None
    assert validation_split_wrapper.scores_train_mean is not None
    assert validation_split_wrapper.scores_validation_mean is not None
    assert validation_split_wrapper.scores_train_std is not None
    assert validation_split_wrapper.scores_validation_std is not None
