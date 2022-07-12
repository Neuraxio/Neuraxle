import numpy as np
from neuraxle.base import ExecutionContext as CX
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import Trainer
from neuraxle.metaopt.callbacks import MetricCallback
from neuraxle.metaopt.data.aggregates import Round
from neuraxle.metaopt.validation import GridExplorationSampler, ValidationSplitter
from neuraxle.steps.misc import FitTransformCallbackStep, TapeCallbackFunction
from sklearn.metrics import mean_squared_error


def test_validation_splitter_handler_methods_should_split_data(tmpdir):
    transform_callback = TapeCallbackFunction()
    fit_callback = TapeCallbackFunction()
    pipeline = FitTransformCallbackStep(
        transform_callback_function=transform_callback,
        fit_callback_function=fit_callback,
        transform_function=lambda di: di * 2
    )
    metric: MetricCallback = MetricCallback("MSE", mean_squared_error, False)
    validation_split_wrapper = Trainer(
        callbacks=[metric],
        validation_splitter=ValidationSplitter(validation_size=0.1),
        n_epochs=1,
    )

    data_inputs = np.random.randint(low=1, high=100, size=(100, 5))
    expected_outputs = np.random.randint(low=1, high=100, size=(100, 5))
    dact = DACT(di=data_inputs, eo=expected_outputs)

    round_scope: Round = Round.dummy().with_metric(metric.name).save(deep=False)
    with round_scope.with_optimizer(GridExplorationSampler(), HyperparameterSpace()).new_rvs_trial() as trial_scope:
        trained_pipeline: FitTransformCallbackStep = validation_split_wrapper.train(
            pipeline, dact, trial_scope, return_trained_pipelines=True)[0]

    predicted_outputs = trained_pipeline.predict(data_inputs)
    fit_callback = trained_pipeline.fit_callback_function
    transform_callback = trained_pipeline.transform_callback_function

    assert np.array_equal(predicted_outputs, data_inputs * 2)

    # should fit on train split
    assert np.array_equal(fit_callback.data[0][0], data_inputs[0:90])
    assert np.array_equal(fit_callback.data[0][1], expected_outputs[0:90])

    # should transform on test split
    assert np.array_equal(transform_callback.data[0], data_inputs[0:90])
    assert np.array_equal(transform_callback.data[1], data_inputs[90:])

    # should predict on all data at the end
    assert np.array_equal(transform_callback.data[2], data_inputs)

    with round_scope.last_trial() as trial_scope:
        assert trial_scope.get_avg_validation_score(metric.name) is not None
