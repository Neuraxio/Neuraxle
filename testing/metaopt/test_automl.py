import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from neuraxle.hyperparams.distributions import FixedHyperparameter
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import InMemoryHyperparamsRepository, AutoML, RandomSearchHyperparameterSelectionStrategy, \
    HyperparamsJSONRepository, kfold_cross_validation_split
from neuraxle.metaopt.callbacks import EarlyStoppingCallback, MetricCallback, EarlyStoppingRefitCallback
from neuraxle.metaopt.random import ValidationSplitWrapper, average_kfold_scores
from neuraxle.pipeline import Pipeline
from neuraxle.steps.misc import FitTransformCallbackStep
from neuraxle.steps.numpy import MultiplyByN, NumpyReshape


def test_automl_early_stopping_callback(tmpdir):
    # Given
    hp_repository = InMemoryHyperparamsRepository(cache_folder=str(tmpdir))
    n_epochs = 60
    auto_ml = AutoML(
        pipeline=Pipeline([
            FitTransformCallbackStep().set_name('callback'),
            MultiplyByN(2).set_hyperparams_space(HyperparameterSpace({
                'multiply_by': FixedHyperparameter(2)
            })),
            NumpyReshape(shape=(-1, 1)),
            linear_model.LinearRegression()
        ]),
        hyperparams_optimizer=RandomSearchHyperparameterSelectionStrategy(),
        validation_split_function=kfold_cross_validation_split(k_fold=2),
        scoring_callback=MetricCallback('mse', average_kfold_scores(mean_squared_error), higher_score_is_better=False),
        callbacks=[
            MetricCallback('mse', metric_function=average_kfold_scores(mean_squared_error), higher_score_is_better=False),
        ],
        n_trials=1,
        refit_trial=True,
        epochs=n_epochs,
        hyperparams_repository=hp_repository
    )

    # When
    data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_outputs = data_inputs * 2
    auto_ml = auto_ml.fit(data_inputs=data_inputs, expected_outputs=expected_outputs)

    # Then
    p = auto_ml.get_best_model()
    callback_step = p.get_step_by_name('callback')

    assert len(callback_step.fit_callback_function.data) < n_epochs


def test_automl_with_kfold(tmpdir):
    # Given
    hp_repository = InMemoryHyperparamsRepository(cache_folder=str(tmpdir))
    auto_ml = AutoML(
        pipeline=Pipeline([
            MultiplyByN(2).set_hyperparams_space(HyperparameterSpace({
                'multiply_by': FixedHyperparameter(2)
            })),
            NumpyReshape(shape=(-1, 1)),
            linear_model.LinearRegression()
        ]),
        validation_split_function=kfold_cross_validation_split(k_fold=2),
        hyperparams_optimizer=RandomSearchHyperparameterSelectionStrategy(),
        scoring_callback=MetricCallback('mse', average_kfold_scores(mean_squared_error), higher_score_is_better=False),
        callbacks=[
            MetricCallback('mse', metric_function=average_kfold_scores(mean_squared_error), higher_score_is_better=False),
        ],
        n_trials=1,
        epochs=10,
        refit_trial=True,
        hyperparams_repository=hp_repository
    )

    data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_outputs = data_inputs * 4

    # When
    auto_ml.fit(data_inputs=data_inputs, expected_outputs=expected_outputs)

    # Then
    p = auto_ml.get_best_model()
    outputs = p.transform(data_inputs)
    mse = mean_squared_error(expected_outputs, outputs)

    assert mse < 500
