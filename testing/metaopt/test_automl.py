import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from neuraxle.hyperparams.distributions import FixedHyperparameter
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import InMemoryHyperparamsRepository, AutoML, RandomSearchHyperparameterOptimizer, \
    EarlyStoppingCallback, HyperparamsJSONRepository, EarlyStoppingRefitCallback
from neuraxle.metaopt.random import ValidationSplitWrapper, KFoldCrossValidationWrapper, average_kfold_scores
from neuraxle.pipeline import Pipeline
from neuraxle.steps.misc import FitTransformCallbackStep
from neuraxle.steps.numpy import MultiplyByN, NumpyReshape


def test_automl_with_validation_split_wrapper(tmpdir):
    # Given
    hp_repository = InMemoryHyperparamsRepository(cache_folder=str(tmpdir))
    auto_ml = AutoML(pipeline=Pipeline([
        MultiplyByN(2).set_hyperparams_space(HyperparameterSpace({
            'multiply_by': FixedHyperparameter(2)
        })),
        NumpyReshape(shape=(-1, 1)),
        linear_model.LinearRegression()
    ]), validation_technique=ValidationSplitWrapper(test_size=0.2, scoring_function=mean_squared_error),
        scoring_function=mean_squared_error, refit_trial=True,
        hyperparams_optimizer=RandomSearchHyperparameterOptimizer(), hyperparams_repository=hp_repository, n_trials=1,
        metrics={'mse': mean_squared_error}, epochs=2, callbacks=[])

    # When
    data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_outputs = data_inputs * 2
    auto_ml.fit(data_inputs=data_inputs, expected_outputs=expected_outputs)

    # Then
    p = auto_ml.get_best_model()
    outputs = p.transform(data_inputs)
    mse = mean_squared_error(expected_outputs, outputs)

    assert mse < 500


def test_automl_with_validation_split_wrapper_and_json_repository(tmpdir):
    # Given
    hp_repository = HyperparamsJSONRepository(cache_folder=str(tmpdir))
    auto_ml = AutoML(pipeline=Pipeline([
        MultiplyByN(2).set_hyperparams_space(HyperparameterSpace({
            'multiply_by': FixedHyperparameter(2)
        })),
        NumpyReshape(shape=(-1, 1)),
        linear_model.LinearRegression()
    ]), validation_technique=ValidationSplitWrapper(test_size=0.2, scoring_function=mean_squared_error),
        scoring_function=mean_squared_error, refit_trial=True,
        hyperparams_optimizer=RandomSearchHyperparameterOptimizer(), hyperparams_repository=hp_repository, n_trials=1,
        metrics={'mse': mean_squared_error}, epochs=2, callbacks=[])

    # When
    data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_outputs = data_inputs * 2
    auto_ml.fit(data_inputs=data_inputs, expected_outputs=expected_outputs)

    # Then
    p = auto_ml.get_best_model()
    outputs = p.transform(data_inputs)
    mse = mean_squared_error(expected_outputs, outputs)

    assert mse < 500


def test_automl_early_stopping_callback(tmpdir):
    # Given
    hp_repository = InMemoryHyperparamsRepository(cache_folder=str(tmpdir))
    n_epochs = 60
    auto_ml = AutoML(pipeline=Pipeline([
        FitTransformCallbackStep().set_name('callback'),
        MultiplyByN(2).set_hyperparams_space(HyperparameterSpace({
            'multiply_by': FixedHyperparameter(2)
        })),
        NumpyReshape(shape=(-1, 1)),
        linear_model.LinearRegression()
    ]), validation_technique=ValidationSplitWrapper(test_size=0.2, scoring_function=mean_squared_error),
        scoring_function=mean_squared_error, refit_trial=True,
        hyperparams_optimizer=RandomSearchHyperparameterOptimizer(), hyperparams_repository=hp_repository, n_trials=1,
        metrics={'mse': mean_squared_error}, epochs=n_epochs,
        callbacks=[EarlyStoppingCallback(n_epochs_without_improvement=3, higher_score_is_better=False)],
        refit_callbacks=[EarlyStoppingRefitCallback(n_epochs_without_improvement=3, higher_score_is_better=False)],
        refit_scoring_function=mean_squared_error, higher_score_is_better=False)

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
    auto_ml = AutoML(pipeline=Pipeline([
        MultiplyByN(2).set_hyperparams_space(HyperparameterSpace({
            'multiply_by': FixedHyperparameter(2)
        })),
        NumpyReshape(shape=(-1, 1)),
        linear_model.LinearRegression()
    ]), validation_technique=KFoldCrossValidationWrapper(
        k_fold=2,
        scoring_function=average_kfold_scores(mean_squared_error),
        split_data_container_during_fit=False,
        predict_after_fit=False
    ), scoring_function=average_kfold_scores(mean_squared_error), refit_trial=True,
        hyperparams_optimizer=RandomSearchHyperparameterOptimizer(), hyperparams_repository=hp_repository, n_trials=1,
        metrics={'mse': average_kfold_scores(mean_squared_error)}, epochs=10, callbacks=[],
        refit_scoring_function=mean_squared_error)

    data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_outputs = data_inputs * 4

    # When
    auto_ml.fit(data_inputs=data_inputs, expected_outputs=expected_outputs)

    # Then
    p = auto_ml.get_best_model()
    outputs = p.transform(data_inputs)
    mse = mean_squared_error(expected_outputs, outputs)

    assert mse < 500
