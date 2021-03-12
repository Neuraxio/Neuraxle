import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from neuraxle.base import Identity
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.steps.sklearn import SKLearnWrapper


def test_sklearn_wrapper_with_an_invalid_step():
    with pytest.raises(ValueError):
        SKLearnWrapper(Identity())


def test_sklearn_wrapper_fit_transform_with_predict():
    p = SKLearnWrapper(LinearRegression())
    data_inputs = np.expand_dims(np.array(list(range(10))), axis=-1)
    expected_outputs = np.expand_dims(np.array(list(range(10, 20))), axis=-1)

    p, outputs = p.fit_transform(data_inputs, expected_outputs)

    assert np.array_equal(outputs, expected_outputs)


def test_sklearn_wrapper_transform_with_predict():
    p = SKLearnWrapper(LinearRegression())
    data_inputs = np.expand_dims(np.array(list(range(10))), axis=-1)
    expected_outputs = np.expand_dims(np.array(list(range(10, 20))), axis=-1)

    p = p.fit(data_inputs, expected_outputs)
    outputs = p.transform(data_inputs)

    assert np.array_equal(outputs, expected_outputs)


def test_sklearn_wrapper_fit_transform_with_transform():
    n_components = 2
    p = SKLearnWrapper(PCA(n_components=n_components))
    dim1 = 10
    dim2 = 10
    data_inputs, expected_outputs = _create_data_source((dim1, dim2))

    p, outputs = p.fit_transform(data_inputs, expected_outputs)

    assert outputs.shape == (dim1, n_components)


def test_sklearn_wrapper_set_hyperparams():
    p = SKLearnWrapper(PCA())
    p.set_hyperparams(HyperparameterSamples({
        'n_components': 2
    }))

    assert p.wrapped_sklearn_predictor.n_components == 2


def test_sklearn_wrapper_update_hyperparams():
    p = SKLearnWrapper(PCA())
    p.set_hyperparams(HyperparameterSamples({
        'n_components': 2,
        'svd_solver': 'full'
    }))
    p.update_hyperparams(HyperparameterSamples({
        'n_components': 4
    }))

    assert p.wrapped_sklearn_predictor.n_components == 4
    assert p.wrapped_sklearn_predictor.svd_solver == 'full'


def _create_data_source(shape):
    data_inputs = np.random.random(shape).astype(np.float32)
    expected_outputs = np.random.random(shape).astype(np.float32)
    return data_inputs, expected_outputs

#
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.metrics import median_absolute_error

from neuraxle.hyperparams.distributions import RandInt, Uniform
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace
from neuraxle.metaopt.auto_ml import KFoldCrossValidationSplitter, AutoML, RandomSearchHyperparameterSelectionStrategy, \
    HyperparamsJSONRepository
from neuraxle.metaopt.callbacks import ScoringCallback

def _test_within_auto_ml_loop(tmpdir, pipeline):
    X_train = np.random.random((25,50)).astype(np.float32)
    Y_train = np.random.random((25,)).astype(np.float32)

    validation_splitter = KFoldCrossValidationSplitter(3)
    scoring_callback = ScoringCallback(
        median_absolute_error, higher_score_is_better=False)

    auto_ml = AutoML(
        pipeline=pipeline,
        hyperparams_optimizer=RandomSearchHyperparameterSelectionStrategy(),
        validation_splitter=validation_splitter,
        scoring_callback=scoring_callback,
        n_trials=10,
        epochs=1,
        hyperparams_repository=HyperparamsJSONRepository(
            cache_folder="cache"),
        refit_trial=True,
        continue_loop_on_error=False)

    auto_ml.fit(X_train, Y_train)

def test_automl_sklearn(tmpdir):
    grad_boost = SKLearnWrapper(GradientBoostingRegressor())
    _test_within_auto_ml_loop(tmpdir, grad_boost)

def test_automl_sklearn_model_with_base_estimator(tmpdir):
    grad_boost = GradientBoostingRegressor()
    bagged_regressor = BaggingRegressor(
        grad_boost, random_state=5, n_jobs=-1)

    wrapped_bagged_regressor = SKLearnWrapper(
        bagged_regressor,
        HyperparameterSpace({
            "n_estimators": RandInt(10, 100),
            "max_features": Uniform(0.6, 1.0)}),
            #  return_all_sklearn_default_params_on_get=True
        )
    _test_within_auto_ml_loop(tmpdir, wrapped_bagged_regressor)

