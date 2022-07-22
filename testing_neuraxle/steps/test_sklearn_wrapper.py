import json

import numpy as np
import pytest
from neuraxle.base import Identity
from neuraxle.hyperparams.distributions import RandInt, Uniform
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace)
from neuraxle.metaopt.auto_ml import AutoML, RandomSearchSampler
from neuraxle.metaopt.callbacks import ScoringCallback
from neuraxle.metaopt.repositories.json import HyperparamsOnDiskRepository
from neuraxle.metaopt.validation import KFoldCrossValidationSplitter
from neuraxle.pipeline import Pipeline
from neuraxle.steps.data import DataShuffler
from neuraxle.steps.flow import TrainOnlyWrapper
from neuraxle.steps.sklearn import SKLearnWrapper
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, SGDClassifier, SGDRegressor
from sklearn.metrics import median_absolute_error


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


def test_sklearn_wrapper_transform_partial_fit_with_predict():
    model = SKLearnWrapper(SGDRegressor(learning_rate='adaptive', eta0=0.05), use_partial_fit=True)
    p = Pipeline([TrainOnlyWrapper(DataShuffler()), model])
    data_inputs = np.expand_dims(np.array(list(range(10))), axis=-1) / 10
    expected_outputs = np.ravel(np.expand_dims(np.array(list(range(10, 20))), axis=-1)) / 10

    for _ in range(30):
        p = p.fit(data_inputs, expected_outputs)
    outputs = p.predict(data_inputs)

    assert all([np.isclose(a, b, atol=0.1) for a, b in zip(expected_outputs, outputs)])


def test_sklearn_wrapper_transform_partial_fit_classifier():
    data_inputs = np.array([[0, 1], [0, 0], [3, -2], [-1, 1], [-2, 1], [2, 0], [2, -1], [4, -2], [-3, 1], [-1, 0]])
    expected_outputs = np.ravel(np.expand_dims(data_inputs[:, 0] + 2 * data_inputs[:, 1] + 1, axis=-1))
    data_inputs = data_inputs / (4 + 1)
    classes = np.array([0, 1, 2, 3])
    model = SKLearnWrapper(
        SGDClassifier(learning_rate='adaptive', eta0=0.05),
        use_partial_fit=True,
        partial_fit_kwargs={'classes': classes}
    )
    p = Pipeline([TrainOnlyWrapper(DataShuffler()), model])

    for _ in range(30):
        p = p.fit(data_inputs, expected_outputs)
    outputs = p.predict(data_inputs)

    assert outputs.shape == (10,)
    assert len(set(outputs) - set(classes)) == 0


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


# With AutoML loop

def _test_within_auto_ml_loop(tmpdir, pipeline):
    X_train = np.random.random((25, 50)).astype(np.float32)
    Y_train = np.random.random((25,)).astype(np.float32)

    validation_splitter = KFoldCrossValidationSplitter(3)
    scoring_callback = ScoringCallback(
        median_absolute_error, higher_score_is_better=False)

    auto_ml = AutoML(
        pipeline=pipeline,
        hyperparams_optimizer=RandomSearchSampler(),
        validation_splitter=validation_splitter,
        scoring_callback=scoring_callback,
        n_trials=2,
        epochs=1,
        hyperparams_repository=HyperparamsOnDiskRepository(cache_folder=tmpdir),
        refit_best_trial=True,
        continue_loop_on_error=False)

    auto_ml.fit(X_train, Y_train)


@pytest.mark.skip(reason="AutoML loop refactor")
def test_automl_sklearn(tmpdir):
    grad_boost = SKLearnWrapper(GradientBoostingRegressor())
    _test_within_auto_ml_loop(tmpdir, grad_boost)


@pytest.mark.skip(reason="AutoML loop refactor")
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
