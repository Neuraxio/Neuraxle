import pytest
import numpy as np
from sklearn.metrics import mean_squared_error
from neuraxle.base import ExecutionContext as CX

from neuraxle.hyperparams.distributions import RandInt
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.callbacks import MetricCallback
from neuraxle.metaopt.validation import KFoldCrossValidationSplitter, ValidationSplitter, RandomSearchSampler
from neuraxle.metaopt.auto_ml import AutoML
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import MultiplyByN
from neuraxle.data_container import DataContainer as DACT


def test_automl_sequence_splitter(tmpdir):
    # Setting seed for better reproducibility
    np.random.seed(68)

    # Given
    data_inputs = np.array(range(100))
    expected_outputs = np.array(range(100, 200))

    hyperparameter_space = HyperparameterSpace({
        'multiplication_1__multiply_by': RandInt(1, 3),
        'multiplication_2__multiply_by': RandInt(1, 3),
        'multiplication_3__multiply_by': RandInt(1, 3),
    })

    pipeline = Pipeline([
        ('multiplication_1', MultiplyByN()),
        ('multiplication_2', MultiplyByN()),
        ('multiplication_3', MultiplyByN())
    ]).set_hyperparams_space(hyperparameter_space)

    auto_ml = AutoML(
        pipeline=pipeline,
        hyperparams_optimizer=RandomSearchSampler(),
        validation_splitter=KFoldCrossValidationSplitter(k_fold=4),
        callbacks=[MetricCallback("MSE", mean_squared_error, False)],
    )

    # When
    auto_ml = auto_ml.handle_fit(
        DACT(data_inputs=data_inputs, expected_outputs=expected_outputs), CX(tmpdir))
    predicted_outputs = auto_ml.transform(data_inputs)

    # Then
    actual_mse = ((predicted_outputs - expected_outputs) ** 2).mean()
    assert actual_mse < 20000


def test_automl_validation_splitter(tmpdir):
    # Setting seed for reproducibility
    np.random.seed(75)
    # Given
    data_inputs = np.array(range(1000, 1020))
    expected_outputs = np.array(range(2020, 2040))

    hyperparameter_space = HyperparameterSpace({
        'multiplication_1__multiply_by': RandInt(1, 3),
        'multiplication_2__multiply_by': RandInt(1, 3),
        'multiplication_3__multiply_by': RandInt(1, 3),
    })

    pipeline = Pipeline([
        ('multiplication_1', MultiplyByN()),
        ('multiplication_2', MultiplyByN()),
        ('multiplication_3', MultiplyByN())
    ]).set_hyperparams_space(hyperparameter_space)

    hp_search = AutoML(
        pipeline=pipeline,
        validation_splitter=ValidationSplitter(validation_size=0.2),
        scoring_callback=MetricCallback("MSE", mean_squared_error, False),
        n_trials=18,
    ).with_context(CX(tmpdir))

    # When
    mse_before = ((data_inputs - expected_outputs) ** 2).mean()
    hp_search = hp_search.fit(data_inputs, expected_outputs)
    predicted_outputs = hp_search.transform(data_inputs)

    # Then
    actual_mse = ((predicted_outputs - expected_outputs) ** 2).mean()
    assert actual_mse < mse_before
