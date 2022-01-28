import pytest
import numpy as np
from sklearn.metrics import mean_squared_error
from neuraxle.base import ExecutionContext as CX

from neuraxle.hyperparams.distributions import RandInt
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.validation import ValidationSplitWrapper
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import MultiplyByN
from neuraxle.data_container import DataContainer as DACT


@pytest.mark.skip(reason="TODO: AutoML Refactor")
def test_automl_sequential_wrapper(tmpdir):
    # Setting seed for reproducibility
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

    auto_ml = RandomSearch(
        KFoldCrossValidationWrapper().set_step(pipeline),
        hyperparams_repository=HyperparamsJSONRepository(tmpdir), n_iter=10
    )

    # When
    auto_ml: AutoMLSequentialWrapper = auto_ml.handle_fit(
        DACT(data_inputs=data_inputs, expected_outputs=expected_outputs), CX(tmpdir))
    best_model: Pipeline = auto_ml.get_best_model()
    predicted_outputs = best_model.transform(data_inputs)

    # Then
    actual_mse = ((predicted_outputs - expected_outputs) ** 2).mean()
    assert actual_mse < 20000


@pytest.mark.skip(reason="TODO: AutoML Refactor")
def test_automl_sequential_wrapper_with_validation_split_wrapper(tmpdir):
    # Setting seed for reproducibility
    np.random.seed(75)
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

    random_search = RandomSearch(
        ValidationSplitWrapper(
            pipeline,
            test_size=0.2,
            scoring_function=mean_squared_error,
            run_validation_split_in_test_mode=False
        ),
        hyperparams_repository=HyperparamsJSONRepository(tmpdir),
        higher_score_is_better=False,
        n_iter=100
    ).with_context(CX(tmpdir))

    # When
    mse_before = ((data_inputs - expected_outputs) ** 2).mean()
    random_search: AutoMLSequentialWrapper = random_search.fit(data_inputs, expected_outputs)
    best_model: Pipeline = random_search.get_best_model()
    predicted_outputs = best_model.transform(data_inputs)

    # Then
    actual_mse = ((predicted_outputs - expected_outputs) ** 2).mean()
    assert actual_mse < mse_before
