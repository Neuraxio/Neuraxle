import numpy as np

from neuraxle.hyperparams.distributions import RandInt
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import HyperparamsJSONRepository, AutoMLSequentialWrapper, RandomSearchBaseAutoMLStrategy
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import MultiplyByN


def test_automl_sequential_wrapper(tmpdir):
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
    ], cache_folder=tmpdir).set_hyperparams_space(hyperparameter_space)

    auto_ml = AutoMLSequentialWrapper(
        auto_ml_strategy=RandomSearchBaseAutoMLStrategy(),
        step=pipeline,
        hyperparams_repository=HyperparamsJSONRepository(tmpdir),
        n_iters=100
    )

    # When
    mse_before = ((data_inputs - expected_outputs) ** 2).mean()
    auto_ml: AutoMLSequentialWrapper = auto_ml.fit(data_inputs, expected_outputs)
    best_model: Pipeline = auto_ml.get_best_model()
    predicted_outputs = best_model.transform(data_inputs)

    # Then
    actual_mse = ((predicted_outputs - expected_outputs) ** 2).mean()
    assert actual_mse < mse_before
