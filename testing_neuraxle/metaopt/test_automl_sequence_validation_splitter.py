from typing import List

import numpy as np
from neuraxle.base import ExecutionContext as CX
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.distributions import RandInt
from neuraxle.hyperparams.space import FlatDict, HyperparameterSpace
from neuraxle.metaopt.auto_ml import AutoML
from neuraxle.metaopt.callbacks import MetricCallback
from neuraxle.metaopt.data.aggregates import Round
from neuraxle.metaopt.data.vanilla import AutoMLContext, ScopedLocation
from neuraxle.metaopt.validation import (GridExplorationSampler,
                                         KFoldCrossValidationSplitter,
                                         RandomSearchSampler,
                                         ValidationSplitter)
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import MultiplyByN
from sklearn.metrics import mean_squared_error


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
    cx = AutoMLContext.from_context()
    data_inputs = np.array(range(1000, 1020))
    expected_outputs = np.array(range(2020, 2040))
    hyperparameter_space = HyperparameterSpace({
        'multiplication_1__multiply_by': RandInt(1, 3),
        'multiplication_2__multiply_by': RandInt(1, 3),
    })
    pipeline = Pipeline([
        ('multiplication_1', MultiplyByN()),
        ('multiplication_2', MultiplyByN()),
    ]).set_hyperparams_space(hyperparameter_space)

    hp_search = AutoML(
        pipeline=pipeline,
        validation_splitter=ValidationSplitter(validation_size=0.2),
        scoring_callback=MetricCallback("MSE", mean_squared_error, False),
        hyperparams_optimizer=GridExplorationSampler(9),
        n_trials=8,
    ).with_context(cx)

    # When
    hp_search = hp_search.fit(data_inputs, expected_outputs)
    predicted_outputs = hp_search.transform(data_inputs)

    # Then
    optimal_mse = mean_squared_error(expected_outputs, data_inputs * 2)
    actual_mse = mean_squared_error(expected_outputs, predicted_outputs)
    assert actual_mse == optimal_mse


def test_grid_exploration_sampler_can_try_everything():
    hp_space = HyperparameterSpace({
        'a': RandInt(1, 3),
        'b': RandInt(1, 3),
        'c': RandInt(1, 3),
    })
    max_trials = 3 * 3 * 3
    ges = GridExplorationSampler(max_trials)
    _round: Round = Round.from_context(AutoMLContext.from_context(loc=ScopedLocation.default(0)))
    _round.with_optimizer(ges, hp_space)

    for _ in range(max_trials):
        with _round.new_rvs_trial():
            pass

    trials_hps: List[FlatDict] = _round.report.get_all_hyperparams(as_flat=True)
    unique_trials = set([tuple(r.items()) for r in trials_hps])
    assert len(unique_trials) == max_trials
