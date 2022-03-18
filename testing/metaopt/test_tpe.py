import os
import random
from typing import List

import numpy as np
import pytest
from joblib import Parallel, delayed
from neuraxle.base import ExecutionContext as CX
from neuraxle.hyperparams.distributions import (Choice, HyperparameterDistribution, LogNormal, LogUniform,
                                                Normal, Quantized, RandInt, Uniform)
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import (AutoML, BaseHyperparameterOptimizer,
                                      ControlledAutoML)
from neuraxle.metaopt.callbacks import ScoringCallback
from neuraxle.metaopt.data.aggregates import Round
from neuraxle.metaopt.data.vanilla import ScopedLocation, VanillaHyperparamsRepository
from neuraxle.metaopt.hyperopt.tpe import TreeParzenEstimator
from neuraxle.metaopt.validation import (GridExplorationSampler,
                                         ValidationSplitter)
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import AddN
from sklearn.metrics import mean_squared_error


N_TRIALS = 40


def _avg(l: List):
    if len(l) == 0:
        return None
    return sum(l) / len(l)


# @pytest.mark.parametrize("reduce_func", [min, _avg])
@pytest.mark.parametrize("add_range", [
    LogNormal(log2_space_mean=1.0, log2_space_std=0.5, hard_clip_min=0, hard_clip_max=6),
    Choice(choice_list=[0, 1.5, 2, 3.5, 4, 5, 6]),
    LogUniform(min_included=1.0, max_included=5.0),
    Normal(mean=3.0, std=2.0, hard_clip_min=0, hard_clip_max=6),
    Quantized(Uniform(0, 10)),
    Uniform(0, 6),
])
def test_tpe(tmpdir, add_range: HyperparameterDistribution, reduce_func=_avg):
    np.random.seed(42)
    random.seed(42)

    # Given
    pipeline = Pipeline([
        AddN(0.).set_hyperparams_space(HyperparameterSpace({'add': add_range})),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({'add': RandInt(1, 2)})),
    ])
    half_trials = int(N_TRIALS / 2)
    tpe: BaseHyperparameterOptimizer = TreeParzenEstimator(
        number_of_initial_random_step=half_trials,
        quantile_threshold=0.3,
        number_good_trials_max_cap=25,
        number_possible_hyperparams_candidates=100,
        use_linear_forgetting_weights=False,
        number_recent_trials_at_full_weights=25
    )
    grid: BaseHyperparameterOptimizer = GridExplorationSampler(N_TRIALS)

    grid_scores = _score_meta_optimizer(grid, pipeline, tmpdir)
    tpe_scores = _score_meta_optimizer(tpe, pipeline, tmpdir)

    mean_compared_grid_score = reduce_func(grid_scores[half_trials:])
    mean_compared_tpe_score = reduce_func(tpe_scores[half_trials:])

    assert mean_compared_tpe_score <= mean_compared_grid_score


def _score_meta_optimizer(meta_optimizer: BaseHyperparameterOptimizer, pipeline, tmpdir):
    expected_output_mult = 3.5
    optim_scores = _score_trials(
        expected_output_mult,
        pipeline,
        meta_optimizer,
        os.path.join(tmpdir, meta_optimizer.__class__.__name__)
    )
    return optim_scores


def _score_trials(
    expected_output_mult,
    pipeline,
    hyperparams_optimizer: BaseHyperparameterOptimizer,
    tmpdir: str
):
    hp_repository: VanillaHyperparamsRepository = VanillaHyperparamsRepository(str(tmpdir))
    n_epochs = 1
    auto_ml: ControlledAutoML = AutoML(
        pipeline=pipeline,
        hyperparams_optimizer=hyperparams_optimizer,
        validation_splitter=ValidationSplitter(0.5),
        scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False, name='mse'),
        n_trials=N_TRIALS,
        refit_best_trial=True,
        epochs=n_epochs,
        hyperparams_repository=hp_repository,
        continue_loop_on_error=False
    )

    # When
    data_inputs = np.array([0, 0])
    # Fitting and optimizing a multiplier problem with a model make to add...
    expected_outputs = expected_output_mult * np.ones_like(data_inputs)
    auto_ml.fit(data_inputs=data_inputs, expected_outputs=expected_outputs)

    # Then
    cx = auto_ml.get_automl_context(CX()).with_loc(ScopedLocation.default(-1))
    trials: Round = Round.from_context(cx)
    #  hp_repository.load(ScopedLocation.default(-1)).filter(TrialStatus.SUCCESS)
    trials_validation_scores: List[float] = [t.get_avg_validation_score() for t in trials]
    return trials_validation_scores
