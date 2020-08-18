import numpy as np
import pytest
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error

from neuraxle.hyperparams.distributions import Uniform, LogNormal, Normal, Choice, Quantized, LogUniform
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import InMemoryHyperparamsRepository, AutoML, ValidationSplitter, \
    RandomSearchHyperparameterSelectionStrategy
from neuraxle.metaopt.callbacks import MetricCallback, ScoringCallback
from neuraxle.metaopt.tpe import TreeParzenEstimatorHyperparameterSelectionStrategy
from neuraxle.pipeline import Pipeline
from neuraxle.steps.misc import FitTransformCallbackStep
from neuraxle.steps.numpy import AddN
import os


@pytest.mark.parametrize("expected_output_mult, pipeline", [
    (3.5, Pipeline([
        FitTransformCallbackStep().set_name('callback'),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': Choice(choice_list=[0, 1.5, 2, 3.5, 4, 5, 6]),
        })),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': Choice(choice_list=[0, 1.5, 2, 3.5, 4, 5, 6]),
        }))
    ])),
    (7.0, Pipeline([
        FitTransformCallbackStep().set_name('callback'),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': Quantized(hd=Uniform(-2, 10)),
        })),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': Quantized(hd=Uniform(-2, 10)),
        }))
    ])),
    (3.5, Pipeline([
        FitTransformCallbackStep().set_name('callback'),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': LogUniform(min_included=1.0, max_included=10.),
        })),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': LogUniform(min_included=1.0, max_included=10.),
        }))
    ])),
    (3.5, Pipeline([
        FitTransformCallbackStep().set_name('callback'),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': LogNormal(log2_space_mean=1.0, log2_space_std=0.5),
        })),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': LogNormal(log2_space_mean=1.0, log2_space_std=0.5),
        }))
    ])),
    (3.5, Pipeline([
        FitTransformCallbackStep().set_name('callback'),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': Normal(mean=2.0, std=2.0, hard_clip_min=-2, hard_clip_max=6),
        })),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': Normal(mean=2.0, std=2.0, hard_clip_min=-2, hard_clip_max=6),
        })),
    ])),
    (1.5, Pipeline([
        FitTransformCallbackStep().set_name('callback'),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': Uniform(-1, 3),
        })),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({
            'add': Uniform(-1, 3),
        }))
    ]))
])
def test_tpe(expected_output_mult, pipeline, tmpdir):
    # Given
    tpe_scores = Parallel(n_jobs=-2)(
        delayed(_test_tpe_score)(expected_output_mult, pipeline, os.path.join(tmpdir, 'tpe', str(i)))
        for i in range(20)
    )
    random_scores = Parallel(n_jobs=-2)(
        delayed(_test_random_score)(expected_output_mult, pipeline, os.path.join(tmpdir, 'random', str(i)))
        for i in range(20)
    )

    average_tpe_scores = (sum(tpe_scores) / len(tpe_scores))
    average_random_scores = (sum(random_scores) / len(random_scores))
    assert average_tpe_scores < average_random_scores



def _test_tpe_score(expected_output_mult, pipeline, tmpdir):
    hp_repository = InMemoryHyperparamsRepository(cache_folder=str(tmpdir))
    n_epochs = 1
    n_trials = 20
    auto_ml = AutoML(
        pipeline=pipeline,
        hyperparams_optimizer=TreeParzenEstimatorHyperparameterSelectionStrategy(
            number_of_initial_random_step=10,
            quantile_threshold=0.3,
            number_good_trials_max_cap=25,
            number_possible_hyperparams_candidates=100,
            prior_weight=0.,
            use_linear_forgetting_weights=False,
            number_recent_trial_at_full_weights=25
        ),
        validation_splitter=ValidationSplitter(0.5),
        scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
        callbacks=[
            MetricCallback('mse', metric_function=mean_squared_error, higher_score_is_better=False),
        ],
        n_trials=n_trials,
        refit_trial=True,
        epochs=n_epochs,
        hyperparams_repository=hp_repository
    )
    # When
    data_inputs = np.array([0, 0])
    expected_outputs = expected_output_mult * np.ones_like(data_inputs)
    auto_ml = auto_ml.fit(data_inputs=data_inputs, expected_outputs=expected_outputs)
    # Then
    p = auto_ml.get_best_model().get_hyperparams()

    return abs(p["AddN__add"] - expected_output_mult)

def _test_random_score(expected_output_mult, pipeline, tmpdir):
    hp_repository = InMemoryHyperparamsRepository(cache_folder=str(tmpdir))
    n_epochs = 1
    n_trials = 20
    auto_ml = AutoML(
        pipeline=pipeline,
        hyperparams_optimizer=RandomSearchHyperparameterSelectionStrategy(),
        validation_splitter=ValidationSplitter(0.5),
        scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
        callbacks=[
            MetricCallback('mse', metric_function=mean_squared_error, higher_score_is_better=False),
        ],
        n_trials=n_trials,
        refit_trial=True,
        epochs=n_epochs,
        hyperparams_repository=hp_repository
    )
    # When
    data_inputs = np.array([0, 0])
    expected_outputs = expected_output_mult * np.ones_like(data_inputs)
    auto_ml = auto_ml.fit(data_inputs=data_inputs, expected_outputs=expected_outputs)
    # Then
    p = auto_ml.get_best_model().get_hyperparams()

    return abs(p["AddN__add"] - expected_output_mult)
