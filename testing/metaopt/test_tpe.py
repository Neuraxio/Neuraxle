import numpy as np
from sklearn.metrics import mean_squared_error

from neuraxle.hyperparams.distributions import Uniform, LogNormal, Normal, Choice, Quantized, LogUniform
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import InMemoryHyperparamsRepository, AutoML, ValidationSplitter
from neuraxle.metaopt.callbacks import MetricCallback, ScoringCallback
from neuraxle.metaopt.tpe import TreeParzenEstimatorHyperparameterSelectionStrategy
from neuraxle.pipeline import Pipeline
from neuraxle.steps.misc import FitTransformCallbackStep
from neuraxle.steps.numpy import AddN


def test_tpe_simple_uniform(tmpdir):
    # Given
    hp_repository = InMemoryHyperparamsRepository(cache_folder=str(tmpdir))
    n_epochs = 1
    n_trials = 50
    expected_output_mult = 1.5
    auto_ml = AutoML(
        pipeline=Pipeline([
            FitTransformCallbackStep().set_name('callback'),
            AddN(0.).set_hyperparams_space(HyperparameterSpace({
                'add': Uniform(-1, 3),
            })),
        ]),
        hyperparams_optimizer=TreeParzenEstimatorHyperparameterSelectionStrategy(
            number_of_initial_random_step=20,
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

    assert abs(p["AddN__add"] - expected_output_mult) < 1e-2


def test_tpe_simple_normal_truncated(tmpdir):
    # Given
    hp_repository = InMemoryHyperparamsRepository(cache_folder=str(tmpdir))
    n_epochs = 1
    n_trials = 50
    expected_output_mult = 3.5
    auto_ml = AutoML(
        pipeline=Pipeline([
            FitTransformCallbackStep().set_name('callback'),
            AddN(0.).set_hyperparams_space(HyperparameterSpace({
                'add': Normal(mean=2.0, std=2.0, hard_clip_min=-2, hard_clip_max=6),
            })),
        ]),
        hyperparams_optimizer=TreeParzenEstimatorHyperparameterSelectionStrategy(
            number_of_initial_random_step=20,
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

    assert abs(p["AddN__add"] - expected_output_mult) < 1e-1


def test_tpe_simple_log_normal(tmpdir):
    # TODO: check why new params suggestion do not converge near the expected output mult.
    # Given
    hp_repository = InMemoryHyperparamsRepository(cache_folder=str(tmpdir))
    n_epochs = 1
    n_trials = 50
    expected_output_mult = 3.5
    auto_ml = AutoML(
        pipeline=Pipeline([
            FitTransformCallbackStep().set_name('callback'),
            AddN(0.).set_hyperparams_space(HyperparameterSpace({
                'add': LogNormal(log2_space_mean=1.0, log2_space_std=0.5),
            })),
        ]),
        hyperparams_optimizer=TreeParzenEstimatorHyperparameterSelectionStrategy(
            number_of_initial_random_step=20,
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

    assert abs((p["AddN__add"] - expected_output_mult) / expected_output_mult) < 0.2


def test_tpe_simple_log_uniform(tmpdir):
    # TODO: check why new params suggestion do not converge near the expected output mult.
    # Given
    hp_repository = InMemoryHyperparamsRepository(cache_folder=str(tmpdir))
    n_epochs = 1
    n_trials = 50
    expected_output_mult = 3.5
    auto_ml = AutoML(
        pipeline=Pipeline([
            FitTransformCallbackStep().set_name('callback'),
            AddN(0.).set_hyperparams_space(HyperparameterSpace({
                'add': LogUniform(min_included=1.0, max_included=10.),
            })),
        ]),
        hyperparams_optimizer=TreeParzenEstimatorHyperparameterSelectionStrategy(
            number_of_initial_random_step=20,
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

    assert abs((p["AddN__add"] - expected_output_mult) / expected_output_mult) < 0.2

def test_tpe_simple_quantized_uniform(tmpdir):
    # Given
    hp_repository = InMemoryHyperparamsRepository(cache_folder=str(tmpdir))
    n_epochs = 1
    n_trials = 10
    expected_output_mult = 7
    auto_ml = AutoML(
        pipeline=Pipeline([
            FitTransformCallbackStep().set_name('callback'),
            AddN(0.).set_hyperparams_space(HyperparameterSpace({
                'add': Quantized(hd=Uniform(-2, 10)),
            })),
        ]),
        hyperparams_optimizer=TreeParzenEstimatorHyperparameterSelectionStrategy(
            number_of_initial_random_step=5,
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

    assert abs(p["AddN__add"] - expected_output_mult) < 1e-2

def test_tpe_simple_categorical_choice(tmpdir):
    # Given
    hp_repository = InMemoryHyperparamsRepository(cache_folder=str(tmpdir))
    n_epochs = 1
    n_trials = 20
    expected_output_mult = 3.5
    auto_ml = AutoML(
        pipeline=Pipeline([
            FitTransformCallbackStep().set_name('callback'),
            AddN(0.).set_hyperparams_space(HyperparameterSpace({
                'add': Choice(choice_list=[0, 1.5, 2, 3.5, 4, 5, 6]),
            })),
        ]),
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

    assert abs(p["AddN__add"] - expected_output_mult) < 1e-2