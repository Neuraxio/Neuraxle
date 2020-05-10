import numpy as np
from sklearn.metrics import mean_squared_error

from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.distributions import FixedHyperparameter, Uniform
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import InMemoryHyperparamsRepository, AutoML, ValidationSplitter
from neuraxle.metaopt.callbacks import MetricCallback, ScoringCallback
from neuraxle.pipeline import Pipeline
from neuraxle.steps.misc import FitTransformCallbackStep
from neuraxle.steps.numpy import MultiplyByN, NumpyReshape, AddN
from neuraxle.metaopt.tpe import TreeParzenEstimatorHyperparameterSelectionStrategy


def test_tpe_simple_uniform(tmpdir):
    # TODO: fix this unit test
    # Given
    hp_repository = InMemoryHyperparamsRepository(cache_folder=str(tmpdir))
    n_epochs = 1
    n_trials = 50
    auto_ml = AutoML(
        pipeline=Pipeline([
            FitTransformCallbackStep().set_name('callback'),
            AddN(0.).set_hyperparams_space((HyperparameterSpace({
                'basic_dist_value': Uniform(-1, 3),
            }))),
        ]),
        hyperparams_optimizer=TreeParzenEstimatorHyperparameterSelectionStrategy(number_of_initial_random_step=20,
                                                                                 quantile_threshold=0.3,
                                                                                 number_good_trials_max_cap=25,
                                                                                 number_possible_hyperparams_candidates=100,
                                                                                 prior_weight=0.,
                                                                                 use_linear_forgetting_weights=False,
                                                                                 number_recent_trial_at_full_weights=25),
        validation_splitter=ValidationSplitter(0.10),
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
    data_inputs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    expected_outputs = 1.5 * np.ones_like(data_inputs)
    auto_ml = auto_ml.fit(data_inputs=data_inputs, expected_outputs=expected_outputs)

    # Then
    p = auto_ml.get_best_model().get_hyperparams()

    assert (p["AddN__basic_dist_value"] - 1.5) < 1e-4
