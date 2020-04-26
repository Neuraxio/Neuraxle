import numpy as np
from sklearn.metrics import mean_squared_error

from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.distributions import FixedHyperparameter, Uniform
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import InMemoryHyperparamsRepository, AutoML,, ValidationSplitter
from neuraxle.metaopt.callbacks import MetricCallback, ScoringCallback
from neuraxle.pipeline import Pipeline
from neuraxle.steps.misc import FitTransformCallbackStep
from neuraxle.steps.numpy import MultiplyByN, NumpyReshape, AddN
from neuraxle.metaopt.tpe import  import TreeParzenEstimatorHyperparameterSelectionStrategy


def test_automl_early_stopping_callback(tmpdir):
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
        hyperparams_optimizer=TreeParzenEstimatorHyperparameterSelectionStrategy(),
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
    expected_outputs = 1.5
    auto_ml = auto_ml.fit(data_inputs=data_inputs, expected_outputs=expected_outputs)

    # Then
    p = auto_ml.get_best_model().get_hyperparams()

    assert (p["basic_dist_value"] - 1.5) < 1e-4
