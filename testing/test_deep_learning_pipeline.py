import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

from neuraxle.api import DeepLearningPipeline
from neuraxle.hyperparams.distributions import RandInt, LogUniform, Boolean
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import NumpyTranspose
from neuraxle.steps.sklearn import SKLearnWrapper
from neuraxle.union import AddFeatures, ModelStacking

N_EPOCHS = 5


def test_deep_learning_pipeline():
    # Given
    boston = load_boston()
    data_inputs, expected_outputs = shuffle(boston.data, boston.target, random_state=13)
    expected_outputs = expected_outputs.astype(np.float32)
    data_inputs = data_inputs.astype(np.float32)

    pipeline = Pipeline([
        AddFeatures([
            SKLearnWrapper(
                PCA(n_components=2),
                HyperparameterSpace({"n_components": RandInt(1, 3)})
            ),
            SKLearnWrapper(
                FastICA(n_components=2),
                HyperparameterSpace({"n_components": RandInt(1, 3)})
            ),
        ]),
        ModelStacking([
            SKLearnWrapper(
                GradientBoostingRegressor(),
                HyperparameterSpace({
                    "n_estimators": RandInt(50, 600), "max_depth": RandInt(1, 10),
                    "learning_rate": LogUniform(0.07, 0.7)
                })
            ),
            SKLearnWrapper(
                KMeans(n_clusters=7),
                HyperparameterSpace({"n_clusters": RandInt(5, 10)})
            ),
        ],
            joiner=NumpyTranspose(),
            judge=SKLearnWrapper(
                Ridge(),
                HyperparameterSpace({"alpha": LogUniform(0.7, 1.4), "fit_intercept": Boolean()})
            ),
        )
    ])

    p = DeepLearningPipeline(
        pipeline,
        validation_size=0.1,
        batch_size=32,
        batch_metrics={'mse': to_numpy_metric_wrapper(mean_squared_error)},
        shuffle_in_each_epoch_at_train=True,
        n_epochs=N_EPOCHS,
        epochs_metrics={'mse': to_numpy_metric_wrapper(mean_squared_error)},
        scoring_function=to_numpy_metric_wrapper(mean_squared_error),
    )

    # When
    p, outputs = p.fit_transform(data_inputs, expected_outputs)

    # Then
    batch_mse_train = p.get_step_by_name('batch_metrics').get_metric_train('mse')
    epoch_mse_train = p.get_step_by_name('epoch_metrics').get_metric_train('mse')

    batch_mse_validation = p.get_step_by_name('batch_metrics').get_metric_validation('mse')
    epoch_mse_validation = p.get_step_by_name('epoch_metrics').get_metric_validation('mse')

    assert len(epoch_mse_train) == N_EPOCHS
    assert len(epoch_mse_validation) == N_EPOCHS

    assert batch_mse_train[-1] < 100
    assert batch_mse_validation[-1] < 100
    assert epoch_mse_train[-1] < 100
    assert epoch_mse_validation[-1] < 100


def to_numpy_metric_wrapper(metric_fun):
    def metric(data_inputs, expected_outputs):
        return metric_fun(np.array(data_inputs), np.array(expected_outputs))

    return metric
