import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from neuraxle.api import DeepLearningPipeline
from neuraxle.hyperparams.distributions import RandInt, LogUniform, Boolean
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import NumpyTranspose
from neuraxle.steps.sklearn import SKLearnWrapper
from neuraxle.union import AddFeatures, ModelStacking


def test_deep_learning_pipeline():
    boston = load_boston()
    data_inputs, expected_outputs = shuffle(boston.data, boston.target, random_state=13)
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
                KMeans(),
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
        batch_metrics=[accuracy_score],
        shuffle_in_each_epoch_at_train=True,
        n_epochs=100,
        epochs_metrics=[accuracy_score],
        scoring_function=accuracy_score,
    )

    p.fit_transform(data_inputs, expected_outputs)

