import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from neuraxle.base import Identity
from neuraxle.hyperparams.distributions import RandInt, LogUniform, Boolean
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import NumpyTranspose, NumpyConcatenateInnerFeatures
from neuraxle.steps.sklearn import SKLearnWrapper
from neuraxle.union import FeatureUnion, ModelStacking


def test_feature_union_should_transform_with_concatenate_inner_features():
    p = Pipeline([
        FeatureUnion([
            Identity(),
            Identity(),
        ], joiner=NumpyConcatenateInnerFeatures())
    ])
    data_inputs = np.random.randint((1, 20))

    outputs = p.transform(data_inputs)

    assert np.array_equal(outputs, np.concatenate([data_inputs, data_inputs]))


def test_feature_union_should_fit_transform_with_concatenate_inner_features():
    p = Pipeline([
        FeatureUnion([
            Identity(),
            Identity(),
        ], joiner=NumpyConcatenateInnerFeatures())
    ])
    data_inputs = np.random.randint((1, 20))
    expected_outputs = np.random.randint((1, 20))

    p, outputs = p.fit_transform(data_inputs, expected_outputs)

    assert np.array_equal(outputs, np.concatenate([data_inputs, data_inputs]))


def test_feature_union_should_transform_with_numpy_transpose():
    p = Pipeline([
        FeatureUnion([
            Identity(),
            Identity(),
        ], joiner=NumpyTranspose())
    ])
    data_inputs = np.random.randint((1, 20))

    outputs = p.transform(data_inputs)

    assert np.array_equal(outputs, np.array([data_inputs, data_inputs]).transpose())


def test_feature_union_should_fit_transform_with_numpy_transpose():
    p = Pipeline([
        FeatureUnion([
            Identity(),
            Identity(),
        ], joiner=NumpyTranspose())
    ])
    data_inputs = np.random.randint((1, 20))
    expected_outputs = np.random.randint((1, 20))

    p, outputs = p.fit_transform(data_inputs, expected_outputs)

    assert np.array_equal(outputs, np.array([data_inputs, data_inputs]).transpose())


def test_feature_union_should_apply_to_self_and_sub_steps():
    p = Pipeline([
        FeatureUnion([
            Identity(),
            Identity(),
        ], joiner=NumpyTranspose())
    ])

    p.apply('set_hyperparams', HyperparameterSamples({'applied': True}))

    assert p.hyperparams['applied']
    assert p['FeatureUnion'].hyperparams['applied']
    assert p['FeatureUnion'][0].hyperparams['applied']
    assert p['FeatureUnion'][1].hyperparams['applied']
    assert p['FeatureUnion'][2].hyperparams['applied']


def test_model_stacking_fit_transform():
    model_stacking = Pipeline([
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
                HyperparameterSpace({
                    "n_clusters": RandInt(5, 10)
                })
            ),
        ],
            joiner=NumpyTranspose(),
            judge=SKLearnWrapper(
                Ridge(),
                HyperparameterSpace({
                    "alpha": LogUniform(0.7, 1.4),
                    "fit_intercept": Boolean()
                })
            ),
        )
    ])
    expected_outputs_shape = (379, 1)
    data_inputs_shape = (379, 13)
    data_inputs = _create_data(data_inputs_shape)
    expected_outputs = _create_data(expected_outputs_shape)

    model_stacking, outputs = model_stacking.fit_transform(data_inputs, expected_outputs)

    assert outputs.shape == expected_outputs_shape


def test_model_stacking_transform():
    model_stacking = Pipeline([
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
                HyperparameterSpace({
                    "n_clusters": RandInt(5, 10)
                })
            ),
        ],
            joiner=NumpyTranspose(),
            judge=SKLearnWrapper(
                Ridge(),
                HyperparameterSpace({
                    "alpha": LogUniform(0.7, 1.4),
                    "fit_intercept": Boolean()
                })
            ),
        )
    ])
    expected_outputs_shape = (379, 1)
    data_inputs_shape = (379, 13)
    data_inputs = _create_data(data_inputs_shape)
    expected_outputs = _create_data(expected_outputs_shape)

    model_stacking = model_stacking.fit(data_inputs, expected_outputs)
    outputs = model_stacking.transform(data_inputs)

    assert outputs.shape == expected_outputs_shape


def _create_data(shape):
    data_inputs = np.random.random(shape).astype(np.float32)
    return data_inputs
