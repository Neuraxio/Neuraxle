import numpy as np

from neuraxle.base import Identity
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import NumpyTranspose, NumpyConcatenateInnerFeatures
from neuraxle.union import FeatureUnion


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
