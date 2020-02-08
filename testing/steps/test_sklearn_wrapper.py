import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from neuraxle.base import Identity
from neuraxle.steps.sklearn import SKLearnWrapper


def test_sklearn_wrapper_with_an_invalid_step():
    with pytest.raises(ValueError):
        SKLearnWrapper(Identity())


def test_sklearn_wrapper_fit_transform_with_predict():
    p = SKLearnWrapper(LinearRegression())
    data_inputs = np.expand_dims(np.array(list(range(10))), axis=-1)
    expected_outputs = np.expand_dims(np.array(list(range(10, 20))), axis=-1)

    p, outputs = p.fit_transform(data_inputs, expected_outputs)

    assert np.array_equal(outputs, expected_outputs)


def test_sklearn_wrapper_transform_with_predict():
    p = SKLearnWrapper(LinearRegression())
    data_inputs = np.expand_dims(np.array(list(range(10))), axis=-1)
    expected_outputs = np.expand_dims(np.array(list(range(10, 20))), axis=-1)

    p = p.fit(data_inputs, expected_outputs)
    outputs = p.transform(data_inputs)

    assert np.array_equal(outputs, expected_outputs)


def test_sklearn_wrapper_fit_transform_with_transform():
    n_components = 2
    p = SKLearnWrapper(PCA(n_components=n_components))
    dim1 = 10
    dim2 = 10
    data_inputs, expected_outputs = _create_data_source((dim1, dim2))

    p, outputs = p.fit_transform(data_inputs, expected_outputs)

    assert outputs.shape == (dim1, n_components)


def _create_data_source(shape):
    data_inputs = np.random.random(shape).astype(np.float32)
    expected_outputs = np.random.random(shape).astype(np.float32)
    return data_inputs, expected_outputs
