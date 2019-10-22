import numpy as np

from examples.nested_pipelines import main


def test_nested_pipelines():
    n_components = main()

    assert isinstance(n_components, np.ndarray)
    assert n_components.shape == (2, 4)
