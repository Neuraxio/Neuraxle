import numpy as np
import pytest

from neuraxle.steps.encoding import OneHotEncoder


@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize("no_columns", [10])
def test_one_hot_encode_should_encode_data_inputs(n_dims, no_columns):
    one_hot_encode = OneHotEncoder(nb_columns=no_columns, name='one_hot')
    data_shape = list(range(100, 200))[:n_dims]
    data_inputs = np.random.randint(low=no_columns, size=data_shape)
    data_inputs[0] = 0
    data_inputs[1] = no_columns - 1
    data_inputs[-2] = -1  # or nan or inf.

    outputs = one_hot_encode.transform(data_inputs)

    assert outputs.shape[-1] == no_columns
    assert ((outputs == 1) | (outputs == 0)).all()

    if n_dims >= 2:
        assert (outputs[0, ..., 0] == 1).all()
        assert (outputs[1, ..., -1] == 1).all()
        assert (outputs[-2, ...] == 0).all()
