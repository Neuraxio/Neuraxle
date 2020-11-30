import pytest
import numpy as np
from neuraxle.base import AssertExpectedOutputNullMixin, BaseStep, Identity
from neuraxle.data_container import DataContainer
from neuraxle.pipeline import Pipeline


class SomeStep(AssertExpectedOutputNullMixin,Identity):
    def __init__(self):
        Identity.__init__(self)

def test_expectedoutputnull_raise_exception_when_notnull(tmpdir):

    data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_outputs = data_inputs * 2

    p = Pipeline([SomeStep()])

    with pytest.raises(AssertionError) as error_info:
        p.fit_transform(data_inputs,expected_outputs)

def test_expectedoutputnull_is_fine_when_null(tmpdir):

    data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_outputs = None

    p = Pipeline([SomeStep()])
    p.fit_transform(data_inputs,expected_outputs)
