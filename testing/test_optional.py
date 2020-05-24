import numpy as np

from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.steps.flow import Optional
from neuraxle.steps.numpy import MultiplyByN


def test_optional_should_disable_wrapped_step_when_disabled():
    p = Optional(MultiplyByN(2), nullified_return_value=[]).set_hyperparams(HyperparameterSamples({
        'enabled': False
    }))
    data_inputs = np.array(list(range(10)))

    outputs = p.transform(data_inputs)

    assert outputs == []


def test_optional_should_enable_wrapped_step_when_enabled():
    p = Optional(MultiplyByN(2), nullified_return_value=[]).set_hyperparams(HyperparameterSamples({
        'enabled': True
    }))
    data_inputs = np.array(list(range(10)))

    outputs = p.transform(data_inputs)

    assert np.array_equal(outputs, data_inputs * 2)

