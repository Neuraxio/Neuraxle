import numpy as np
import pytest

from neuraxle.base import ForceHandleMixin, Identity, BaseStep, ExecutionContext
from neuraxle.data_container import DataContainer
from neuraxle.pipeline import Pipeline


class BadForceHandleStep(ForceHandleMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        ForceHandleMixin.__init__(self)


class ForceHandleIdentity(ForceHandleMixin, Identity):
    def __init__(self):
        Identity.__init__(self)
        ForceHandleMixin.__init__(self)


def test_raises_exception_if_method_not_redefined(tmpdir):
    # Now that I think about it, this really just is a complicated way to test the self._ensure_method_overriden function.
    with pytest.raises(NotImplementedError) as exception_info:
        BadForceHandleStep()

    assert "_fit_data_container" in exception_info.value.args[0]

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext):
        return self

    BadForceHandleStep._fit_data_container = _fit_data_container

    with pytest.raises(NotImplementedError) as exception_info:
        BadForceHandleStep()

    assert "_fit_transform_data_container" in exception_info.value.args[0]

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
        return self, data_container

    BadForceHandleStep._fit_data_container = _fit_data_container

    with pytest.raises(NotImplementedError) as exception_info:
        BadForceHandleStep()

    assert "_transform_data_container" in exception_info.value.args[0]


def test_forcehandleidentity_does_not_crash(tmpdir):
    p = Pipeline([
        ForceHandleIdentity()
    ])
    data_inputs = np.array([0, 1, 2, 3])
    expected_outputs = data_inputs * 2
    p.fit(data_inputs, expected_outputs)
    p.fit_transform(data_inputs, expected_outputs)
    p.transform(data_inputs=data_inputs)
