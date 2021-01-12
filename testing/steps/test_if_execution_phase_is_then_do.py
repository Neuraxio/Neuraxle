import numpy as np
import pytest
from pytest import skip

from neuraxle.base import ExecutionContext, ExecutionPhase
from neuraxle.data_container import DataContainer
from neuraxle.steps.flow import IfExecutionPhaseIsThenDo, ExecutionPhaseSwitch
from testing.test_forcehandle_mixin import ForceHandleIdentity


# TODO : add test for ExecutionPhaseSwitch


class SomeStep(ForceHandleIdentity):
    def __init__(self):
        ForceHandleIdentity.__init__(self)
        self.did_process = False

    def _did_process(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        self.did_process = True
        return data_container


def test_ifexecphase_same_then_execute_step(tmpdir):
    _run(tmpdir, ExecutionPhase.TRAIN, True)


def test_ifexecphase_different_then_skip_step(tmpdir):
    _run(tmpdir, ExecutionPhase.TEST, False)


def _run(tmpdir, phase, expected):
    context = ExecutionContext(root=tmpdir, execution_phase=phase)
    data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    some_step = SomeStep()
    p = IfExecutionPhaseIsThenDo(ExecutionPhase.TRAIN, some_step)
    p = p.with_context(context)

    p.fit_transform(data_inputs)
    assert some_step.did_process is expected


def test_ifexecphase_raise_exception_when_unspecified(tmpdir):
    context = ExecutionContext(root=tmpdir)
    data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    some_step = SomeStep()
    p = IfExecutionPhaseIsThenDo(ExecutionPhase.TRAIN, some_step)
    p = p.with_context(context)

    with pytest.raises(ValueError) as error_info:
        p.fit_transform(data_inputs)
    assert some_step.did_process is False


def test_execswitch(tmpdir):
    context = ExecutionContext(root=tmpdir, execution_phase=ExecutionPhase.TRAIN)
    data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    phase_to_step = {p: SomeStep() for p in (ExecutionPhase.PRETRAIN, ExecutionPhase.TRAIN, ExecutionPhase.TEST)}
    p = ExecutionPhaseSwitch(phase_to_step)
    p_c = p.with_context(context)

    p_c.fit_transform(data_inputs)
    assert phase_to_step[ExecutionPhase.PRETRAIN].did_process is False
    assert phase_to_step[ExecutionPhase.TRAIN].did_process is True
    assert phase_to_step[ExecutionPhase.TEST].did_process is False

    p_c = p.with_context(context.set_execution_phase(ExecutionPhase.UNSPECIFIED))
    with pytest.raises(KeyError) as error_info:
        p_c.fit_transform(data_inputs)
