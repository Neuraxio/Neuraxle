import os
from abc import ABC, abstractmethod

import pytest

from neuraxle.base import Identity, ExecutionContext, ForceHandleMixin
from neuraxle.data_container import DataContainer
from neuraxle.pipeline import Pipeline
import numpy as np


class BaseService(ABC):
    @abstractmethod
    def service_method(self, data):
        pass


class SomeService(BaseService):
    def service_method(self, data):
        self.data = data


class SomeStepThatChangesTheRootOfTheExecutionContext(ForceHandleMixin, Identity):
    def __init__(self):
        Identity.__init__(self)
        ForceHandleMixin.__init__(self)

    def _will_process(self, data_container: DataContainer, context: ExecutionContext):
        context.root = 'invalid_root'
        super()._will_process(data_container, context)
        return data_container, context

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
        return data_container


class SomeStep(ForceHandleMixin, Identity):
    def __init__(self):
        Identity.__init__(self)
        ForceHandleMixin.__init__(self)

    def _will_process(self, data_container: DataContainer, context: ExecutionContext):
        data_container, context = super()._will_process(data_container, context)
        service = context.get_service(BaseService)
        return data_container, context

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
        service: BaseService = context.get_service(BaseService)
        service.service_method(data_container.data_inputs)
        return data_container


def test_step_with_context_should_only_save_wrapped_step(tmpdir):
    context = ExecutionContext(root=tmpdir)
    service = SomeService()
    context.set_service_locator({BaseService: service})
    p = Pipeline([
        SomeStep().assert_has_services(BaseService)
    ]).with_context(context=context)

    p.save(context, full_dump=True)

    p: Pipeline = ExecutionContext(root=tmpdir).load(os.path.join('StepWithContext', 'Pipeline'))
    assert isinstance(p, Pipeline)


def test_with_context_should_inject_dependencies_properly(tmpdir):
    data_inputs = np.array([0, 1, 2, 3])
    context = ExecutionContext(root=tmpdir)
    service = SomeService()
    context.set_service_locator({BaseService: service})
    p = Pipeline([
        SomeStep().assert_has_services(BaseService)
    ]).with_context(context=context)

    p.transform(data_inputs=data_inputs)

    assert np.array_equal(service.data, data_inputs)


def test_add_service_assertions_should_fail_when_services_are_missing(tmpdir):
    with pytest.raises(AssertionError) as exception_info:
        context = ExecutionContext(root=tmpdir)
        p = Pipeline([
            SomeStep().assert_has_services(BaseService)
        ]).with_context(context=context)
        data_inputs = np.array([0, 1, 2, 3])

        p.transform(data_inputs=data_inputs)

    assert 'BaseService dependency missing' in exception_info.value.args[0]

