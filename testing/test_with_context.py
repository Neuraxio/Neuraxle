from abc import ABC, abstractmethod

import pytest

from neuraxle.base import Identity, ExecutionContext, ForceHandleMixin
from neuraxle.data_container import DataContainer
from neuraxle.pipeline import Pipeline


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


def test_add_service_assertions_should_fail_when_services_are_missing():
    with pytest.raises(AssertionError) as exception:
        SomeStep().with_assertion_has_services(BaseService).transform(list(range(10)))


def test_with_context_should_add_expected_root_path_and_assert_it_is_as_expected(tmpdir):
    with pytest.raises(AssertionError) as exception:
        step = Pipeline([
            SomeStepThatChangesTheRootOfTheExecutionContext()
        ]).with_context(ExecutionContext(root=tmpdir))

    step.transform(list(range(10)))


def test_with_context_should_inject_dependencies_properly(tmpdir):
    context = ExecutionContext(root=tmpdir)
    service = SomeService()
    context.set_service_locator({BaseService: service})
    data_inputs = list(range(10))
    step = Pipeline([
        SomeStep().with_assertion_has_services(BaseService)
    ]).with_context(context)

    step.transform(data_inputs)

    assert service.data == data_inputs


def test_step_with_context_should_be_saveable(tmpdir):
    context = ExecutionContext(root=tmpdir)
    service = SomeService()
    context.set_service_locator({BaseService: service})
    p = Pipeline([
        SomeStep().with_assertion_has_services(BaseService)
    ]).with_context(context)

    p.save(context, full_dump=True)

    p: Pipeline = p.load(context, full_dump=True)
    assert isinstance(p, Pipeline)
