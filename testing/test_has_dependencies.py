from abc import ABC, abstractmethod

import pytest

from neuraxle.base import Identity, ExecutionContext, ForceHandleMixin
from neuraxle.data_container import DataContainer


class BaseService(ABC):
    @abstractmethod
    def service_method(self, data):
        pass

class SomeService(BaseService):
    def service_method(self, data):
        self.data = data


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

def test_add_service_assertions():
    with pytest.raises(Exception) as exception:
        Identity().add_service_assertions(BaseService)

def test_register_service_with_step():
    step = SomeStep()
    service = SomeService()
    step.register_service(BaseService, service)
    data_inputs = list(range(10))

    step.transform(data_inputs)

    assert service.data == data_inputs

def test_set_services_with_step():
    step = SomeStep()
    service = SomeService()
    step.set_services({
        BaseService: service
    })
    data_inputs = list(range(10))

    step.transform(data_inputs)

    assert service.data == data_inputs

def test_register_service_with_context():
    step = SomeStep()
    service = SomeService()
    data_inputs = list(range(10))

    step.handle_transform(DataContainer(data_inputs=data_inputs), ExecutionContext().register_service(BaseService, service))

    assert service.data == data_inputs

def test_set_services_with_context():
    step = SomeStep()
    service = SomeService()
    data_inputs = list(range(10))

    step.handle_transform(DataContainer(data_inputs=data_inputs), ExecutionContext().set_services({
        BaseService: service
    }))

    assert service.data == data_inputs
