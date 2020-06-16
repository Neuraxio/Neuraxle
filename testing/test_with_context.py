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
        SomeStep().with_assertion_has_services(BaseService).transform(list(range(10)))

def test_with_context(tmpdir):
    step = SomeStep().with_context(ExecutionContext(root=tmpdir))
    step.transform(list(range(10)))
