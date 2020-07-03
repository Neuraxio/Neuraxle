"""
Neuraxle's Higher Order Steps
===============================================================================================================
A higher-order step is a function that takes an existing step and returns another step that wraps it.
An higher-order step is never be saved because it is note serializable.
It is useful for applying side effects that are outside of the scope of the pipeline execution.

..
    Copyright 2019, Neuraxio Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""
from abc import ABC

from neuraxle.base import BaseTransformer, ExecutionContext, MetaStep, ForceHandleMixin
from neuraxle.data_container import DataContainer


class BaseHigherOrderStep(ForceHandleMixin, MetaStep, ABC):
    """
    Base class to represent an higher order step.
    A higher-order step is just a function that takes an existing step and returns another step that wraps it.

    An higher order step:
        - cannot be saved
        - cannot modify the data container
        - cannot push himself in the execution context

    .. seealso::
        :class:`~neuraxle.base.MetaStep`
    """
    def __init__(self, step: BaseTransformer):
        MetaStep.__init__(self, wrapped=step)
        ForceHandleMixin.__init__(self)

    def save(self, context: ExecutionContext, full_dump=False):
        """
        Don't save the higher order step. Save the wrapped step only.

        :param context: context to save.
        :param full_dump: save full dump or not
        :return: saved wrapped step
        """
        if len(context) > 0 and isinstance(context.parents[-1], BaseHigherOrderStep):
            context.pop()
        return self.wrapped.save(context, full_dump=full_dump)

    def _will_process(self, data_container: DataContainer, context: ExecutionContext) -> (DataContainer, ExecutionContext):
        return data_container, context

    def _will_fit(self, data_container: DataContainer, context: ExecutionContext) -> (DataContainer, ExecutionContext):
        return data_container, context

    def _will_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> (DataContainer, ExecutionContext):
        return data_container, context

    def _will_fit_transform(self, data_container: DataContainer, context: ExecutionContext):
        return data_container, context

    def _did_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        return data_container

    def _did_process(self, data_container: DataContainer, context: ExecutionContext):
        return data_container

    def _did_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        return data_container


def WithContext(step: BaseTransformer, context: ExecutionContext):
    """
    An higher order step to inject a context inside a step.
    A step with a context forces the pipeline to use that context through handler methods.
    This is useful for dependency injection because you can register services inside the :class:`ExecutionContext`.
    It also ensures that the context has registered all the necessary services.

    .. code-block:: python

        context = ExecutionContext(root=tmpdir)
        context.set_service_locator(ServiceLocator().services) # where services is of type Dict[Type, object]

        p = WithContext(Pipeline([
            SomeStep().with_assertion_has_services(BaseService)
        ]), context)


    .. seealso::
        :class:`BaseStep`,
        :class:`ExecutionContext`,
        :class:`BaseTransformer`
    """
    class StepWithContext(BaseHigherOrderStep):
        def __init__(self, wrapped: BaseTransformer):
            super().__init__(step=wrapped)
            self.apply('_assert_has_services', context=context)

        def _will_process(self, data_container: DataContainer, _: ExecutionContext) -> (DataContainer, ExecutionContext):
            """
            Inject the given context before processing the wrapped step.

            :param data_container: data container to process
            :return: data container, execution context
            """
            return data_container, context

    return StepWithContext(wrapped=step)
