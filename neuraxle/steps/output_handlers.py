"""
Output Handlers Steps
====================================
You can find here output handlers steps that changes especially the data outputs.

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

..
    Thanks to Umaneo Technologies Inc. for their contributions to this Machine Learning
    project, visit https://www.umaneo.com/ for more information on Umaneo Technologies Inc.

"""
import copy
from abc import ABC
from typing import List

from neuraxle.base import ExecutionContext, BaseStep, MetaStep, ForceHandleOnlyMixin, BaseHasher, \
    MixinForBaseTransformer
from neuraxle.data_container import DataContainer


class OutputTransformerWrapper(ForceHandleOnlyMixin, MetaStep):
    """
    Transform expected output wrapper step that can sends the expected_outputs to the wrapped step
    so that it can transform the expected outputs.
    """

    def __init__(self, wrapped, cache_folder_when_no_handle=None):
        MetaStep.__init__(self, wrapped)
        ForceHandleOnlyMixin.__init__(self, cache_folder_when_no_handle)

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Handle transform by passing expected outputs to the wrapped step transform method.
        Update the expected outputs with the outputs.

        :param context: execution context
        :param data_container:
        :return: data container
        :rtype: DataContainer
        """
        new_expected_outputs_data_container = self.wrapped.handle_transform(
            DataContainer(
                data_inputs=data_container.expected_outputs,
                current_ids=data_container.current_ids,
                expected_outputs=None
            ),
            context
        )
        self._set_expected_outputs(data_container, new_expected_outputs_data_container)

        return data_container

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> (
            BaseStep, DataContainer):
        """
        Handle fit by passing expected outputs to the wrapped step fit method.

        :param context: execution context
        :type context: ExecutionContext
        :param data_container: data container to fit on
        :return: self, data container
        :rtype: (BaseStep, DataContainer)
        """
        self.wrapped = self.wrapped.handle_fit(
            DataContainer(
                data_inputs=data_container.expected_outputs,
                current_ids=data_container.current_ids,
                expected_outputs=None
            ),
            context
        )

        return self, data_container

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> (
            BaseStep, DataContainer):
        """
        Handle fit transform by passing expected outputs to the wrapped step fit method.
        Update the expected outputs with the outputs.

        :param context: execution context
        :type context: ExecutionContext
        :param data_container: data container to fit on
        :return: self, data container
        :rtype: (BaseStep, DataContainer)
        """
        self.wrapped, new_expected_outputs_data_container = self.wrapped.handle_fit_transform(
            DataContainer(
                data_inputs=data_container.expected_outputs,
                current_ids=data_container.current_ids,
                expected_outputs=None
            ),
            context
        )
        self._set_expected_outputs(data_container, new_expected_outputs_data_container)

        return self, data_container

    def handle_inverse_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Handle inverse transform by passing expected outputs to the wrapped step inverse transform method.
        Update the expected outputs with the outputs.

        :param context: execution context
        :param data_container:
        :return: data container
        :rtype: DataContainer
        """
        new_expected_outputs_data_container = self.wrapped.handle_inverse_transform(
            DataContainer(
                data_inputs=data_container.expected_outputs,
                current_ids=data_container.current_ids,
                expected_outputs=None
            ),
            context.push(self.wrapped)
        )

        self._set_expected_outputs(data_container, new_expected_outputs_data_container)

        return data_container

    def _set_expected_outputs(self, data_container, new_expected_outputs_data_container) -> DataContainer:
        if len(data_container.data_inputs) != len(data_container.expected_outputs):
            raise AssertionError(
                'OutputTransformerWrapper: Found different len for data inputs, and expected outputs. '
                'Please return the same the same amount of data inputs, and expected outputs, '
                'or otherwise create your own handler methods to do more funky things.')

        data_container.set_expected_outputs(new_expected_outputs_data_container.data_inputs)
        data_container.set_current_ids(new_expected_outputs_data_container.current_ids)

        return data_container


class _DidProcessInputOutputHandlerMixin(MixinForBaseTransformer):
    def _did_process(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        di, eo = data_container.data_inputs
        if len(di) != len(eo):
            raise AssertionError(
                '{}: Found different len for data inputs, and expected outputs. Please return the same the same amount of data inputs, and expected outputs, or otherwise create your own handler methods to do more funky things.'.format(
                    self.name))

        data_container.set_data_inputs(data_inputs=di)
        data_container.set_expected_outputs(expected_outputs=eo)

        data_container = super()._did_process(data_container, context)

        if len(data_container.current_ids) != len(data_container.data_inputs):
            raise AssertionError(
                '{}: Caching broken because there is a different len of current ids, and data inputs. Please use InputAndOutputTransformerWrapper if you plan to change the len of the data inputs.'.format(
                    self.name))

        return data_container


class InputAndOutputTransformerWrapper(_DidProcessInputOutputHandlerMixin, ForceHandleOnlyMixin, MetaStep):
    """
    Wrapper step to transform both data inputs, and expected output at the same.
    It sends the data_inputs, and the expected_outputs to the wrapped step so that it can transform them.

    .. seealso::
        :class:`~neuraxle.base.BaseStep`,
        :class:`~neuraxle.base.ForceHandleOnlyMixin`
    """

    def __init__(self, wrapped, hashers: List[BaseHasher] = None, cache_folder_when_no_handle=None):
        MetaStep.__init__(self, wrapped, hashers=hashers)
        ForceHandleOnlyMixin.__init__(self, cache_folder_when_no_handle)
        _DidProcessInputOutputHandlerMixin.__init__(self)

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Handle transform by passing data_inputs, and expected outputs to the wrapped step transform method.
        Update the expected outputs with the outputs.

        :param context: execution context
        :param data_container:
        :return: data container
        :rtype: DataContainer
        """
        output_data_container = self.wrapped.handle_transform(
            DataContainer(
                data_inputs=(data_container.data_inputs, data_container.expected_outputs),
                current_ids=data_container.current_ids,
                expected_outputs=None
            ),
            context
        )

        return output_data_container

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> (
            BaseStep, DataContainer):
        """
        Handle fit by passing the data inputs, and the expected outputs to the wrapped step fit method.

        :param context: execution context
        :type context: ExecutionContext
        :param data_container: data container to fit on
        :return: self, data container
        :rtype: (BaseStep, DataContainer)
        """
        self.wrapped = self.wrapped.handle_fit(
            DataContainer(
                data_inputs=(copy.copy(data_container.data_inputs), copy.copy(data_container.expected_outputs)),
                current_ids=data_container.current_ids,
                expected_outputs=None
            ),
            context
        )

        data_container.set_data_inputs((data_container.data_inputs, data_container.expected_outputs))
        data_container.set_expected_outputs(expected_outputs=None)

        return self

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> (BaseStep, DataContainer):
        """
        Handle fit transform by passing the data inputs, and the expected outputs to the wrapped step fit method.
        Update the expected outputs with the outputs.

        :param context: execution context
        :type context: ExecutionContext
        :param data_container: data container to fit on
        :return: self, data container
        :rtype: (BaseStep, DataContainer)
        """
        self.wrapped, output_data_container = self.wrapped.handle_fit_transform(
            DataContainer(
                data_inputs=(data_container.data_inputs, data_container.expected_outputs),
                current_ids=data_container.current_ids,
                expected_outputs=None
            ),
            context
        )
        return self, output_data_container

    def handle_inverse_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Handle inverse transform by passing the data inputs, and the expected outputs to the wrapped step inverse transform method.
        Update the expected outputs with the outputs.

        :param context: execution context
        :param data_container:
        :return: data container
        :rtype: DataContainer
        """
        output_data_container = self.wrapped.handle_inverse_transform(
            DataContainer(
                data_inputs=(data_container.data_inputs, data_container.expected_outputs),
                current_ids=data_container.current_ids,
                expected_outputs=None
            ),
            context.push(self.wrapped)
        )

        current_ids = self.hash(data_container)
        data_container.set_current_ids(current_ids)

        return output_data_container


class InputAndOutputTransformerMixin(_DidProcessInputOutputHandlerMixin):
    """
    Base output transformer step that can modify data inputs, and expected_outputs at the same time.
    """

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Handle inverse transform by updating the data inputs, and expected outputs inside the data container.

        :param context: execution context
        :param data_container:
        :return:
        """
        di_eo = (data_container.data_inputs, data_container.expected_outputs)
        new_data_inputs, new_expected_outputs = self.transform(di_eo)
        data_container.set_data_inputs((new_data_inputs, new_expected_outputs))
        return data_container

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> 'BaseStep':
        """
        Handle transform by fitting the step,
        and updating the data inputs, and expected outputs inside the data container.

        :param context: execution context
        :param data_container:
        :return:
        """
        new_self = self.fit((data_container.data_inputs, data_container.expected_outputs), None)
        data_container.set_data_inputs((data_container.data_inputs, data_container.expected_outputs))
        data_container.set_expected_outputs(expected_outputs=None)
        return new_self

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        """
        Handle transform by fitting the step,
        and updating the data inputs, and expected outputs inside the data container.

        :param context: execution context
        :param data_container:
        :return:
        """
        new_self, (new_data_inputs, new_expected_outputs) = self.fit_transform(
            (data_container.data_inputs, data_container.expected_outputs), None)
        data_container.set_data_inputs((new_data_inputs, new_expected_outputs))
        return new_self, data_container
