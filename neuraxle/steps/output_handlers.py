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
from typing import Tuple

from neuraxle.base import BaseStep
from neuraxle.base import ExecutionContext as CX
from neuraxle.base import (ForceHandleOnlyMixin, MetaStep,
                           MixinForBaseTransformer)
from neuraxle.data_container import DataContainer as DACT


class OutputTransformerWrapper(ForceHandleOnlyMixin, MetaStep):
    """
    A step that can sends the expected_outputs to the wrapped step
    so that it can transform the expected outputs.
    """

    def __init__(self, wrapped):
        MetaStep.__init__(self, wrapped)
        ForceHandleOnlyMixin.__init__(self)

    def _transform_data_container(self, data_container: DACT, context: CX) -> DACT:
        """
        Handle transform by passing expected outputs to the wrapped step transform method.
        Update the expected outputs with the outputs.

        :param context: execution context
        :param data_container:
        :return: data container
        :rtype: DataContainer
        """
        new_expected_outputs_data_container = self.wrapped.handle_transform(
            DACT(
                ids=data_container.ids,
                data_inputs=data_container.eo,
                expected_outputs=None
            ),
            context
        )
        self._set_expected_outputs(data_container, new_expected_outputs_data_container, context)

        return data_container

    def _fit_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple[BaseStep, DACT]:
        """
        Handle fit by passing expected outputs to the wrapped step fit method.

        :param context: execution context
        :type context: ExecutionContext
        :param data_container: data container to fit on
        :return: self, data container
        """
        self.wrapped = self.wrapped.handle_fit(
            DACT(
                data_inputs=data_container.expected_outputs,
                ids=data_container.ids,
                expected_outputs=None
            ),
            context
        )

        return self, data_container

    def _fit_transform_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple[BaseStep, DACT]:
        """
        Handle fit transform by passing expected outputs to the wrapped step fit method.
        Update the expected outputs with the outputs.

        :param context: execution context
        :type context: ExecutionContext
        :param data_container: data container to fit on
        :return: self, data container
        """
        self.wrapped, new_expected_outputs_data_container = self.wrapped.handle_fit_transform(
            DACT(
                data_inputs=data_container.expected_outputs,
                ids=data_container.ids,
                expected_outputs=None
            ),
            context
        )
        self._set_expected_outputs(data_container, new_expected_outputs_data_container, context)

        return self, data_container

    def handle_inverse_transform(self, data_container: DACT, context: CX) -> DACT:
        """
        Handle inverse transform by passing expected outputs to the wrapped step inverse transform method.
        Update the expected outputs with the outputs.

        :param context: execution context
        :param data_container:
        :return: data container
        :rtype: DataContainer
        """
        new_expected_outputs_data_container = self.wrapped.handle_inverse_transform(
            DACT(
                data_inputs=data_container.expected_outputs,
                ids=data_container.ids,
                expected_outputs=None
            ),
            context.push(self.wrapped)
        )

        self._set_expected_outputs(data_container, new_expected_outputs_data_container, context)

        return data_container

    def _set_expected_outputs(
            self, data_container: DACT, new_expected_outputs_data_container: DACT, context: CX
    ) -> DACT:

        self._assert(
            len(data_container) == len(new_expected_outputs_data_container),
            'OutputTransformerWrapper: Found different len for old data inputs, and expected outputs '
            'to reinsert. Please return the same the same amount of data inputs, and expected outputs, '
            'or otherwise create your own handler methods to do more funky things.',
            context
        )

        data_container.set_expected_outputs(new_expected_outputs_data_container.data_inputs)
        data_container.set_ids(new_expected_outputs_data_container._ids)

        return data_container


class _DidProcessIdsInputOutputHandlerMixin(MixinForBaseTransformer):
    def _did_process(self, data_container: DACT, context: CX) -> DACT:
        ids, di, eo = data_container.data_inputs
        self._assert(
            di is None or eo is None or ids is None or (len(di) == len(eo) and len(ids) == len(di)),
            f'{self.name}: Found different len for non-null data inputs, and expected outputs. '
            f'Please return the same the same amount of data inputs, and expected outputs, or '
            f'otherwise create your own handler methods to do more funky things.',
            context
        )

        data_container.set_ids(ids=ids)
        data_container.set_data_inputs(data_inputs=di)
        data_container.set_expected_outputs(expected_outputs=eo)

        data_container = super()._did_process(data_container, context)

        self._assert(
            len(data_container.ids) == len(data_container.data_inputs),
            f'{self.name}: Caching broken because there is a different len of current ids, and data inputs. Please use InputAndOutputTransformerWrapper if you plan to change the len of the data inputs.',
            context
        )

        return data_container


class IdsInputAndOutputTransformerWrapper(_DidProcessIdsInputOutputHandlerMixin, ForceHandleOnlyMixin, MetaStep):
    """
    Wrapper step to transform both ids, data inputs, and expected output at the same time using classical fit and transform methods.
    It sends the data_inputs, and the expected_outputs to the wrapped step so that it can transform them.

    .. seealso::
        :class:`~neuraxle.base.BaseStep`,
        :class:`~neuraxle.base.ForceHandleOnlyMixin`
    """

    def __init__(self, wrapped):
        MetaStep.__init__(self, wrapped)
        ForceHandleOnlyMixin.__init__(self)
        _DidProcessIdsInputOutputHandlerMixin.__init__(self)

    def _transform_data_container(self, data_container: DACT, context: CX) -> DACT:
        """
        Handle transform by passing data_inputs, and expected outputs to the wrapped step transform method.
        Update the expected outputs with the outputs.

        :param context: execution context
        :param data_container:
        :return: data container
        :rtype: DataContainer
        """
        output_data_container = self.wrapped.handle_transform(
            DACT(
                ids=data_container._ids,
                data_inputs=(data_container._ids, data_container.data_inputs, data_container.expected_outputs),
                expected_outputs=None
            ),
            context
        )

        return output_data_container

    def _fit_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple[BaseStep, DACT]:
        """
        Handle fit by passing the data inputs, and the expected outputs to the wrapped step fit method.

        :param context: execution context
        :type context: ExecutionContext
        :param data_container: data container to fit on
        :return: self, data container
        """
        self.wrapped = self.wrapped.handle_fit(
            DACT(
                ids=data_container._ids,
                data_inputs=(copy.copy(data_container._ids), copy.copy(
                    data_container.data_inputs), copy.copy(data_container.expected_outputs)),
                expected_outputs=None
            ),
            context
        )

        data_container.set_data_inputs(
            (data_container._ids, data_container.data_inputs, data_container.expected_outputs))
        data_container.set_expected_outputs(expected_outputs=None)  # TODO: thy this?

        return self

    def _fit_transform_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple[BaseStep, DACT]:
        """
        Handle fit transform by passing the data inputs, and the expected outputs to the wrapped step fit method.
        Update the expected outputs with the outputs.

        :param context: execution context
        :type context: ExecutionContext
        :param data_container: data container to fit on
        :return: self, data container
        """
        self.wrapped, output_data_container = self.wrapped.handle_fit_transform(
            DACT(
                ids=data_container._ids,
                data_inputs=(data_container._ids, data_container.data_inputs, data_container.expected_outputs),
                expected_outputs=None
            ),
            context
        )
        return self, output_data_container

    def handle_inverse_transform(self, data_container: DACT, context: CX) -> DACT:
        """
        Handle inverse transform by passing the data inputs, and the expected outputs to the wrapped step inverse transform method.
        Update the expected outputs with the outputs.

        :param context: execution context
        :param data_container:
        :return: data container
        :rtype: DataContainer
        """
        output_data_container = self.wrapped.handle_inverse_transform(
            DACT(
                ids=data_container._ids,
                data_inputs=(data_container._ids, data_container.data_inputs, data_container.expected_outputs),
                expected_outputs=None
            ),
            context.push(self.wrapped)
        )

        return output_data_container


class IdsAndInputAndOutputTransformerMixin(_DidProcessIdsInputOutputHandlerMixin):
    """
    Base output transformer step that can modify ids, data inputs, and expected_outputs at the same time.
    """

    def _transform_data_container(self, data_container: DACT, context: CX) -> DACT:
        """
        Handle inverse transform by updating the data inputs, and expected outputs inside the data container.

        :param context: execution context
        :param data_container:
        :return:
        """
        di_eo = (data_container._ids, data_container.data_inputs, data_container.expected_outputs)
        new_ids, new_data_inputs, new_expected_outputs = self.transform(di_eo)
        data_container.set_data_inputs((new_ids, new_data_inputs, new_expected_outputs))
        return data_container

    def _fit_data_container(self, data_container: DACT, context: CX) -> 'BaseStep':
        """
        Handle transform by fitting the step,
        and updating the data inputs, and expected outputs inside the data container.

        :param context: execution context
        :param data_container:
        :return:
        """
        new_self = self.fit((data_container._ids, data_container.data_inputs, data_container.expected_outputs), None)
        data_container.set_data_inputs(
            (data_container._ids, data_container.data_inputs, data_container.expected_outputs))
        data_container.set_expected_outputs(expected_outputs=None)  # TODO: check if this None eo is correct
        return new_self

    def _fit_transform_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple['BaseStep', DACT]:
        """
        Handle transform by fitting the step,
        and updating the data inputs, and expected outputs inside the data container.

        :param context: execution context
        :param data_container:
        :return:
        """
        new_self, (new_ids, new_data_inputs, new_expected_outputs) = self.fit_transform(
            (data_container._ids, data_container.data_inputs, data_container.expected_outputs), None)
        data_container.set_data_inputs((new_ids, new_data_inputs, new_expected_outputs))
        return new_self, data_container
