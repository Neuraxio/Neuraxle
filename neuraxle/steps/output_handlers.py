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
from neuraxle.base import ExecutionContext, BaseStep, MetaStepMixin
from neuraxle.data_container import DataContainer


class OutputTransformerWrapper(MetaStepMixin, BaseStep):
    """
    Transform expected output wrapper step that can sends the expected_outputs to the wrapped step
    so that it can transform the expected outputs.
    """

    def __init__(self, wrapped):
        MetaStepMixin.__init__(self, wrapped)
        BaseStep.__init__(self)

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
                current_ids=data_container.current_ids,
                data_inputs=data_container.expected_outputs,
                expected_outputs=None
            ),
            context
        )
        data_container.set_expected_outputs(new_expected_outputs_data_container.data_inputs)

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
                current_ids=data_container.current_ids,
                data_inputs=data_container.expected_outputs,
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
                current_ids=data_container.current_ids,
                data_inputs=data_container.expected_outputs,
                expected_outputs=None
            ),
            context
        )
        data_container.set_expected_outputs(new_expected_outputs_data_container.data_inputs)

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
                current_ids=data_container.current_ids,
                data_inputs=data_container.expected_outputs,
                expected_outputs=None
            ),
            context.push(self.wrapped)
        )

        data_container.set_expected_outputs(new_expected_outputs_data_container.data_inputs)

        current_ids = self.hash(data_container)
        data_container.set_current_ids(current_ids)

        return data_container

    def fit(self, data_inputs, expected_outputs=None):
        raise NotImplementedError('must be used inside a pipeline')

    def fit_transform(self, data_inputs, expected_outputs=None):
        raise NotImplementedError('must be used inside a pipeline')

    def transform(self, data_inputs):
        raise NotImplementedError('must be used inside a pipeline')

    def inverse_transform(self, processed_outputs):
        raise NotImplementedError('must be used inside a pipeline')


class InputAndOutputTransformerMixin:
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

        data_container.set_data_inputs(new_data_inputs)
        data_container.set_expected_outputs(new_expected_outputs)

        return data_container

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        """
        Handle transform by fitting the step,
        and updating the data inputs, and expected outputs inside the data container.

        :param context: execution context
        :param data_container:
        :return:
        """
        new_self = self.fit((data_container.data_inputs, data_container.expected_outputs), None)
        return new_self, data_container

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        """
        Handle transform by fitting the step,
        and updating the data inputs, and expected outputs inside the data container.

        :param context: execution context
        :param data_container:
        :return:
        """
        new_self, (new_data_inputs, new_expected_outputs) = \
            self.fit_transform((data_container.data_inputs, data_container.expected_outputs), None)

        data_container.set_data_inputs(new_data_inputs)
        data_container.set_expected_outputs(new_expected_outputs)

        return new_self, data_container

    def fit(self, data_inputs, expected_outputs=None):
        raise NotImplementedError('must be used inside a pipeline')

    def fit_transform(self, data_inputs, expected_outputs=None):
        raise NotImplementedError('must be used inside a pipeline')

    def transform(self, data_inputs):
        raise NotImplementedError('must be used inside a pipeline')

    def inverse_transform(self, processed_outputs):
        raise NotImplementedError('must be used inside a pipeline')
