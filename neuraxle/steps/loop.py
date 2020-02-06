"""
Pipeline Steps For Looping
=====================================

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
from typing import List, Any

from neuraxle.base import MetaStepMixin, BaseStep, DataContainer, ExecutionContext, ResumableStepMixin, HandleOnlyMixin
from neuraxle.data_container import ListDataContainer
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace


class ForEachDataInput(HandleOnlyMixin, ResumableStepMixin, MetaStepMixin, BaseStep):
    """
    Truncable step that fits/transforms each step for each of the data inputs, and expected outputs.
    """

    def __init__(
            self,
            wrapped: BaseStep
    ):
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        HandleOnlyMixin.__init__(self)

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> BaseStep:
        """
        Fit each step for each data inputs, and expected outputs

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: self
        """
        for current_id, di, eo in data_container:
            self.wrapped = self.wrapped.handle_fit(
                DataContainer(data_inputs=di, current_ids=None, expected_outputs=eo),
                context
            )

        return self

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
        """
        Transform each step for each data inputs.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: self
        """
        output_data_container = ListDataContainer.empty(original_data_container=data_container)

        for current_id, di, eo in data_container:
            output = self.wrapped.handle_transform(
                DataContainer(data_inputs=di, current_ids=None, expected_outputs=eo),
                context
            )

            output_data_container.append(
                current_id,
                output.data_inputs,
                output.expected_outputs
            )
        output_data_container.summary_id = data_container.summary_id

        return output_data_container

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
        """
        Fit transform each step for each data inputs, and expected outputs

        :param data_container: data container to fit transform
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext

        :return: self, transformed_data_container
        """
        output_data_container = ListDataContainer.empty(original_data_container=data_container)

        for current_id, di, eo in data_container:
            self.wrapped, output = self.wrapped.handle_fit_transform(
                DataContainer(data_inputs=di, current_ids=None, expected_outputs=eo),
                context
            )

            output_data_container.append(
                current_id,
                output.data_inputs,
                output.expected_outputs
            )

        output_data_container.summary_id = data_container.summary_id

        return self, output_data_container

    def hash_data_container(self, data_container):
        output_data_container = self.wrapped.hash_data_container(data_container)
        output_data_container.summary_id = data_container.summary_id

        return output_data_container

    def should_resume(self, data_container: DataContainer, context: ExecutionContext):
        context = context.push(self)

        if isinstance(self.wrapped, ResumableStepMixin) and self.wrapped.should_resume(data_container, context):
            return True
        return False


class StepClonerForEachDataInput(HandleOnlyMixin, MetaStepMixin, BaseStep):
    def __init__(self, wrapped: BaseStep, copy_op=copy.deepcopy):
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        HandleOnlyMixin.__init__(self)

        self.set_step(wrapped)
        self.steps: List[BaseStep] = []
        self.copy_op = copy_op

    def set_hyperparams(self, hyperparams: HyperparameterSamples) -> BaseStep:
        MetaStepMixin.set_hyperparams(self, hyperparams)
        self.steps = [s.set_hyperparams(self.wrapped.get_hyperparams()) for s in self.steps]
        return self

    def update_hyperparams(self, hyperparams: HyperparameterSamples) -> BaseStep:
        """
        Update the step hyperparameters without removing the already-set hyperparameters.
        Please refer to :func:`~BaseStep.update_hyperparams`.

        :param hyperparams: hyperparams to update
        :type hyperparams: HyperparameterSamples
        :return: self
        :rtype: BaseStep

        .. seealso::
            :func:`~BaseStep.update_hyperparams`,
            :class:`HyperparameterSamples`
        """
        MetaStepMixin.update_hyperparams(self, hyperparams)
        self.steps = [s.set_hyperparams(self.wrapped.get_hyperparams()) for s in self.steps]
        return self

    def set_hyperparams_space(self, hyperparams_space: HyperparameterSpace) -> 'BaseStep':
        MetaStepMixin.set_hyperparams_space(self, hyperparams_space)
        self.steps = [s.set_hyperparams_space(self.wrapped.get_hyperparams_space()) for s in self.steps]
        return self

    def _will_process(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        data_container, context = BaseStep._will_process(self, data_container, context)

        if len(self.steps) != len(data_container.data_inputs):
            self._copy_one_step_per_data_input(data_container)

        return data_container, context

    def _copy_one_step_per_data_input(self, data_container):
        # One copy of step per data input:
        self.steps = [self.copy_op(self.wrapped) for _ in range(len(data_container))]
        self.is_invalidated = True

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        fitted_steps_data_containers = []
        for i, (current_ids, data_inputs, expected_outputs) in enumerate(data_container):
            fitted_step_data_container = self.steps[i].handle_fit_transform(
                DataContainer(current_ids=current_ids, data_inputs=data_inputs, expected_outputs=expected_outputs),
                context
            )

            fitted_steps_data_containers.append(fitted_step_data_container)

        self.steps = fitted_steps_data_containers
        self.steps = [step for step, _ in fitted_steps_data_containers]

        output_data_container = ListDataContainer.empty()
        [output_data_container.append_data_contianer(data_container_batch) for _, data_container_batch in fitted_steps_data_containers]

        return self, output_data_container.to_numpy()

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        fitted_steps = []
        for i, (current_ids, data_inputs, expected_outputs) in enumerate(data_container):
            fitted_step = self.steps[i].handle_fit(
                DataContainer(current_ids=current_ids, data_inputs=data_inputs, expected_outputs=expected_outputs),
                context
            )

            fitted_steps.append(fitted_step)

        self.steps = fitted_steps

        return self

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        transform_results = []
        for i, (current_ids, data_inputs, expected_outputs) in enumerate(data_container):
            transform_result = self.steps[i].handle_transform(
                DataContainer(current_ids=current_ids, data_inputs=data_inputs, expected_outputs=expected_outputs),
                context
            )

            transform_results.append(transform_result)

        output_data_container = ListDataContainer.empty()
        [output_data_container.append_data_contianer(data_container_batch) for data_container_batch in transform_results]

        return output_data_container.to_numpy()

    def _inverse_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        inverse_transform_results = []
        for i, (current_ids, data_inputs, expected_outputs) in enumerate(data_container):
            inverse_transform_result = self.steps[i].handle_inverse_transform(
                DataContainer(current_ids=current_ids, data_inputs=data_inputs, expected_outputs=expected_outputs),
                context
            )

            inverse_transform_results.append(inverse_transform_result)

        output_data_container = ListDataContainer.empty()
        [output_data_container.append_data_contianer(data_container_batch) for data_container_batch in inverse_transform_results]

        return output_data_container.to_numpy()
