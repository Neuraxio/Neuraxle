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
from operator import itemgetter
from typing import Callable, List, Optional, Tuple, Union

from neuraxle.base import (CX, DACT, BaseStep, BaseTransformer, ForceHandleMixin, ForceHandleOnlyMixin, Identity,
                           MetaStep, NamedStepsList, TruncableJoblibStepSaver)
from neuraxle.data_container import DIT, EOT, IDT, DACTData, ListDataContainer
from neuraxle.steps.flow import ExecuteIf

import numpy as np


class ForEach(ForceHandleOnlyMixin, MetaStep):
    """
    Truncable step that fits/transforms each step for each of the data inputs, and expected outputs.

    .. seealso::
        :class:`~neuraxle.base.BaseStep`,
        :class:`~neuraxle.base.BaseSaver`,
        :class:`~neuraxle.base.BaseHasher`,
        :class:`~neuraxle.base.NonTransformableMixin`,
        :class:`~neuraxle.pipeline.Pipeline`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def __init__(self, wrapped: BaseTransformer, cache_folder_when_no_handle=None):
        MetaStep.__init__(self, wrapped)
        ForceHandleOnlyMixin.__init__(self, cache_folder_when_no_handle)

    def _fit_data_container(self, data_container: DACT, context: CX) -> BaseStep:
        """
        Fit each step for each data inputs, and expected outputs

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: self
        """
        for i, (_id, di, eo) in enumerate(data_container):
            try:
                # TODO: DataDescriptor object.
                self.wrapped = self.wrapped.handle_fit(
                    DACT(data_inputs=di, ids=f"{_id}_{i}", expected_outputs=eo),
                    context
                )
            except ContinueInterrupt:
                continue
            except BreakInterrupt:
                break
        return self

    def _transform_data_container(self, data_container: DACT, context: CX) -> DACT:
        """
        Transform each step for each data inputs.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: self
        """
        output_data_container: ListDataContainer = ListDataContainer.empty(original_data_container=data_container)

        for _id, di, eo in data_container:
            try:
                output = self.wrapped.handle_transform(
                    DACT(data_inputs=di, ids=None, expected_outputs=eo),
                    context
                )

                output_data_container.append(
                    _id,
                    output.data_inputs,
                    output.expected_outputs
                )
            except ContinueInterrupt:
                continue
            except BreakInterrupt:
                break

        return output_data_container

    def _fit_transform_data_container(self, data_container: DACT, context: CX) -> Tuple[
            BaseStep, DACT]:
        """
        Fit transform each step for each data inputs, and expected outputs

        :param data_container: data container to fit transform
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext

        :return: self, transformed_data_container
        """
        output_data_container: DACT = ListDataContainer.empty(original_data_container=data_container)

        for current_id, di, eo in data_container:
            try:
                self.wrapped, output = self.wrapped.handle_fit_transform(
                    DACT(data_inputs=di, ids=None, expected_outputs=eo),
                    context
                )
                output_data_container.append(
                    current_id,
                    output.data_inputs,
                    output.expected_outputs
                )
            except ContinueInterrupt:
                continue
            except BreakInterrupt:
                break

        return self, output_data_container


class ContinueInterrupt(Exception):
    """This exception is used to signal to the minibatch iterator to skip the rest of the execution of the current iteration."""
    pass


class BreakInterrupt(Exception):
    """This exception is used to signal the interruption of a minibatch"""
    pass


class Break(ForceHandleMixin, Identity):
    def __init__(self):
        Identity.__init__(self)

    def _did_process(self, data_container: DACT, context: CX) -> DACT:
        raise BreakInterrupt()


class BreakIf(ExecuteIf):
    def __init__(self, condition_function: Callable):
        ExecuteIf.__init__(self, condition_function, Break())


class Continue(ForceHandleMixin, Identity):
    def __init__(self):
        Identity.__init__(self)

    def _did_process(self, data_container: DACT, context: CX) -> DACT:
        raise ContinueInterrupt()


class ContinueIf(ExecuteIf):
    def __init__(self, condition_function: Callable):
        ExecuteIf.__init__(self, condition_function, Continue())


class StepClonerForEachDataInput(ForceHandleOnlyMixin, MetaStep):
    def __init__(self, wrapped: BaseTransformer, copy_op=copy.deepcopy):
        MetaStep.__init__(self, wrapped=wrapped)
        ForceHandleOnlyMixin.__init__(self)
        self.savers.append(TruncableJoblibStepSaver())

        self.steps_as_tuple: List[NamedStepsList] = []
        self.copy_op = copy_op

    def get_children(self) -> List[BaseStep]:
        """
        Get the list of all the children for that step.

        :return: list of children. The first is the original wrapped step, the others are the steps that are cloned.
        """
        wrapped: List[BaseStep] = [self.get_step()]
        cloned_children = [step for _, step in self.steps_as_tuple]
        wrapped.extend(cloned_children)
        return wrapped

    def _will_process(
        self, data_container: DACT, context: CX
    ) -> Tuple['BaseStep', DACT]:
        data_container, context = super()._will_process(data_container, context)

        if len(self.steps_as_tuple) != len(data_container.data_inputs):
            self._copy_one_step_per_data_input(data_container)

        return data_container, context

    def _copy_one_step_per_data_input(self, data_container):
        # One copy of step per data input:
        steps = [self.copy_op(self.wrapped).set_name('{}[{}]'.format(self.wrapped.name, i)) for i in
                 range(len(data_container))]
        self.steps_as_tuple = [(step.name, step) for step in steps]
        self._invalidate()

    def _fit_transform_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple[BaseStep, DACT]:
        fitted_steps_data_containers = []
        for i, (ids, data_inputs, expected_outputs) in enumerate(data_container):
            fitted_step_data_container = self[i].handle_fit_transform(
                DACT(ids=ids, data_inputs=data_inputs, expected_outputs=expected_outputs),
                context
            )
            fitted_steps_data_containers.append(fitted_step_data_container)

        self.steps_as_tuple = [(step.name, step) for step, _ in fitted_steps_data_containers]

        output_data_container = ListDataContainer.empty()
        for _, data_container_batch in fitted_steps_data_containers:
            output_data_container.append_data_container(data_container_batch)

        return self, output_data_container

    def _fit_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple['BaseStep', DACT]:
        fitted_steps = []
        for i, (ids, data_inputs, expected_outputs) in enumerate(data_container):
            fitted_step = self[i].handle_fit(
                DACT(ids=ids, data_inputs=data_inputs, expected_outputs=expected_outputs),
                context
            )
            fitted_steps.append(fitted_step)

        self.steps_as_tuple = [(step.name, step) for step in fitted_steps]

        return self

    def _transform_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple['BaseStep', DACT]:
        transform_results = []
        for i, (ids, data_inputs, expected_outputs) in enumerate(data_container):
            transform_result = self[i].handle_transform(
                DACT(ids=ids, data_inputs=data_inputs, expected_outputs=expected_outputs),
                context
            )
            transform_results.append(transform_result)

        output_data_container = ListDataContainer.empty()
        for data_container_batch in transform_results:
            output_data_container.append_data_container(data_container_batch)
        return output_data_container

    def _inverse_transform_data_container(self, data_container: DACT, context: CX) -> DACT:
        inverse_transform_results = []
        for i, (ids, data_inputs, expected_outputs) in enumerate(data_container):
            inverse_transform_result = self[i].handle_inverse_transform(
                DACT(ids=ids, data_inputs=data_inputs, expected_outputs=expected_outputs),
                context
            )
            inverse_transform_results.append(inverse_transform_result)

        output_data_container = ListDataContainer.empty()
        for data_container_batch in inverse_transform_results:
            output_data_container.append_data_container(data_container_batch)
        return output_data_container

    def inverse_transform(self, processed_outputs):
        return [self[i].inverse_transform(di) for i, di in enumerate(processed_outputs)]

    def __getitem__(self, item):
        """
        Get cloned step at the given index.

        :return: iter(self.steps_as_tuple)
        """
        return self.steps_as_tuple[item][1]

    def __iter__(self):
        """
        Iterate through the steps.

        :return: iter(self.steps_as_tuple)
        """
        return iter(self.steps_as_tuple)

    def __len__(self):
        """
        Get number of steps cloned for each data input.

        :return: len(self.steps_as_tuple)
        """
        return len(self.steps_as_tuple)

    def items(self):
        return copy.copy(self.steps_as_tuple)

    def values(self):
        return list(map(itemgetter(1), self.steps_as_tuple))

    def keys(self):
        return list(map(itemgetter(0), self.steps_as_tuple))


lens = int  # Lengths of the dact data items that was pre-flattening


class FlattenForEach(ForceHandleMixin, MetaStep):
    """
    Step that reduces a dimension instead of manually looping on it.

    Using this step is equivalent to doing `sum(dact_data, [])` for each dact_data
    that might be a IDT, DIT, or EOT of a DACT (DAtaConTainer) to flatten the data,
    then the data is by default unflattened at the end of the loop in _did_process.
    """

    def __init__(
            self,
            wrapped: BaseTransformer,
            then_unflatten: bool = True
    ):
        MetaStep.__init__(self, wrapped)
        ForceHandleMixin.__init__(self)

        self.then_unflatten = then_unflatten

        self.spare_ids: Optional[IDT] = None
        # Lengths temporarily stored in _will_process to be able to unflatten the data container in _did_process:
        self.len_ids: List[lens] = []
        self.len_di: List[lens] = []
        self.len_eo: List[lens] = []

    def _will_process(
        self, data_container: DACT[Optional[Union[IDT, List[IDT]]], Optional[List[DIT]], Optional[List[EOT]]], context: CX
    ) -> Tuple['BaseTransformer', DACT]:
        """
        Flatten data container before any processing is done on the wrapped step, using lists.
        """
        data_container, context = super()._will_process(data_container, context)

        di, self.len_di = self._flatten_list(data_container.data_inputs)
        eo, self.len_eo = self._flatten_list(data_container.expected_outputs)

        # If is ID and not iterable nested thing, treat them as a special case that replicates the DIT:
        if data_container._ids is not None and len(self.len_di) > 0 and all(isinstance(i, (str, int)) for i in data_container._ids):
            # TODO: this code could be put inside the _flatten_list function to avoid duplicating it,
            #           but it would be more complicated to implement as we'd need to consider the len_di
            #           for the ids and eo and add a spare_eo as well.
            self.spare_ids = data_container._ids
            _ids: List[Union[int, str]] = sum([
                copy.deepcopy([_id] * _count)
                for _id, _count
                in zip(self.spare_ids, self.len_di)
            ], [])
            self.len_ids = copy.copy(self.len_di)
        else:
            self.spare_ids = None
            _ids, self.len_ids = self._flatten_list(data_container._ids)

        flattened_data_container = DACT(
            ids=_ids,
            data_inputs=di,
            expected_outputs=eo,
            sub_data_containers=data_container.sub_data_containers
        )

        self._invariant(data_container, flattened_data_container)

        return flattened_data_container, context

    def _flatten_list(self, _data: Union[Optional[DACTData], List[DACTData]]) -> Tuple[Optional[DACTData], List[lens]]:
        """
        Flatten the first dimension of a list.

        :param list_to_flatten: list to flatten
        :return: flattened list, len flattened lists
        """
        if _data is None or all(v is None for v in _data):
            return None, []

        if len(_data) != 0:
            try:
                iter(_data)
            except TypeError:
                return _data, [1 for x in _data]

        _data = [list(x) for x in _data]
        len_list_to_flatten = [len(x) for x in _data]
        flattened_list = sum(_data, [])

        return flattened_list, len_list_to_flatten

    def _did_process(self, data_container: DACT, context: CX) -> DACT:
        """
        Reaugment the flattened data container back to its full original shape using lists.
        """
        reaug_dact = super()._did_process(data_container, context).copy()

        if self.then_unflatten:
            if reaug_dact._ids is not None and self.spare_ids is None:
                reaug_dact.set_ids(self._reaugment_list(reaug_dact._ids, self.len_ids))
            else:
                reaug_dact.set_ids(self.spare_ids)
            if reaug_dact.data_inputs is not None:
                reaug_dact.set_data_inputs(self._reaugment_list(reaug_dact.data_inputs, self.len_di))
            if reaug_dact.expected_outputs is not None:
                reaug_dact.set_expected_outputs(self._reaugment_list(reaug_dact.expected_outputs, self.len_eo))

        self._invariant(reaug_dact, data_container)

        self.spare_ids = None
        self.len_ids = []
        self.len_di = []
        self.len_eo = []

        return reaug_dact

    def _reaugment_list(self, _data: DACTData, flattened_dims_lengths: List[lens]) -> List[DACTData]:
        """
        Reaugment list with the flattened dimension lengths.
        """
        if not self.then_unflatten or _data is None:
            return _data

        reaugmented_list: List[DACTData] = []
        i = 0
        for list_length in flattened_dims_lengths:
            sub_list: DACTData = _data[i:i + list_length]
            reaugmented_list.append(sub_list)
            i += list_length

        return reaugmented_list

    def _invariant(self, augmented_dact, flattened_dact):
        """
        Data consitency checks.
        """
        _raise = False
        if flattened_dact.di is not None:
            if flattened_dact.eo is not None and len(flattened_dact.di) != len(flattened_dact.eo):
                if all(v is None for v in flattened_dact.eo):
                    flattened_dact.eo = [None] * len(flattened_dact.di)
                else:
                    _raise = True
            if flattened_dact._ids is not None and len(flattened_dact.di) != len(flattened_dact._ids):
                _raise = True
        if _raise:
            raise ValueError(
                f"FlattenForEach: Cannot flatten or unflatten data properly. Expected outputs has a "
                f"different len than data inputs for flattened DACT: {flattened_dact}, and for "
                f"augmented DACT: {augmented_dact}."
            )
