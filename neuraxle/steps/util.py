"""
Util Pipeline Steps
====================================
You can find here misc. pipeline steps, for example, callbacks useful for debugging, and a step cloner.

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

import copy
from abc import ABC, abstractmethod
from typing import List, Any, Tuple

from neuraxle.base import BaseStep, NonFittableMixin, NonTransformableMixin, MetaStepMixin, DataContainer
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace


class BaseCallbackStep(BaseStep, ABC):
    """Base class for callback steps."""

    def __init__(self, callback_function, more_arguments: List = tuple(), hyperparams = None):
        """
        Create the callback step with a function and extra arguments to send to the function

        :param callback_function: The function that will be called on events.
        :param more_arguments: Extra arguments that will be sent to the callback after the processed data (optional).
        """
        super().__init__(hyperparams=hyperparams)
        self.callback_function = callback_function
        self.more_arguments = more_arguments

    def _callback(self, data):
        """
        Will call the self.callback_function() with the data being processed and the extra arguments specified.
        It has no other effect.

        :param data_inputs: the data to process
        :return: None
        """
        self.callback_function(data, *self.more_arguments)


class FitCallbackStep(NonTransformableMixin, BaseCallbackStep):
    """Call a callback method on fit."""

    def fit(self, data_inputs, expected_outputs=None) -> 'FitCallbackStep':
        """
        Will call the self._callback() with the data being processed and the extra arguments specified.
        Note that here, the data to process is packed into a tuple of (data_inputs, expected_outputs).
        It has no other effect.

        :param data_inputs: the data to process
        :param expected_outputs: the data to process
        :return: self
        """
        self._callback((data_inputs, expected_outputs))
        return self

    def fit_one(self, data_input, expected_output=None) -> 'FitCallbackStep':
        """
        Will call the self._callback() with the data being processed and the extra arguments specified.
        Note that here, the data to process is packed into a tuple of (data_input, expected_output).
        It has no other effect.

        :param data_input: the data to process
        :param expected_output: the data to process
        :return: self
        """
        self._callback((data_input, expected_output))
        return self


class TransformCallbackStep(NonFittableMixin, BaseCallbackStep):
    """Call a callback method on transform and inverse transform."""

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BaseStep', Any):
        self._callback(data_inputs)

        return self, data_inputs

    def transform(self, data_inputs):
        """
        Will call the self._callback() with the data being processed and the extra arguments specified.
        It has no other effect.

        :param data_inputs: the data to process
        :return: the same data as input, unchanged (like the Identity class).
        """
        self._callback(data_inputs)
        return data_inputs

    def transform_one(self, data_input):
        """
        Will call the self._callback() with the data being processed and the extra arguments specified.
        It has no other effect.

        :param data_input: the data to process
        :return: the same data as input, unchanged (like the Identity class).
        """
        self._callback(data_input)
        return data_input

    def inverse_transform(self, processed_outputs):
        """
        Will call the self._callback() with the data being processed and the extra arguments specified.
        It has no other effect.

        :param processed_outputs: the data to process
        :return: the same data as input, unchanged (like the Identity class).
        """
        self._callback(processed_outputs)
        return processed_outputs

    def inverse_transform_one(self, processed_output):
        """
        Will call the self._callback() with the data being processed and the extra arguments specified.
        It has no other effect.

        :param processed_output: the data to process
        :return: the same data as input, unchanged (like the Identity class).
        """
        self._callback(processed_output)
        return processed_output


class TapeCallbackFunction:
    """This class's purpose is to be sent to the callback to accumulate information.

    Example usage:

    .. code-block:: python

        expected_tape = ["1", "2", "3", "a", "b", "4"]
        tape = TapeCallbackFunction()

        p = Pipeline([
            Identity(),
            TransformCallbackStep(tape.callback, ["1"]),
            TransformCallbackStep(tape.callback, ["2"]),
            TransformCallbackStep(tape.callback, ["3"]),
            AddFeatures([
                TransformCallbackStep(tape.callback, ["a"]),
                TransformCallbackStep(tape.callback, ["b"]),
            ]),
            TransformCallbackStep(tape.callback, ["4"]),
            Identity()
        ])
        p.fit_transform(np.ones((1, 1)))

        assert expected_tape == tape.get_name_tape()

    """

    def __init__(self):
        """Initialize the tape (cache lists)."""
        self.data: List = []
        self.name_tape: List[str] = []

    def __call__(self, *args, **kwargs):
        return self.callback(*args, **kwargs)

    def callback(self, data, name: str = ""):
        """
        Will stick the data and name to the tape.

        :param data: data to save
        :param name: name to save (string)
        :return: None
        """
        self.data.append(data)
        self.name_tape.append(name)

    def get_data(self) -> List:
        """
        Get the data tape

        :return: The list of data.
        """
        return self.data

    def get_name_tape(self) -> List[str]:
        """
        Get the data tape

        :return: The list of names.
        """
        return self.name_tape


class StepClonerForEachDataInput(MetaStepMixin, BaseStep):
    def __init__(self, wrapped: BaseStep, copy_op=copy.deepcopy):
        # TODO: set params on wrapped.
        # TODO: use MetaStep*s*Mixin (plural) and review.
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self)
        self.set_step(wrapped)
        self.steps: List[BaseStep] = []
        self.copy_op = copy_op

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BaseStep', Any):
        # One copy of step per data input:
        self.steps = [self.copy_op(self.step) for _ in range(len(data_inputs))]

        if expected_outputs is None:
            expected_outputs = [None] * len(data_inputs)

        fit_transform_result = [self.steps[i].fit_transform(di, eo) for i, (di, eo) in enumerate(zip(data_inputs, expected_outputs))]
        self.steps = [step for step, di in fit_transform_result]
        data_inputs = [di for step, di in fit_transform_result]

        return self, data_inputs

    def fit(self, data_inputs: List, expected_outputs: List = None) -> 'StepClonerForEachDataInput':
        # One copy of step per data input:
        self.steps = [self.copy_op(self.step) for _ in range(len(data_inputs))]

        # Fit them all.
        if expected_outputs is None:
            expected_outputs = [None] * len(data_inputs)
        self.steps = [self.steps[i].fit(di, eo) for i, (di, eo) in enumerate(zip(data_inputs, expected_outputs))]

        return self

    def transform(self, data_inputs: List) -> List:
        assert len(data_inputs) >= len(self.steps)

        return [self.steps[i].transform(di) for i, di in enumerate(data_inputs)]

    def inverse_transform(self, data_output):
        assert len(data_output) >= len(self.steps)

        return [self.steps[i].inverse_transform(di) for i, di in enumerate(data_output)]

    def set_hyperparams(self, hyperparams: HyperparameterSamples) -> BaseStep:
        self.step = self.step.set_hyperparams(hyperparams.to_flat())
        return self

    def get_hyperparams(self) -> HyperparameterSamples:
        return self.step.get_hyperparams()

    def set_hyperparams_space(self, hyperparams_space: HyperparameterSpace) -> 'BaseStep':
        self.step = self.step.set_hyperparams_space(hyperparams_space.to_flat())
        return self

    def get_hyperparams_space(self) -> HyperparameterSpace:
        return self.step.get_hyperparams_space()

    def reverse(self) -> 'BaseStep':
        """
        The object will mutate itself such that the ``.transform`` method (and of all its underlying objects
        if applicable) be replaced by the ``.inverse_transform`` method.

        Note: the reverse may fail if there is a pending mutate that was set earlier with ``.will_mutate_to``.

        :return: a copy of self, reversed. Each contained object will also have been reversed if self is a pipeline.
        """
        for step in self.steps:
            step.mutate(new_method="inverse_transform", method_to_assign_to="transform")
        return self

    def __reversed__(self) -> 'BaseStep':
        """
        The object will mutate itself such that the ``.transform`` method (and of all its underlying objects
        if applicable) be replaced by the ``.inverse_transform`` method.

        Note: the reverse may fail if there is a pending mutate that was set earlier with ``.will_mutate_to``.

        :return: a copy of self, reversed. Each contained object will also have been reversed if self is a pipeline.
        """
        return self.reverse()


class DataShuffler:
    pass  # TODO.


class OutputTransformerMixin:
    """
    Mixin to be able to modify expected_outputs inside a step
    """

    @abstractmethod
    def transform_input_output(self, data_inputs, expected_outputs=None) -> Tuple[Any, Any]:
        """
        Transform data inputs, and expected outputs at the same time

        :param data_inputs:
        :param expected_outputs:
        :return: tuple(data_inputs, expected_outputs)
        """
        raise NotImplementedError()


class OutputTransformerWrapper(MetaStepMixin, BaseStep):
    """
    Output transformer wrapper wraps a step that inherits OutputTransformerMixin,
    and updates the data inputs, and expected outputs for each transform.
    """
    def __init__(self, wrapped: BaseStep):
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)

    def handle_transform(self, data_container: DataContainer) -> DataContainer:
        """
        Handle transform by updating the data inputs, and expected outputs inside the data container.

        :param data_container:
        :return:
        """
        new_data_inputs, new_expected_outputs = self.wrapped.transform_input_output(
            data_inputs=data_container.data_inputs, expected_outputs=data_container.expected_outputs
        )
        data_container.set_data_inputs(new_data_inputs)
        data_container.set_expected_outputs(new_expected_outputs)

        current_ids = self.hasher.hash(data_container.current_ids, self.hyperparams, data_container.data_inputs)
        data_container.set_current_ids(current_ids)

        return data_container

    def handle_fit_transform(self, data_container: DataContainer) -> ('BaseStep', DataContainer):
        """
        Handle transform by fitting the step,
        and updating the data inputs, and expected outputs inside the data container.

        :param data_container:
        :return:
        """
        new_self = self.wrapped.fit(data_container.data_inputs, data_container.expected_outputs)

        data_container = self.handle_transform(data_container)

        return new_self, data_container