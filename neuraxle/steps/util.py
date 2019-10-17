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
from abc import ABC

VALUE_CACHING = 'value_caching'
from typing import List, Any

from neuraxle.base import BaseStep, NonFittableMixin, NonTransformableMixin, MetaStepMixin
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace


class BaseCallbackStep(BaseStep, ABC):
    """Base class for callback steps."""

    def __init__(self, callback_function, more_arguments: List = tuple(),
                 hyperparams=None, fit_callback_function=None, transform_function=None):
        """
        Create the callback step with a function and extra arguments to send to the function

        :param callback_function: The function that will be called on events.
        :param more_arguments: Extra arguments that will be sent to the callback after the processed data (optional).
        """
        super().__init__(hyperparams=hyperparams)
        self.transform_function = transform_function
        self.callback_function = callback_function
        self.fit_callback_function = fit_callback_function
        self.more_arguments = more_arguments

    def _fit_callback(self, data_inputs, expected_outputs):
        """
        Will call the self.fit_callback_function() with the data being processed and the extra arguments specified.
        It has no other effect.

        :param data_inputs: data inputs to fit
        :param expected_outputs: expected outputs to fit

        :return: self
        """
        self.fit_callback_function((data_inputs, expected_outputs), *self.more_arguments)

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
        if self.transform_function is not None:
            return self.transform_function(data_inputs)

        return data_inputs

    def inverse_transform(self, processed_outputs):
        """
        Will call the self._callback() with the data being processed and the extra arguments specified.
        It has no other effect.

        :param processed_outputs: the data to process
        :return: the same data as input, unchanged (like the Identity class).
        """
        self._callback(processed_outputs)
        return processed_outputs


class FitTransformCallbackStep(BaseStep):
    def __init__(self, transform_callback_function, fit_callback_function, more_arguments: List = tuple(),
                 transform_function=None,
                 hyperparams=None):
        BaseStep.__init__(self, hyperparams)
        self.transform_function = transform_function
        self.more_arguments = more_arguments
        self.fit_callback_function = fit_callback_function
        self.transform_callback_function = transform_callback_function

    def fit(self, data_inputs, expected_outputs=None):
        self.fit_callback_function((data_inputs, expected_outputs), *self.more_arguments)
        return self

    def transform(self, data_inputs):
        self.transform_callback_function(data_inputs, *self.more_arguments)
        if self.transform_function is not None:
            return self.transform_function(data_inputs)
        return data_inputs

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BaseStep', Any):
        self.fit_callback_function((data_inputs, expected_outputs), *self.more_arguments)
        self.transform_callback_function(data_inputs, *self.more_arguments)
        if self.transform_function is not None:
            return self, self.transform_function(data_inputs)

        return self, data_inputs


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
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        self.set_step(wrapped)
        self.steps: List[BaseStep] = []
        self.copy_op = copy_op

    def set_hyperparams(self, hyperparams: HyperparameterSamples) -> BaseStep:
        super().set_hyperparams(hyperparams)
        self.steps = [s.set_hyperparams(self.wrapped.hyperparams) for s in self.steps]
        return self

    def set_hyperparams_space(self, hyperparams_space: HyperparameterSpace) -> 'BaseStep':
        super().set_hyperparams_space(hyperparams_space)
        self.steps = [s.set_hyperparams_space(self.wrapped.hyperparams_space) for s in self.steps]
        return self

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BaseStep', Any):
        # One copy of step per data input:
        self.steps = [self.copy_op(self.wrapped) for _ in range(len(data_inputs))]

        if expected_outputs is None:
            expected_outputs = [None] * len(data_inputs)

        fit_transform_result = [self.steps[i].fit_transform(di, eo) for i, (di, eo) in
                                enumerate(zip(data_inputs, expected_outputs))]
        self.steps = [step for step, di in fit_transform_result]
        data_inputs = [di for step, di in fit_transform_result]

        return self, data_inputs

    def fit(self, data_inputs: List, expected_outputs: List = None) -> 'StepClonerForEachDataInput':
        # One copy of step per data input:
        self.steps = [self.copy_op(self.wrapped) for _ in range(len(data_inputs))]

        # Fit them all.
        if expected_outputs is None:
            expected_outputs = [None] * len(data_inputs)
        self.steps = [self.steps[i].fit(di, eo) for i, (di, eo) in enumerate(zip(data_inputs, expected_outputs))]

        return self

    def transform(self, data_inputs: List) -> List:
        return [self.steps[i].transform(di) for i, di in enumerate(data_inputs)]

    def inverse_transform(self, data_output):
        return [self.steps[i].inverse_transform(di) for i, di in enumerate(data_output)]


class DataShuffler:
    pass  # TODO.

