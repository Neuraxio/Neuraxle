"""
Util Pipeline Steps
====================================
You can find here misc. pipeline steps, for example, callbacks.
"""
import copy
from abc import ABC
from typing import List

from neuraxle.base import BaseStep, NonFittableMixin, NonTransformableMixin, MetaStepMixin


class BaseCallbackStep(BaseStep, ABC):
    """Base class for callback steps."""

    def __init__(self, callback_function, more_arguments: List = tuple()):
        """
        Create the callback step with a function and extra arguments to send to the function

        :param callback_function: The function that will be called on events.
        :param more_arguments: Extra arguments that will be sent to the callback after the processed data (optional).
        """
        super().__init__()
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

    def fit(self, data_inputs, expected_outputs=None):
        """
        Will call the self._callback() with the data being processed and the extra arguments specified.
        Note that here, the data to process is packed into a tuple of (data_inputs, expected_outputs).
        It has no other effect.

        :param data_inputs: the data to process
        :param expected_outputs: the data to process
        :return: self
        """
        self._callback((data_inputs, expected_outputs))

    def fit_one(self, data_input, expected_output=None):
        """
        Will call the self._callback() with the data being processed and the extra arguments specified.
        Note that here, the data to process is packed into a tuple of (data_input, expected_output).
        It has no other effect.

        :param data_input: the data to process
        :param expected_output: the data to process
        :return: self
        """
        self._callback((data_input, expected_output))


class TransformCallbackStep(NonFittableMixin, BaseCallbackStep):
    """Call a callback method on transform and inverse transform."""

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

    ```
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
    ```
    """

    def __init__(self):
        """Initialize the tape (cache lists)."""
        self.data: List = []
        self.name_tape: List[str] = []

    def callback(self, data, name: str):
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
    def __init__(self, wrapped: BaseStep):
        # TODO: set params on wrapped.
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self)
        self.set_step(wrapped)

    def fit(self, data_inputs: List, expected_outputs: List = None) -> BaseStep:
        # One copy of step per data input:
        self.step: List[BaseStep] = [copy.deepcopy(self.step) for _ in range(len(data_inputs))]

        # Fit them all.
        self.step = [self.step[i].fit(di) for i, di in enumerate(data_inputs)]

        return self

    def transform(self, data_inputs: List) -> List:
        # As many data inputs than we have cloned steps: each transforms the one it has.
        assert len(data_inputs) <= len(self.step), "Can't have more data_inputs than cloned steps to process them."

        return [self.step[i].transform(di) for i, di in enumerate(data_inputs)]


class DataShuffler:
    pass  # TODO.