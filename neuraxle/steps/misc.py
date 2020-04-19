"""
Miscelaneous Pipeline Steps
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

..
    Thanks to Umaneo Technologies Inc. for their contributions to this Machine Learning
    project, visit https://www.umaneo.com/ for more information on Umaneo Technologies Inc.

"""

import time
from abc import ABC

VALUE_CACHING = 'value_caching'
from typing import List, Any

from neuraxle.base import BaseStep, NonFittableMixin, NonTransformableMixin, ExecutionContext, MetaStepMixin, \
    HandleOnlyMixin
from neuraxle.data_container import DataContainer


class BaseCallbackStep(BaseStep, ABC):
    """Base class for callback steps."""

    def __init__(self, callback_function, more_arguments: List = tuple(),
                 hyperparams=None, fit_callback_function=None, transform_function=None):
        """
        Create the callback step with a function and extra arguments to send to the function

        :param callback_function: The function that will be called on events.
        :param more_arguments: Extra arguments that will be sent to the callback after the processed data (optional).
        """
        BaseStep.__init__(self, hyperparams=hyperparams)
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

        if self.transform_function is not None:
            return self, self.transform_function(data_inputs)
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
    def __init__(self, transform_callback_function=None, fit_callback_function=None, more_arguments: List = tuple(),
                 transform_function=None,
                 hyperparams=None):
        BaseStep.__init__(self, hyperparams)
        if transform_callback_function is None:
            transform_callback_function = TapeCallbackFunction()
        if fit_callback_function is None:
            fit_callback_function = TapeCallbackFunction()

        self.transform_function = transform_function
        self.more_arguments = more_arguments

        if transform_callback_function is None:
            transform_callback_function = TapeCallbackFunction()

        if fit_callback_function is None:
            fit_callback_function = TapeCallbackFunction()

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

    def inverse_transform(self, processed_outputs):
        return processed_outputs

    def clear_callbacks(self):
        cleared_callbacks = {
           self.name: {
               'transform': self.transform_callback_function.data,
               'fit': self.fit_callback_function.data
           }
        }

        self.transform_callback_function.data = []
        self.transform_callback_function.name_tape = []

        self.fit_callback_function.data = []
        self.fit_callback_function.name_tape = []

        return cleared_callbacks


class CallbackWrapper(HandleOnlyMixin, MetaStepMixin, BaseStep):
    """
    A step that calls a callback function for each of his methods : transform, fit, fit_transform, and even inverse_transform.
    To be used with :class:`TapeCallbackFunction`.

    .. code-block:: python

        tape_fit = TapeCallbackFunction()
        tape_transform = TapeCallbackFunction()
        tape_inverse_transform = TapeCallbackFunction()

        callback_wrapper = CallbackWrapper(MultiplyByN(2), tape_transform_preprocessing, tape_fit_preprocessing, tape_inverse_transform_preprocessing)

    .. seealso::
        :class:`~neuraxle.base.HandleOnlyMixin`,
        :class:`~neuraxle.base.MetaStepMixin`,
        :class:`~neuraxle.base.BaseStep`
    """
    def __init__(
            self,
            wrapped,
            transform_callback_function,
            fit_callback_function,
            inverse_transform_callback_function=None,
            more_arguments: List = tuple(),
            hyperparams=None
    ):
        BaseStep.__init__(self, hyperparams)
        MetaStepMixin.__init__(self, wrapped)

        self.inverse_transform_callback_function = inverse_transform_callback_function
        self.more_arguments = more_arguments
        self.fit_callback_function = fit_callback_function
        self.transform_callback_function = transform_callback_function

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        """
        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: step, data_container
        :type: (BaseStep, DataContainer)
        """
        self.fit_callback_function((data_container.data_inputs, data_container.expected_outputs), *self.more_arguments)
        self.wrapped = self.wrapped.handle_fit(data_container, context)
        return self

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        """
        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: step, data_container
        :type: (BaseStep, DataContainer)
        """
        self.fit_callback_function((data_container.data_inputs, data_container.expected_outputs), *self.more_arguments)
        self.transform_callback_function(data_container.data_inputs, *self.more_arguments)
        self.wrapped, data_container = self.wrapped.handle_fit_transform(data_container, context)
        return self, data_container

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: step, data_container
        :type: DataContainer
        """
        self.transform_callback_function(data_container.data_inputs, *self.more_arguments)
        return self.wrapped.handle_transform(data_container, context.push(self.wrapped))

    def handle_inverse_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        :param context: execution context
        :type context: ExecutionContext
        :param data_container: data containerj
        :type data_container: DataContainer
        :return: data container
        :rtype: DataContainer
        """
        self.inverse_transform_callback_function(data_container.data_inputs, *self.more_arguments)
        data_container = self.wrapped.handle_inverse_transform(data_container, context.push(self.wrapped))
        current_ids = self.hash(data_container)
        data_container.set_current_ids(current_ids)

        return data_container


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

    def reset(self):
        """
        Reset callback data.
        :return: None
        """
        self.data = []
        self.name_tape = []

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


class HandleCallbackStep(HandleOnlyMixin, BaseStep):
    def __init__(
            self,
            handle_fit_callback,
            handle_transform_callback,
            handle_fit_transform_callback
    ):
        HandleOnlyMixin.__init__(self)
        BaseStep.__init__(self)
        self.handle_fit_callback = handle_fit_callback
        self.handle_fit_transform_callback = handle_fit_transform_callback
        self.handle_transform_callback = handle_transform_callback

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        self.handle_fit_callback((data_container, context))
        return self, data_container

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        self.handle_transform_callback((data_container, context))
        return data_container

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        self.handle_fit_transform_callback((data_container, context))
        return self, data_container


class Sleep(NonFittableMixin, BaseStep):
    def __init__(self, sleep_time=0.1, hyperparams=None, hyperparams_space=None):
        BaseStep.__init__(self, hyperparams=hyperparams, hyperparams_space=hyperparams_space)
        self.sleep_time = sleep_time

    def transform(self, data_inputs):
        time.sleep(self.sleep_time)
        return data_inputs
