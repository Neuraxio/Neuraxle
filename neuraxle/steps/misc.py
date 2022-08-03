"""
Miscelaneous Pipeline Steps
====================================
You can find here misc. pipeline steps, for example, callbacks useful for debugging, testing, and so forth.

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

import random
import time
import uuid
from abc import ABC
from typing import Any, Callable, List, Optional, Tuple, Union

from neuraxle.base import BaseStep, BaseTransformer
from neuraxle.base import ExecutionContext as CX
from neuraxle.base import ForceHandleOnlyMixin, HandleOnlyMixin, MetaStep, NonFittableMixin
from neuraxle.data_container import DIT, EOT, DACTData
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.space import HyperparameterSamples, RecursiveDict

VALUE_CACHING = 'value_caching'


class AssertFalseStep(HandleOnlyMixin, BaseStep):
    """
    Assert False upon _transform_data_container and _fit_data_container.
    """

    def __init__(self, message: str = "This step should not fit nor transform."):
        BaseStep.__init__(self)
        self.message: str = message

    def _transform_data_container(self, data_container, context):
        self._assert(False, self.message, context)

    def _fit_data_container(self, data_container, context):
        self._assert(False, self.message, context)


NoneType = type(None)


class BaseCallbackStep(BaseStep, ABC):
    """Base class for callback steps."""

    def __init__(
        self,
        callback_function: Callable[[DACTData], NoneType],
        more_arguments: List[Any] = tuple(),
        hyperparams: HyperparameterSamples = None,
        fit_callback_function: Callable[[DACTData], NoneType] = None,
        transform_function: Callable[[DACTData], NoneType] = None
    ):
        """
        Create the callback step with a function and extra arguments to send to the function

        :param callback_function: The function that will be called on events.
        :param more_arguments: Extra arguments that will be star-sent (*) to the callback after the processed data.
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

    def fit(self, data_inputs, expected_outputs=None):
        return self

    def transform(self, data_inputs):
        return data_inputs

    def inverse_transform(self, processed_outputs):
        return processed_outputs


class FitCallbackStep(BaseCallbackStep):
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


class TransformCallbackStep(BaseCallbackStep):
    """Call a callback method on transform and inverse transform."""

    def fit_transform(self, data_inputs, expected_outputs=None) -> Tuple['BaseStep', Any]:
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
        self.fit_callback_function: Optional[Callable] = fit_callback_function
        self.transform_callback_function: Optional[Callable] = transform_callback_function

    def fit(self, data_inputs, expected_outputs=None):
        self.fit_callback_function((data_inputs, expected_outputs), *self.more_arguments)
        return self

    def transform(self, data_inputs):
        self.transform_callback_function(data_inputs, *self.more_arguments)
        if self.transform_function is not None:
            return self.transform_function(data_inputs)
        return data_inputs

    def fit_transform(self, data_inputs, expected_outputs=None) -> Tuple['BaseStep', Any]:
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

        return RecursiveDict(cleared_callbacks)


class CallbackWrapper(HandleOnlyMixin, MetaStep):
    """
    A step that calls a callback function for each of his methods: transform, fit, fit_transform, and even inverse_transform.
    To be used with :class:`TapeCallbackFunction` most of the time, passed in the constructor.

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
            transform_callback_function: Union['TapeCallbackFunction', Callable],
            fit_callback_function: Union['TapeCallbackFunction', Callable],
            inverse_transform_callback_function: Union['TapeCallbackFunction', Callable] = None,
            more_arguments: List = tuple(),
            hyperparams=None
    ):
        MetaStep.__init__(self, wrapped=wrapped, hyperparams=hyperparams)

        self.inverse_transform_callback_function = inverse_transform_callback_function
        self.more_arguments = more_arguments
        self.fit_callback_function = fit_callback_function
        self.transform_callback_function = transform_callback_function

    def _fit_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple['BaseStep', DACT]:
        """
        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: step, data_container
        """
        self.fit_callback_function((data_container.data_inputs, data_container.expected_outputs), *self.more_arguments)
        self.wrapped = self.wrapped.handle_fit(data_container, context)
        return self

    def _fit_transform_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple['BaseStep', DACT]:
        """
        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: step, data_container
        """
        self.fit_callback_function((data_container.data_inputs, data_container.expected_outputs), *self.more_arguments)
        self.transform_callback_function(data_container.data_inputs, *self.more_arguments)
        self.wrapped, data_container = self.wrapped.handle_fit_transform(data_container, context)
        return self, data_container

    def _transform_data_container(self, data_container: DACT, context: CX) -> DACT:
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

    def handle_inverse_transform(self, data_container: DACT, context: CX) -> DACT:
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
        self.data: List[DACTData] = []  # at each time the callback is called, data is appened.
        self.name_tape: List[str] = []

    def __call__(self, *args, **kwargs):
        return self.callback(*args, **kwargs)

    def callback(self, data: Tuple[DACTData, ...], name: str = ""):
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


class HandleCallbackStep(ForceHandleOnlyMixin, BaseStep):
    def __init__(
            self,
            handle_fit_callback,
            handle_transform_callback,
            handle_fit_transform_callback
    ):
        BaseStep.__init__(self)
        ForceHandleOnlyMixin.__init__(self)

        self.handle_fit_callback = handle_fit_callback
        self.handle_fit_transform_callback = handle_fit_transform_callback
        self.handle_transform_callback = handle_transform_callback

    def _fit_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple['BaseStep', DACT]:
        self.handle_fit_callback((data_container, context))
        return self, data_container

    def _transform_data_container(
        self, data_container: DACT, context: CX
    ) -> DACT:
        self.handle_transform_callback((data_container, context))
        return data_container

    def _fit_transform_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple['BaseStep', DACT]:
        self.handle_fit_transform_callback((data_container, context))
        return self, data_container


class Sleep(BaseTransformer):
    def __init__(self, sleep_time: float = 0.1, add_random_quantity: float = 0.0):
        """
        Sleep for a given time, given in seconds.
        """
        BaseTransformer.__init__(self)
        self.sleep_time = sleep_time
        self.add_random_quantity = add_random_quantity

    def transform(self, data_inputs):
        seconds = (
            self.sleep_time
            if self.add_random_quantity == 0.0 else
            self.sleep_time + random.random() * self.add_random_quantity
        )
        time.sleep(seconds)
        return data_inputs


class FitTransformCounterLoggingStep(HandleOnlyMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        HandleOnlyMixin.__init__(self)
        self.logging_call_counter = 0

    def _fit_data_container(self, data_container: DACT, context: CX) -> BaseStep:
        self._log(context, "fit")
        return self

    def _transform_data_container(self, data_container: DACT, context: CX) -> DACT:
        self._log(context, "transform")
        return data_container

    def _fit_transform_data_container(self, data_container: DACT, context: CX) -> DACT:
        self._log(context, "fit_transform")
        return self, data_container

    def _log(self, context, func_name):
        context.logger.info(
            f"{self.name} - {func_name} call - logging call #{self.logging_call_counter} with UUID={uuid.uuid4()}")
        self.logging_call_counter += 1


class TransformOnlyCounterLoggingStep(NonFittableMixin, FitTransformCounterLoggingStep):
    def __init__(self):
        FitTransformCounterLoggingStep.__init__(self)
        NonFittableMixin.__init__(self)
