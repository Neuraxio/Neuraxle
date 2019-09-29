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
import hashlib
import os
import pickle
import shutil
from abc import ABC, abstractmethod
from typing import Iterable

from neuraxle.pipeline import PipelineSaver, DEFAULT_CACHE_FOLDER

VALUE_CACHING = 'value_caching'
from typing import List, Any

from neuraxle.base import BaseStep, NonFittableMixin, NonTransformableMixin, MetaStepMixin, DataContainer
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


class FitTransformCallbackStep(BaseCallbackStep):
    """Call a callback method on transform and inverse transform."""

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        self._fit_callback(data_inputs, expected_outputs)
        return self

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

    def transform_one(self, data_input):
        """
        Will call the self._callback() with the data being processed and the extra arguments specified.
        It has no other effect.

        :param data_input: the data to process
        :return: the same data as input, unchanged (like the Identity class).
        """
        self._callback(data_input)

        if self.transform_function is not None:
            return self.transform_function([data_input])[0]

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

    def transform_one(self, data_input):
        """
        Will call the self._callback() with the data being processed and the extra arguments specified.
        It has no other effect.

        :param data_input: the data to process
        :return: the same data as input, unchanged (like the Identity class).
        """
        self._callback(data_input)

        if self.transform_function is not None:
            return self.transform_function([data_input])[0]

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


class IdentityPipelineSaver(PipelineSaver):
    def can_load(self, pipeline: 'Pipeline', data_container: DataContainer) -> bool:
        return True

    def save(self, pipeline: 'Pipeline', data_container: DataContainer) -> 'Pipeline':
        return pipeline

    def load(self, pipeline: 'Pipeline', data_container: DataContainer) -> 'Pipeline':
        return pipeline


class ValueCachingWrapper(MetaStepMixin, BaseStep):
    """
    Value caching wrapper wraps a step that inherits OutputTransformerMixin,
    """

    def __init__(self, wrapped: BaseStep, cache_folder: str = DEFAULT_CACHE_FOLDER):
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        self.cache_folder = cache_folder

    def setup(self, step_path: str, setup_arguments: dict = None):
        """
        Fit transform data container using value caching.

        :param setup_arguments: optional additional setup arguments
        :type setup_arguments: dict

        :param step_path: path of the step in the pipeline ex: `̀pipeline/step_name/`̀
        :type step_path: str

        :return: tuple(fitted pipeline, data_container)
        """
        self.create_checkpoint_path(step_path)

    def handle_fit_transform(self, data_container: DataContainer) -> ('BaseStep', DataContainer):
        """
        Fit transform data container.

        :param data_container: the data container to transform
        :type data_container: DataContainer

        :return: tuple(fitted pipeline, data_container)
        """
        self.flush_cache()
        self.wrapped = self.wrapped.fit(data_container.data_inputs, data_container.expected_outputs)
        outputs = self._transform_with_cache(data_container)

        data_container.set_data_inputs(outputs)

        current_ids = self.hash(data_container.current_ids, self.hyperparams, outputs)
        data_container.set_current_ids(current_ids)

        return self, data_container

    def handle_transform(self, data_container: DataContainer) -> DataContainer:
        """
        Transform data container.

        :param data_container: the data container to transform
        :type data_container: DataContainer

        :return: transformed data container
        """
        outputs = self._transform_with_cache(data_container)

        data_container.set_data_inputs(outputs)

        current_ids = self.hash(data_container.current_ids, self.hyperparams, outputs)
        data_container.set_current_ids(current_ids)

        return data_container

    def _hash_value(self, data_input):
        m = hashlib.md5()
        m.update(str.encode(str(data_input)))

        return m.hexdigest()

    def _transform_with_cache(self, data_container: DataContainer) -> Iterable:
        """
        Transform data container using value caching.

        :param data_container: the data container to transform
        :type data_container: DataContainer

        :return: iterable
        """
        outputs = []
        for current_id, data_input, expected_output in data_container:
            if self.contains_cache_for(data_input):
                outputs.extend(self.read_cache(data_input))
            else:
                output = self.wrapped.transform([data_input])
                self.write_cache(data_input, output)
                outputs.extend(output)
        return outputs

    @abstractmethod
    def create_checkpoint_path(self, step_path: str) -> str:
        """
        Create checkpoint path.

        :param step_path: step path inside pipeline ex: ``Pipeline/step_name/`` 
        :type step_path: str

        :return: checkpoint path
        """
        raise NotImplementedError()

    @abstractmethod
    def flush_cache(self):
        """
        Flush all cached values
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def read_cache(self, data_input) -> Any:
        """
        Read cache for a given data input.

        :param data_input: data input to get cache for
        :type data_input: Any

        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def write_cache(self, data_input, output):
        """
        Write cache for a given data input and output.

        :param data_input: data input to write cache for
        :type data_input: Any

        :param output: output to write cache for
        :type output: Any

        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def contains_cache_for(self, data_input) -> bool:
        """
        Returns true if the data input transform output is cached.

        :param data_input: to get cache from
        :return: boolean to indicate if a cache is present for the given data input
        """
        raise NotImplementedError()

    @abstractmethod
    def get_cache_path_for(self, data_input) -> str:
        """
        Get the cache path for the given data input.

        :param data_input: data input to get cache path for
        :return: str for cache path
        """
        raise NotImplementedError()


class OutputTransformerMixin:
    """
    Base output transformer step that can modify data inputs, and expected_outputs at the same time.
    """

    def handle_transform(self, data_container: DataContainer) -> DataContainer:
        """
        Handle transform by updating the data inputs, and expected outputs inside the data container.

        :param data_container:
        :return:
        """
        di_eo = (data_container.data_inputs, data_container.expected_outputs)
        new_data_inputs, new_expected_outputs = self.transform(di_eo)

        data_container.set_data_inputs(new_data_inputs)
        data_container.set_expected_outputs(new_expected_outputs)

        current_ids = self.hash(data_container.current_ids, self.hyperparams, data_container.data_inputs)
        data_container.set_current_ids(current_ids)

        return data_container

    def handle_fit_transform(self, data_container: DataContainer) -> ('BaseStep', DataContainer):
        """
        Handle transform by fitting the step,
        and updating the data inputs, and expected outputs inside the data container.

        :param data_container:
        :return:
        """
        new_self = self.fit(data_container.data_inputs, data_container.expected_outputs)

        data_container = self.handle_transform(data_container)

        return new_self, data_container


class PickleValueCachingWrapper(ValueCachingWrapper):
    """
    Value Caching Wrapper class that caches the wrapped step transformed data inputs using python ``pickle`` library.
    """

    def create_checkpoint_path(self, step_path: str) -> str:
        self.checkpoint_path = os.path.join(self.cache_folder, step_path, VALUE_CACHING)

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        return self.checkpoint_path

    def flush_cache(self):
        shutil.rmtree(self.checkpoint_path)
        os.mkdir(self.checkpoint_path)

    def read_cache(self, data_input):
        with open(self.get_cache_path_for(data_input), 'rb') as file_:
            return pickle.load(file_)

    def write_cache(self, data_input, output):
        with open(self.get_cache_path_for(data_input), 'wb') as file_:
            return pickle.dump(output, file_)

    def contains_cache_for(self, data_input) -> bool:
        return os.path.exists(self.get_cache_path_for(data_input))

    def get_cache_path_for(self, data_input):
        hash_value = self._hash_value(data_input)
        return os.path.join(self.checkpoint_path, '{0}.pickle'.format(hash_value))
