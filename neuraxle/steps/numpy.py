"""
Pipeline Steps Based on NumPy
=====================================
Those steps works with NumPy (np) arrays.

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

import numpy as np

from neuraxle.base import NonFittableMixin, BaseStep


class NumpyFlattenDatum(NonFittableMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)

    def transform(self, data_inputs):
        return data_inputs.reshape(data_inputs.shape[0], -1)

    def transform_one(self, data_inputs):
        return data_inputs.flatten()


class NumpyConcatenateOnCustomAxis(NonFittableMixin, BaseStep):
    """
    Numpy concetenation step where the concatenation is performed along the specified custom axis.
    """

    def __init__(self, axis):
        """
        Create a numpy concatenate on custom axis object.
        :param axis: the axis where the concatenation is performed.
        :return: NumpyConcatenateOnCustomAxis instance.
        """
        self.axis = axis
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)

    def transform(self, data_inputs):
        """
        Apply the concatenation transformation along the specified axis.
        :param data_inputs:
        :return: Numpy array
        """
        return self._concat(data_inputs)

    def transform_one(self, data_inputs):
        """
        Apply the concatenation transformation along the specified axis
        :param data_inputs:
        :return: Numpy array
        """
        return self._concat(data_inputs)

    def _concat(self, data_inputs):
        return np.concatenate(data_inputs, axis=self.axis)

class NumpyConcatenateInnerFeatures(NumpyConcatenateOnCustomAxis):
    """
    Numpy concetenation step where the concatenation is performed along `axis=-1`.
    """

    def __init__(self):
        """
        Create a numpy concatenate inner features object.
        :return: NumpyConcatenateOnCustomAxis instance.
        """
        # The concatenate is on the inner features so axis = -1.
        super().__init__(axis=-1)

class NumpyConcatenateOuterBatch(NumpyConcatenateOnCustomAxis):
    """
    Numpy concetenation step where the concatenation is performed along `axis=0`.
    """

    def __init__(self):
        """
        Create a numpy concetenate outer batch object.
        :return: NumpyConcatenateOnCustomAxis instance which is inherited by base step.
        """
        super().__init__(axis=0)


class NumpyTranspose(NonFittableMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)

    def transform(self, data_inputs):
        return self._transpose(data_inputs)

    def transform_one(self, data_input):
        raise BrokenPipeError("Cannot simply `transform_one` here: transpose must be done at a higher level.")

    def inverse_transform(self, data_inputs):
        return self._transpose(data_inputs)

    def inverse_transform_one(self, data_input):
        raise BrokenPipeError("Cannot simply `inverse_transform_one` here: transpose must be done at a higher level.")

    def _transpose(self, data_inputs):
        return np.array(data_inputs).transpose()


class NumpyShapePrinter(NonFittableMixin, BaseStep):

    def __init__(self, custom_message: str = ""):
        self.custom_message = custom_message
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)

    def transform(self, data_inputs):
        self._print(data_inputs)
        return data_inputs

    def transform_one(self, data_input):
        self._print_one(data_input)

    def inverse_transform(self, processed_outputs):
        self._print(processed_outputs)
        return processed_outputs

    def inverse_transform_one(self, processed_output):
        self._print_one(processed_output)

    def _print(self, data_inputs):
        print(self.__class__.__name__ + ":", data_inputs.shape, self.custom_message)

    def _print_one(self, data_input):
        print(self.__class__.__name__ + " (one):", data_input.shape, self.custom_message)
