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

..
    Thanks to Umaneo Technologies Inc. for their contributions to this Machine Learning
    project, visit https://www.umaneo.com/ for more information on Umaneo Technologies Inc.

"""

import numpy as np

from neuraxle.base import NonFittableMixin, BaseStep, DataContainer, ExecutionContext, ForceHandleOnlyMixin, \
    ForceHandleMixin
from neuraxle.hyperparams.space import HyperparameterSamples


class NumpyFlattenDatum(NonFittableMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)

    def transform(self, data_inputs):
        return data_inputs.reshape(data_inputs.shape[0], -1)


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

    def _transform_data_container(self, data_container, context):
        """
        Handle transform.

        :param data_container: the data container to join
        :param context: execution context
        :return: transformed data container
        """
        data_inputs = self.transform([dc.data_inputs for dc in data_container.data_inputs])
        data_container = DataContainer(data_inputs=data_inputs, current_ids=data_container.current_ids,
                                       expected_outputs=data_container.expected_outputs)
        data_container.set_data_inputs(data_inputs)

        return data_container

    def transform(self, data_inputs):
        """
        Apply the concatenation transformation along the specified axis.
        :param data_inputs:
        :return: Numpy array
        """
        return self._concat(data_inputs)

    def _concat(self, data_inputs):
        return np.concatenate(data_inputs, axis=self.axis)


class NumpyConcatenateInnerFeatures(NumpyConcatenateOnCustomAxis):
    """
    Numpy concatenation step where the concatenation is performed along `axis=-1`.
    """

    def __init__(self):
        """
        Create a numpy concatenate inner features object.
        :return: NumpyConcatenateOnCustomAxis instance.
        """
        # The concatenate is on the inner features so axis = -1.
        NumpyConcatenateOnCustomAxis.__init__(self, axis=-1)


class NumpyConcatenateOuterBatch(NumpyConcatenateOnCustomAxis):
    """
    Numpy concetenation step where the concatenation is performed along `axis=0`.
    """

    def __init__(self):
        """
        Create a numpy concatenate outer batch object.
        :return: NumpyConcatenateOnCustomAxis instance which is inherited by base step.
        """
        NumpyConcatenateOnCustomAxis.__init__(self, axis=0)


class NumpyTranspose(NonFittableMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)

    def _transform_data_container(self, data_container, context):
        """
        Handle transform.

        :param data_container: the data container to join
        :param context: execution context
        :return: transformed data container
        """
        data_inputs = self.transform([dc.data_inputs for dc in data_container.data_inputs])
        data_container = DataContainer(data_inputs=data_inputs, current_ids=data_container.current_ids,
                                       expected_outputs=data_container.expected_outputs)
        data_container.set_data_inputs(data_inputs)

        return data_container

    def transform(self, data_inputs):
        return self._transpose(data_inputs)

    def inverse_transform(self, data_inputs):
        return self._transpose(data_inputs)

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

    def inverse_transform(self, processed_outputs):
        self._print(processed_outputs)
        return processed_outputs

    def _print(self, data_inputs):
        print(self.__class__.__name__ + ":", data_inputs.shape, self.custom_message)

    def _print_one(self, data_input):
        print(self.__class__.__name__ + " (one):", data_input.shape, self.custom_message)


class MultiplyByN(NonFittableMixin, BaseStep):
    """
    Step to multiply a numpy array.
    Accepts an integer for the number to multiply by.

    Example usage:

    .. code-block:: python

        pipeline = Pipeline([
            MultiplyByN(3)
        ])
        outputs = pipeline.transform(np.array([1])
        # outputs => np.array([3])

    .. seealso::
        :class:`~neuraxle.base.NonFittableMixin`,
        :class:`~neuraxle.base.BaseStep`
    """

    def __init__(self, multiply_by=1):
        NonFittableMixin.__init__(self)
        BaseStep.__init__(
            self,
            hyperparams=HyperparameterSamples({
                'multiply_by': multiply_by
            })
        )

    def transform(self, data_inputs):
        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        return data_inputs * self.hyperparams['multiply_by']

    def inverse_transform(self, data_inputs):
        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        return data_inputs / self.hyperparams['multiply_by']


class AddN(NonFittableMixin, BaseStep):
    """
    Step to add a scalar to a numpy array.
    Accepts an integer for the number to add to every data inputs.

    Example usage:

    .. code-block:: python

        pipeline = Pipeline([
            AddN(1)
        ])
        outputs = pipeline.transform(np.array([1])
        # outputs => np.array([2])

    .. seealso::
        :class:`~neuraxle.base.NonFittableMixin`,
        :class:`~neuraxle.base.BaseStep`
    """

    def __init__(self, add=1):
        NonFittableMixin.__init__(self)
        BaseStep.__init__(
            self,
            hyperparams=HyperparameterSamples({
                'add': add
            })
        )

    def transform(self, data_inputs):
        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        return data_inputs + self.hyperparams['add']

    def inverse_transform(self, data_inputs):
        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        return data_inputs - self.hyperparams['add']


class Sum(NonFittableMixin, BaseStep):
    """
    Step sum numpy array using np.sum.

    Example usage:

    .. code-block:: python

        pipeline = Pipeline([
            Sum(axis=-1)
        ])

        outputs = pipeline.transform(np.array([1, 2, 3])
        # outputs => 6)

    .. seealso::
        :class:`NonFittableMixin`,
        :class:`BaseStep`
    """

    def __init__(self, axis):
        NonFittableMixin.__init__(self)
        BaseStep.__init__(self)
        self.axis = axis

    def transform(self, data_inputs):
        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        data_inputs = np.expand_dims(np.sum(data_inputs, axis=self.axis), axis=-1)
        return data_inputs


class OneHotEncoder(NonFittableMixin, BaseStep):
    """
    Step to one hot a set of columns.
    Accepts Integer Columns and converts it ot a one_hot.
    Rounds floats  to integer for safety in the transform.
    
    Example usage: 
    
    1. Set up data

    .. code-block:: python

       import numpy as np
       a = np.array([1,0,3])
       b = np.array([[0,1,0,0], [1,0,0,0], [0,0,0,1]])

    2. **Do the actual conversion**

    .. code-block:: python

       from neuraxle.steps.numpy import OneHotEncoder
       encoder = OneHotEncoder(nb_columns=4)
       b_pred = encoder.transform(a)

    3. Assert it works

    .. code-block:: python

       assert b_pred == b
    
    .. seealso::
        `StackOverflow answer about one hot encoding <https://stackoverflow.com/a/59262363/2476920>`__
    
    """

    def __init__(self, nb_columns, name):
        super().__init__(name=name)
        self.nb_columns = nb_columns

    def transform(self, data_inputs):
        """
        Transform data inputs using one hot encoding, adding no_columns to the -1 axis.
        :param data_inputs: data inputs to encode
        :return: one hot encoded data inputs
        """
        # validate enum values
        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        # treats invalid values as having no columns activated. create a temporary column for invalid values
        data_inputs[data_inputs == None] = self.nb_columns
        data_inputs[data_inputs >= self.nb_columns] = self.nb_columns
        data_inputs[data_inputs < 0] = self.nb_columns

        # finally, one hot encode data inputs
        outputs_ = np.eye(self.nb_columns + 1)[np.array(data_inputs, dtype=np.int32)]

        # delete the invalid values column, and zero hot the invalid values
        outputs_ = np.delete(outputs_, self.nb_columns, axis=-1)

        return outputs_.squeeze()


class ToNumpy(ForceHandleMixin, BaseStep):
    """
    Convert data inputs, and expected outputs to a numpy array.
    """

    def _will_process(self, data_container: DataContainer, context: ExecutionContext) -> (DataContainer, ExecutionContext):
        return data_container.to_numpy(), context


class NumpyReshape(NonFittableMixin, BaseStep):
    """
    Reshape numpy array in data inputs.

    .. code-block:: python

       import numpy as np
       a = np.array([1,0,3])
       outputs = NumpyReshape(shape=(-1,1)).transform(a)
       assert np.array_equal(outputs, np.array([[1],[0],[3]]))

    .. seealso::
        :class:`NonFittableMixin`
        :class:`BaseStep`
    """

    def __init__(self, new_shape):
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)
        self.new_shape = new_shape

    def transform(self, data_inputs):
        return np.reshape(data_inputs, newshape=self.new_shape)
