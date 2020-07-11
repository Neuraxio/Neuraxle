"""
Neuraxle's DataContainer classes
====================================
Classes for containing the data that flows throught the pipeline steps.

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
import hashlib
import math
from typing import Any, Iterable, List, Tuple, Union

import numpy as np
from conv import convolved_1d

NamedDataContainerTuple = Tuple[str, 'DataContainer']


class DataContainer:
    """
    DataContainer class to store data inputs, expected outputs, and ids together.
    Each :class:`BaseStep` needs to rehash ids with hyperparameters so that the :class:`Checkpoint` step
    can create checkpoints for a set of hyperparameters.

    The DataContainer object is passed to all of the :class:`BaseStep` handle methods :
        * :func:`~neuraxle.base.BaseStep.handle_transform`
        * :func:`~neuraxle.base.BaseStep.handle_fit_transform`
        * :func:`~neuraxle.base.BaseStep.handle_fit`

    Most of the time, you won't need to care about the DataContainer because it is the pipeline that manages it.

    .. seealso::
        :class:`~neuraxle.base.BaseHasher`,
        :class: `neuraxle.base.BaseStep`
    """

    def __init__(
            self,
            data_inputs: Any,
            current_ids=None,
            summary_id=None,
            expected_outputs: Any = None,
            sub_data_containers: List['NamedDataContainerTuple'] = None
    ):
        self.summary_id = summary_id
        self.data_inputs = data_inputs

        if current_ids is None:
            if hasattr(data_inputs, '__len__'):
                current_ids = [str(c) for c in range(len(data_inputs))]
            else:
                current_ids = str(0)

        self.current_ids = current_ids

        if expected_outputs is None and isinstance(data_inputs, Iterable):
            self.expected_outputs = [None] * len(data_inputs)
        else:
            self.expected_outputs = expected_outputs

        if sub_data_containers is None:
            sub_data_containers = []

        self.sub_data_containers: List[NamedDataContainerTuple] = sub_data_containers

    def set_data_inputs(self, data_inputs: Iterable):
        """
        Set data inputs.

        :param data_inputs: data inputs
        :type data_inputs: Iterable
        :return:
        """
        self.data_inputs = data_inputs
        return self

    def set_expected_outputs(self, expected_outputs: Iterable):
        """
        Set expected outputs.

        :param expected_outputs: expected outputs
        :type expected_outputs: Iterable
        :return:
        """
        self.expected_outputs = expected_outputs
        return self

    def set_current_ids(self, current_ids: List[str]):
        """
        Set current ids.

        :param current_ids: data inputs
        :type current_ids: List[str]
        :return:
        """
        self.current_ids = current_ids
        return self

    def set_summary_id(self, summary_id: str):
        """
        Set summary id.

        :param summary_id: str
        :return:
        """
        self.summary_id = summary_id
        return self

    def set_sub_data_containers(self, sub_data_containers: List['DataContainer']):
        """
        Set sub data containers
        :return:
        """
        self.sub_data_containers = sub_data_containers

    def hash_summary(self):
        """
        Hash :class:`DataContainer`.current_ids, data inputs, and hyperparameters together into one id.

        :return: single hashed current id for all of the current ids
        :rtype: str
        """
        m = hashlib.md5()
        for current_id in self.current_ids:
            m.update(str.encode(str(current_id)))
        return m.hexdigest()

    def convolved_1d(self, stride, kernel_size) -> Iterable['DataContainer']:
        """
        Returns an iterator that iterates through batches of the DataContainer.

        :param stride: step size for the convolution operation
        :param kernel_size:
        :return: an iterator of DataContainer
        :rtype: Iterable[DataContainer]

        .. seealso::
            `<https://github.com/guillaume-chevalier/python-conv-lib>`_
        """
        conv_current_ids = convolved_1d(stride=stride, iterable=self.current_ids, kernel_size=kernel_size,
                                        include_incomplete_pass=True)
        conv_data_inputs = convolved_1d(stride=stride, iterable=self.data_inputs, kernel_size=kernel_size,
                                        include_incomplete_pass=True)
        conv_expected_outputs = convolved_1d(stride=stride, iterable=self.expected_outputs, kernel_size=kernel_size,
                                             include_incomplete_pass=True)

        for current_ids, data_inputs, expected_outputs in zip(conv_current_ids, conv_data_inputs,
                                                              conv_expected_outputs):
            for i, (ci, di, eo) in enumerate(zip(current_ids, data_inputs, expected_outputs)):
                if di is None:
                    current_ids = current_ids[:i]
                    data_inputs = data_inputs[:i]
                    expected_outputs = expected_outputs[:i]
                    break

            yield DataContainer(
                data_inputs=data_inputs,
                current_ids=current_ids,
                summary_id=self.summary_id,
                expected_outputs=expected_outputs,
                sub_data_containers=self.sub_data_containers
            )

    def get_n_batches(self, batch_size: int) -> int:
        return math.ceil(len(self.data_inputs) / batch_size)

    def copy(self):
        return DataContainer(
            data_inputs=self.data_inputs,
            current_ids=self.current_ids,
            summary_id=self.summary_id,
            expected_outputs=self.expected_outputs,
            sub_data_containers=[(name, data_container.copy()) for name, data_container in self.sub_data_containers]
        )

    def add_sub_data_container(self, name: str, data_container: 'DataContainer'):
        """
        Get sub data container if item is str, otherwise get a zip of current ids, data inputs, and expected outputs.

        :type name: sub data container name
        :type data_container: sub data container
        :return: self
        """
        self.sub_data_containers.append((name, data_container))
        return self

    def get_sub_data_container_names(self):
        """
        Get sub data container names.

        :return: list of names
        """
        return [name for name, _ in self.sub_data_containers]

    def __contains__(self, item):
        """
        return true if sub container name is in the sub data containers.

        :return: contains
        :rtype: bool
        """
        if isinstance(item, str):
            contains = False
            for name, data_container in self.sub_data_containers:
                if name == item:
                    contains = True
            return contains
        else:
            raise NotImplementedError('DataContainer.__contains__ not implemented for this type: {}'.format(type(item)))

    def __getitem__(self, item: Union[str, int]):
        """
        If item is a string, then get then sub container with the same name as item in the list of sub data containers.
        If item is an int, then return a tuple of (current id, data input, expected output) for the given item index.

        :param item: sub data container str, or item index as int
        :type item: Union[str, int]
        :return: data container, or tuple of current ids, data inputs, expected outputs.
        :rtype: Union[DataContainer, Tuple]
        """
        if isinstance(item, str):
            for name, data_container in self.sub_data_containers:
                if name == item:
                    return data_container
            raise KeyError("sub_data_container {} not found in data container".format(item))
        else:
            if self.current_ids is None:
                current_ids = [None] * len(self)
            else:
                current_ids = self.current_ids[item]

            return current_ids, self.data_inputs[item], self.expected_outputs[item]

    def tolist(self):
        current_ids = self.current_ids
        data_inputs = self.data_inputs
        expected_outputs = self.expected_outputs

        if isinstance(self.current_ids, np.ndarray):
            current_ids = self.current_ids.tolist()

        if isinstance(self.data_inputs, np.ndarray):
            data_inputs = self.data_inputs.tolist()

        if isinstance(self.expected_outputs, np.ndarray):
            expected_outputs = self.expected_outputs.tolist()

        self.set_current_ids(current_ids)
        self.set_data_inputs(data_inputs)
        self.set_expected_outputs(expected_outputs)

        return self

    def tolistshallow(self):
        self.set_current_ids(list(self.current_ids))
        self.set_data_inputs(list(self.data_inputs))
        self.set_expected_outputs(list(self.expected_outputs))

        return self

    def to_numpy(self):
        self.set_current_ids(np.array(self.current_ids))
        self.set_data_inputs(np.array(self.data_inputs))
        self.set_expected_outputs(np.array(self.expected_outputs))
        return self

    def __iter__(self):
        """
        Iter method returns a zip of all of the current_ids, data_inputs, and expected_outputs in the data container.

        :return: iterator of tuples containing current_ids, data_inputs, and expected outputs
        :rtype: Iterator[Tuple]
        """
        current_ids = self.current_ids
        if self.current_ids is None:
            current_ids = [None] * len(self.data_inputs)

        expected_outputs = self.expected_outputs
        if self.expected_outputs is None:
            expected_outputs = [None] * len(self.data_inputs)

        return zip(current_ids, self.data_inputs, expected_outputs)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.__class__.__name__ + "(current_ids=" + repr(list(self.current_ids)) + ", summary_id=" + repr(
            self.summary_id)

    def __len__(self):
        return len(self.data_inputs)


class ExpandedDataContainer(DataContainer):
    """
    Sub class of DataContainer to expand data container dimension.

    .. seealso::
        :class:`DataContainer`,
    """

    def __init__(self, data_inputs, current_ids, expected_outputs, summary_id, old_current_ids):
        DataContainer.__init__(
            self,
            data_inputs=data_inputs,
            current_ids=current_ids,
            summary_id=summary_id,
            expected_outputs=expected_outputs,
        )

        self.old_current_ids = old_current_ids

    def reduce_dim(self) -> 'DataContainer':
        """
        Reduce DataContainer to its original shape with a list of multiple current_ids, data_inputs, and expected outputs.

        :return: reduced data container
        :rtype: DataContainer
        """
        if len(self.data_inputs) != 1:
            raise ValueError(
                'Invalid Expanded Data Container. Please create ExpandedDataContainer with ExpandedDataContainer.create_from(data_container) method.')

        return DataContainer(
            data_inputs=self.data_inputs[0],
            current_ids=self.old_current_ids,
            summary_id=self.summary_id,
            expected_outputs=self.expected_outputs[0],
            sub_data_containers=self.sub_data_containers
        )

    @staticmethod
    def create_from(data_container: DataContainer) -> 'ExpandedDataContainer':
        """
        Create ExpandedDataContainer with the given summary hash for the single current id.

        :param data_container: data container to transform
        :type data_container: DataContainer
        :return: expanded data container
        :rtype: ExpandedDataContainer
        """
        return ExpandedDataContainer(
            data_inputs=[data_container.data_inputs],
            current_ids=[data_container.summary_id],
            summary_id=data_container.summary_id,
            expected_outputs=[data_container.expected_outputs],
            old_current_ids=data_container.current_ids
        )


class ZipDataContainer(DataContainer):
    """
    Sub class of DataContainer to zip two data sources together.

    .. seealso::
        :class:`DataContainer`,
    """

    @staticmethod
    def create_from(data_container: DataContainer, *other_data_containers: DataContainer) -> 'ZipDataContainer':
        """
        Create ZipDataContainer that merges two data sources together.

        :param data_container: data container to transform
        :type data_container: DataContainer
        :param other_data_containers: other data containers to zip with data container
        :type other_data_containers: List[DataContainer]
        :return: expanded data container
        :rtype: ExpandedDataContainer
        """
        new_data_inputs = []

        for i, (_, di, eo) in enumerate(data_container):
            new_data_input = [di]

            for other_data_container in other_data_containers:
                _, di, eo = other_data_container[i]
                new_data_input.append(di)

            new_data_inputs.append(tuple(new_data_input))

        return ZipDataContainer(
            data_inputs=new_data_inputs,
            current_ids=data_container.current_ids,
            summary_id=data_container.summary_id,
            expected_outputs=data_container.expected_outputs,
            sub_data_containers=data_container.sub_data_containers
        )

    def concatenate_inner_features(self):
        """
        Concatenate inner features from zipped data inputs.
        Broadcast data inputs if the dimension is smaller.
        """
        new_data_inputs = [di[0] for di in self.data_inputs]

        for i, data_input in enumerate(self.data_inputs):
            for di in data_input[1:]:
                new_data_inputs[i] = _inner_concatenate_np_array(new_data_inputs[i], di)

        self.set_data_inputs(new_data_inputs)


class ListDataContainer(DataContainer):
    """
    Sub class of DataContainer to perform list operations.
    It allows to perform append, and concat operations on a DataContainer.

    .. seealso::
        :class:`DataContainer`
    """

    def __init__(self, data_inputs: Any, current_ids=None, summary_id=None, expected_outputs: Any = None, sub_data_containers=None):
        DataContainer.__init__(self, data_inputs, current_ids, summary_id, expected_outputs, sub_data_containers)
        self.tolistshallow()

    @staticmethod
    def empty(original_data_container: DataContainer = None) -> 'ListDataContainer':
        if original_data_container is None:
            sub_data_containers = []
        else:
            sub_data_containers = original_data_container.sub_data_containers

        return ListDataContainer([], [], [], sub_data_containers=sub_data_containers)

    def append(self, current_id: str, data_input: Any, expected_output: Any):
        """
        Append a new data input to the DataContainer.

        :param current_id: current id for the data input
        :type current_id: str
        :param data_input: data input
        :param expected_output: expected output
        :return:
        """
        self.current_ids.append(current_id)
        self.data_inputs.append(data_input)
        self.expected_outputs.append(expected_output)

    def append_data_container_in_data_inputs(self, other: DataContainer) -> 'ListDataContainer':
        """
        Append a data container to the data inputs of this data container.

        :param other: data container
        :type other: DataContainer
        :return:
        """
        self.data_inputs.append(other)
        return self

    def append_data_container(self, other: DataContainer) -> 'ListDataContainer':
        """
        Append a data container to the DataContainer.

        :param other: data container
        :type other: DataContainer
        :return:
        """
        self.current_ids.append(other.current_ids)
        self.data_inputs.append(other.data_inputs)
        self.expected_outputs.append(other.expected_outputs)

        return self

    def concat(self, data_container: DataContainer):
        """
        Concat the given data container to the current data container.

        :param data_container: data container
        :type data_container: DataContainer
        :return:
        """
        data_container.tolistshallow()

        self.current_ids.extend(data_container.current_ids)
        self.data_inputs.extend(data_container.data_inputs)
        self.expected_outputs.extend(data_container.expected_outputs)

        return self


def _inner_concatenate_np_array(np_array, np_array_to_zip):
    """
    Concatenate numpy arrays on the last axis using expand dim, and broadcasting.

    :param np_array: data container
    :type np_array: np.ndarray
    :param np_array_to_zip: numpy array to zip with the other
    :type np_array_to_zip: np.ndarray
    :return: concatenated np array
    :rtype: np.ndarray
    """
    while len(np_array_to_zip.shape) < len(np_array.shape):
        np_array_to_zip = np.expand_dims(np_array_to_zip, axis=-1)

    target_shape = tuple(list(np_array.shape[:-1]) + [np_array_to_zip.shape[-1]])
    np_array_to_zip = np.broadcast_to(np_array_to_zip, target_shape)

    return np.concatenate((np_array, np_array_to_zip), axis=-1)
