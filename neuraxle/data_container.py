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
import copy
import math
from operator import attrgetter
from typing import Any, Callable, Iterable, List, Tuple, Union

import numpy as np

NamedDataContainerTuple = Tuple[str, 'DataContainer']


class AbsentValuesNullObject:
    """
    This object, when passed to the default_value_data_inputs argument of the DataContainer.batch method, will return the minibatched data containers such that the last batch won't have the full batch_size if it was incomplete with trailing None values at the end.
    """
    pass


class DataContainer:
    """
    DataContainer class to store data inputs, expected outputs, and ids together.
    can create checkpoints for a set of hyperparameters.

    The DataContainer object is passed to all of the :class:`BaseStep` handle methods :
        * :func:`~neuraxle.base.BaseStep.handle_transform`
        * :func:`~neuraxle.base.BaseStep.handle_fit_transform`
        * :func:`~neuraxle.base.BaseStep.handle_fit`

    Most of the time, the steps will manage it by themselves.

    .. seealso::
        :class:`~neuraxle.base.BaseHasher`,
        :class:`~neuraxle.base.BaseStep`,
        :class:`~neuraxle.data_container.DataContainer.AbsentValuesNullObject`
    """

    def __init__(
            self,
            data_inputs: Iterable,
            ids=None,
            expected_outputs: Any = None,
            sub_data_containers: List['NamedDataContainerTuple'] = None
    ):
        self.data_inputs: Iterable = data_inputs

        if ids is None:
            if hasattr(data_inputs, '__len__') or hasattr(data_inputs, '__iter__'):
                ids = [str(c) for c in range(len(data_inputs))]
            else:
                ids = str(0)

        self.ids: Iterable = ids

        if expected_outputs is None and (hasattr(data_inputs, '__len__') or hasattr(data_inputs, '__iter__')):
            self.expected_outputs: Iterable = [None] * len(data_inputs)
        else:
            self.expected_outputs: Iterable = expected_outputs

        if sub_data_containers is None:
            sub_data_containers = []

        self.sub_data_containers: List[NamedDataContainerTuple] = sub_data_containers

    def set_data_inputs(self, data_inputs: Iterable):
        """
        Set data inputs.

        :param data_inputs: data inputs
        :type data_inputs: Iterable
        :return: self
        """
        self.data_inputs = data_inputs
        return self

    def set_expected_outputs(self, expected_outputs: Iterable):
        """
        Set expected outputs.

        :param expected_outputs: expected outputs
        :type expected_outputs: Iterable
        :return: self
        """
        self.expected_outputs = expected_outputs
        return self

    def set_ids(self, ids: List[str]):
        """
        Set ids.

        :param ids: data inputs
        :return: self
        """
        self.ids = ids
        return self

    def get_ids_summary(self):
        return ','.join([i for i in self.ids if i is not None])

    def set_sub_data_containers(self, sub_data_containers: List['DataContainer']) -> 'DataContainer':
        """
        Set sub data containers
        :return: self
        """
        self.sub_data_containers = sub_data_containers
        return self

    def minibatches(
            self,
            batch_size: int,
            keep_incomplete_batch: bool = True,
            default_value_data_inputs=None,
            default_value_expected_outputs=None
    ) -> Iterable['DataContainer']:
        """
        Yields minibatches extracted from looping on the DataContainer's content with a batch_size and a certain behavior for the last batch when the batch_size is uneven with the total size.


        .. code-block:: python

            data_container = DataContainer(data_inputs=np.array(list(range(10)))
            for data_container_batch in data_container.minibatches(batch_size=2):
                print(data_container_batch.data_inputs)
                print(data_container_batch.expected_outputs)
            # [array([0, 1]), array([2, 3]), ..., array([8, 9])]

            data_container = DataContainer(data_inputs=np.array(list(range(10)))
            for data_container_batch in data_container.minibatches(batch_size=3, keep_incomplete_batch=False):
                print(data_container_batch.data_inputs)
            # [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8])]

            data_container = DataContainer(data_inputs=np.array(list(range(10)))
            for data_container_batch in data_container.minibatches(
                batch_size=3,
                keep_incomplete_batch=True,
                default_value_data_inputs=None,
                default_value_expected_outputs=None
            ):
                print(data_container_batch.data_inputs)
            # [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8]), array([9, None, None])]

            data_container = DataContainer(data_inputs=np.array(list(range(10)))
            for data_container_batch in data_container.minibatches(
                batch_size=3,
                keep_incomplete_batch=True,
                default_value_data_inputs=AbsentValuesNullObject()
            ):
                print(data_container_batch.data_inputs)
            # [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8]), array([9])]


        :param batch_size: number of elements to combine into a single batch
        :param keep_incomplete_batch: (Optional.) A bool representing
        whether the last batch should be dropped in the case it has fewer than
        `batch_size` elements; the default behavior is to keep the smaller
        batch.
        :param default_value_data_inputs: expected_outputs default fill value
        for padding and values outside iteration range, or :class:`~neuraxle.data_container.DataContainer.AbsentValuesNullObject`
        to trim absent values from the batch
        :param default_value_expected_outputs: expected_outputs default fill value
        for padding and values outside iteration range, or :class:`~neuraxle.data_container.DataContainer.AbsentValuesNullObject`
        to trim absent values from the batch
        :return: an iterator of DataContainer
        :rtype: Iterable[DataContainer]

        .. seealso::
            :class:`~neuraxle.data_container.DataContainer.AbsentValuesNullObject`
        """
        for i in range(0, len(self.data_inputs), batch_size):
            data_container = DataContainer(
                ids=self.ids[i:i + batch_size],
                data_inputs=self.data_inputs[i:i + batch_size],
                expected_outputs=self.expected_outputs[i:i + batch_size]
            )

            incomplete_batch = len(data_container.data_inputs) < batch_size
            if incomplete_batch:
                if not keep_incomplete_batch:
                    break

                data_container = _pad_or_keep_incomplete_batch(
                    data_container,
                    batch_size,
                    default_value_data_inputs,
                    default_value_expected_outputs
                )

            yield data_container

    def get_n_batches(self, batch_size: int, keep_incomplete_batch: bool = True) -> int:
        if keep_incomplete_batch:
            return math.ceil(len(self.data_inputs) / batch_size)
        else:
            return math.floor(len(self.data_inputs) / batch_size)

    def copy(self):
        return DataContainer(
            data_inputs=self.data_inputs,
            ids=self.ids,
            expected_outputs=self.expected_outputs,
            sub_data_containers=[(name, data_container.copy()) for name, data_container in self.sub_data_containers]
        )

    def add_sub_data_container(self, name: str, data_container: 'DataContainer') -> 'DataContainer':
        """
        Get sub data container if item is str, otherwise get a zip of ids, data inputs, and expected outputs.

        :type name: sub data container name
        :type data_container: sub data container
        :return: self
        """
        self.sub_data_containers.append((name, data_container))
        return self

    def get_sub_data_container_names(self) -> List[str]:
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
        If item is an int, then return a tuple of (id, data input, expected output) for the given item index.

        :param item: sub data container str, or item index as int
        :type item: Union[str, int]
        :return: data container, or tuple of ids, data inputs, expected outputs.
        :rtype: Union[DataContainer, Tuple]
        """
        if isinstance(item, str):
            for name, data_container in self.sub_data_containers:
                if name == item:
                    return data_container
            raise KeyError("sub_data_container {} not found in data container".format(item))
        else:
            if self.ids is None:
                ids = [None] * len(self)
            else:
                ids = self.ids[item]

            return ids, self.data_inputs[item], self.expected_outputs[item]

    def tolist(self) -> 'DataContainer':
        def pandas_tonp(df):
            if 'DataFrame' in str(type(df)) or 'Series' in str(type(df)):
                return df.values.tolist()
            else:
                return df

        def ndarr_tolist(ndarr):
            if isinstance(ndarr, np.ndarray):
                return ndarr.tolist()
            else:
                return ndarr

        self.apply_conversion_func(pandas_tonp)
        self.apply_conversion_func(ndarr_tolist)
        return self

    def tolistshallow(self) -> 'DataContainer':
        return self.apply_conversion_func(list)

    def to_numpy(self) -> 'DataContainer':
        return self.apply_conversion_func(np.array)

    def apply_conversion_func(self, conversion_function: Callable[[Any], Any]) -> 'DataContainer':
        """
        Apply conversion function to data inputs, expected outputs, and ids,
        and set the new values in self. Returns self.
        Conversion function must be able to handle None values.
        """
        self.set_ids(conversion_function(self.ids))
        self.set_data_inputs(conversion_function(self.data_inputs))
        self.set_expected_outputs(conversion_function(self.expected_outputs))
        return self

    def unpack(self) -> Tuple[Iterable[Any], Iterable[Any], Iterable[Any]]:
        """
        Unpack to a tuples of (ids, data input, expected output).

        :return: tuple of ids, data inputs, expected outputs
        """
        return self.ids, self.data_inputs, self.expected_outputs

    def __iter__(self):
        """
        Iter method returns a zip of all of the ids, data_inputs, and expected_outputs in the data container.

        :return: iterator of tuples containing ids, data_inputs, and expected outputs
        :rtype: Iterator[Tuple]
        """
        ids = self.ids
        if self.ids is None:
            ids = [None] * len(self.data_inputs)

        expected_outputs = self.expected_outputs
        if self.expected_outputs is None:
            expected_outputs = [None] * len(self.data_inputs)

        return zip(ids, self.data_inputs, expected_outputs)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.__class__.__name__ + "(ids=" + repr(list(self.ids)) + ")"

    def __len__(self):
        return len(self.data_inputs)


class ExpandedDataContainer(DataContainer):
    """
    Sub class of DataContainer to expand data container dimension.

    .. seealso::
        :class:`DataContainer`
    """

    def __init__(self, data_inputs, ids, expected_outputs, old_ids):
        DataContainer.__init__(
            self,
            data_inputs=data_inputs,
            ids=ids,
            expected_outputs=expected_outputs,
        )

        self.old_ids = old_ids

    def reduce_dim(self) -> 'DataContainer':
        """
        Reduce DataContainer to its original shape with a list of multiple ids, data_inputs, and expected outputs.

        :return: reduced data container
        :rtype: DataContainer
        """
        if len(self.data_inputs) != 1:
            raise ValueError(
                'Invalid Expanded Data Container. Please create ExpandedDataContainer with ExpandedDataContainer.create_from(data_container) method.')

        return DataContainer(
            data_inputs=self.data_inputs[0],
            ids=self.old_ids,
            expected_outputs=self.expected_outputs[0],
            sub_data_containers=self.sub_data_containers
        )

    @staticmethod
    def create_from(data_container: DataContainer) -> 'ExpandedDataContainer':
        """
        Create ExpandedDataContainer with a summary id for the new single id.

        :param data_container: data container to transform
        :type data_container: DataContainer
        :return: expanded data container
        :rtype: ExpandedDataContainer
        """
        return ExpandedDataContainer(
            data_inputs=[data_container.data_inputs],
            ids=[data_container.get_ids_summary()],
            expected_outputs=[data_container.expected_outputs],
            old_ids=data_container.ids
        )


class ZipDataContainer(DataContainer):
    """
    Sub class of DataContainer to zip two data sources together.

    .. seealso::
        :class: `DataContainer`
    """

    @staticmethod
    def create_from(data_container: DataContainer, *other_data_containers: List[DataContainer], zip_expected_outputs: bool = False) -> 'ZipDataContainer':
        """
        Merges two data sources together. Zips only the data input part and keeps the expected output of the first DataContainer as is.
        NOTE: Expects that all DataContainer are at least as long as data_container.

        :param data_container: the main data container, the attribute of this data container will be kept by the returned ZipDataContainer.
        :type data_container: DataContainer
        :param other_data_containers: other data containers to zip with data container
        :type other_data_containers: List[DataContainer]
        :param zip_expected_outputs: Determines wether we kept the expected_output of data_container or we zip the expected_outputs of all DataContainer provided
        :return: expanded data container
        :rtype: ExpandedDataContainer
        """

        new_data_inputs = tuple(zip(*map(attrgetter("data_inputs"), [data_container] + list(other_data_containers))))
        if zip_expected_outputs:
            expected_outputs = tuple(
                zip(*map(attrgetter("expected_outputs"), [data_container] + list(other_data_containers))))
        else:
            expected_outputs = data_container.expected_outputs

        return ZipDataContainer(
            data_inputs=new_data_inputs,
            expected_outputs=expected_outputs,
            ids=data_container.ids,
            sub_data_containers=data_container.sub_data_containers
        )

    def concatenate_inner_features(self):
        """
        Concatenate inner features from zipped data inputs.
        Assumes each data_input entry is an iterable of numpy arrays.
        """
        new_data_inputs = [di[0] for di in self.data_inputs]

        for i, data_input in enumerate(self.data_inputs):
            new_data_inputs[i] = _inner_concatenate_np_array(list(data_input))

        self.set_data_inputs(new_data_inputs)


class ListDataContainer(DataContainer):
    """
    Sub class of DataContainer to perform list operations.
    It allows to perform append, and concat operations on a DataContainer.

    .. seealso::
        :class:`DataContainer`
    """

    def __init__(self, data_inputs: Any, ids=None, expected_outputs: Any = None,
                 sub_data_containers=None):
        DataContainer.__init__(self, data_inputs, ids, expected_outputs, sub_data_containers)
        self.tolistshallow()

    @staticmethod
    def empty(original_data_container: DataContainer = None) -> 'ListDataContainer':
        if original_data_container is None:
            sub_data_containers = []
        else:
            sub_data_containers = original_data_container.sub_data_containers

        return ListDataContainer([], [], [], sub_data_containers=sub_data_containers)

    def append(self, _id: str, data_input: Any, expected_output: Any):
        """
        Append a new data input to the DataContainer.

        :param _id: id for the data input
        :type _id: str
        :param data_input: data input
        :param expected_output: expected output
        :return:
        """
        self.ids.append(_id)
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
        self.ids.append(other.get_ids_summary())
        return self

    def append_data_container(self, other: DataContainer) -> 'ListDataContainer':
        """
        Append a data container to the DataContainer.

        :param other: data container
        :type other: DataContainer
        :return:
        """
        self.ids.append(other.ids)
        self.data_inputs.append(other.data_inputs)
        self.expected_outputs.append(other.expected_outputs)

        return self

    def concat(self, data_container: DataContainer):
        """
        Concat the given data container to the data container.

        :param data_container: data container
        :type data_container: DataContainer
        :return:
        """
        data_container.tolistshallow()

        self.ids.extend(data_container.ids)
        self.data_inputs.extend(data_container.data_inputs)
        self.expected_outputs.extend(data_container.expected_outputs)

        return self


def _pad_or_keep_incomplete_batch(
        data_container,
        batch_size,
        default_value_data_inputs,
        default_value_expected_outputs
) -> 'DataContainer':
    should_pad_right = not isinstance(default_value_data_inputs, AbsentValuesNullObject)

    if should_pad_right:
        data_container = _pad_incomplete_batch(
            data_container,
            batch_size,
            default_value_data_inputs,
            default_value_expected_outputs
        )

    return data_container


def _pad_incomplete_batch(
        data_container: 'DataContainer',
        batch_size: int,
        default_value_data_inputs: Any,
        default_value_expected_outputs: Any
) -> 'DataContainer':
    data_container = DataContainer(
        ids=_pad_data(
            data_container.ids,
            default_value=None,
            batch_size=batch_size
        ),
        data_inputs=_pad_data(
            data_container.data_inputs,
            default_value=default_value_data_inputs,
            batch_size=batch_size
        ),
        expected_outputs=_pad_data(
            data_container.expected_outputs,
            default_value=default_value_expected_outputs,
            batch_size=batch_size
        )
    )

    return data_container


def _pad_data(data: Iterable, default_value: Any, batch_size: int):
    data_ = []
    data_.extend(data)
    padding = copy.copy([default_value] * (batch_size - len(data)))
    data_.extend(padding)
    return data_


def _inner_concatenate_np_array(np_arrays_to_concatenate: List[np.ndarray]):
    """
    Concatenate numpy arrays on the last axis, expanding and broadcasting if necessary.

    :param np_arrays_to_concatenate: numpy arrays to zip with the other
    :type np_arrays_to_concatenate: Iterable[np.ndarray]
    :return: concatenated np array
    :rtype: np.ndarray
    """
    n_arrays = len(np_arrays_to_concatenate)
    target_n_dims = max(map(lambda x: len(x.shape), np_arrays_to_concatenate))
    # These three next lines may be a bit hard to follow;
    # they compute the max for every dimensions of the np arrays except the last.
    # Previous code just assumed np_arrays_to_concatenate[0].shape[:-1]
    _temp = map(attrgetter('shape'), np_arrays_to_concatenate)
    _temp = map(lambda x: list(x)+[0 for _ in range(target_n_dims-len(x))], _temp)
    target_dims_m1 = list(map(max, *list(_temp)))[:-1]

    for i in range(n_arrays):
        while len(np_arrays_to_concatenate[i].shape) < target_n_dims:
            np_arrays_to_concatenate[i] = np.expand_dims(np_arrays_to_concatenate[i], axis=-1)

        target_shape = tuple(target_dims_m1 + [np_arrays_to_concatenate[i].shape[-1]])
        np_arrays_to_concatenate[i] = np.broadcast_to(np_arrays_to_concatenate[i], target_shape)

    return np.concatenate(np_arrays_to_concatenate, axis=-1)
