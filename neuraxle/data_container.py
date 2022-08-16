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
from typing import Any, Callable, Generic, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

NamedDACTTuple = Tuple[str, 'DataContainer']
IDT = TypeVar('IDT', bound=Iterable)  # Ids Type that is often a list of things
DIT = TypeVar('DIT', bound=Iterable)  # Data Inputs Type that is often a list of things
EOT = TypeVar('EOT', bound=Iterable)  # Expected Outputs Type that is often a list of things
DACTData = Union[IDT, DIT, EOT]  # Any of the 3 types

ARG_X_INPUTTED = DIT
ARG_Y_EXPECTED = EOT
ARG_Y_PREDICTD = DIT


class StripAbsentValues:
    """
    This object, when passed to the default_value_data_inputs argument of the DataContainer.batch method,
    will return the minibatched data containers such that the last batch won't have the full batch_size
    if it was incomplete with trailing None values at the end.
    """
    pass


class DataContainer(Generic[IDT, DIT, EOT]):
    """
    DataContainer (dact) class to store IDs (ids), data inputs (di), and expected outputs (eo) together.
    In some dacts, you could have only ids and data inputs, and in other dacts you could have only expected outputs,
    or you could have all if you want, such as when your :class:`~neuraxle.pipeline.Pipeline` is used to train a model
    in a certain :class:`~neuraxle.base.ExecutionMode` within a certain :class:`~neuraxle.base.ExecutionContext`.

    You can use typing for your dact, and create a dact, such as:

    .. code-block:: python

        from typing import List
        from neuraxle.data_container import DataContainer as DACT

        dact: DACT[List[str], List[int], List[float]] = DACT(
            ids=['a', 'b', 'c'],
            data_inputs=[1, 2, 3],
            expected_outputs=[1.0, 2.0, 3.0]
        )


    This is because the DataContainer inherits from the :class:`~typing.Generic` type
    as ``class DataContainer(Generic[IDT, DIT, EOT]): ...``.

    The DataContainer object is passed to all of the :class:`~neuraxle.base.BaseStep` 's handle methods :
        * :func:`~neuraxle.base.BaseStep.handle_transform`
        * :func:`~neuraxle.base.BaseStep.handle_fit_transform`
        * :func:`~neuraxle.base.BaseStep.handle_fit`

    Most of the time, the steps will manage it in the handler methods.

    .. seealso::
        :class:`~neuraxle.base.BaseStep`,
        :class:`~neuraxle.data_container.StripAbsentValues`
    """

    def __init__(
            self,
            data_inputs: Optional[DIT] = None,
            ids: Optional[IDT] = None,
            expected_outputs: Optional[EOT] = None,
            sub_data_containers: List['NamedDACTTuple'] = None,
            *,
            di: Optional[DIT] = None,
            eo: Optional[EOT] = None,

    ):
        """
        Create a DataContainer[IDT, DIT, EOT] object from specified ids, di, and eo.

        :param ids: ids that are iterable. If None, will put a range of integers of data_inputs length. Often a list of integers.
        :param di: same as ``data_inputs``, but shorter.
        :param eo: same as ``expected_outputs``, but shorter.
        :param data_inputs: data inputs that are iterable. Can use di instead.
        :param expected_outputs: expected outputs that are iterable. If None, will put a list of None of data_inputs length.
        :param sub_data_containers: sub data containers.
        """
        self._ids: IDT = ids
        self.data_inputs: DIT = data_inputs if di is None else di
        self.expected_outputs: EOT = expected_outputs if eo is None else eo
        self.sub_data_containers: List[NamedDACTTuple] = sub_data_containers or []

    @property
    def ids(self) -> IDT:
        """
        Get ids.

        If the ids are None, the following IDs will be returned:

        - If the data_inputs is a :class:`~pandas.DataFrame`, will return the index of the DF.
        - Else if the ids are None, will return a range of integers of data_inputs length.
        - Else if the data_inputs aren't iterable, will return a range of integers of expected_outputs length.

        :return: ids
        :rtype: Iterable
        """
        if self._ids is None:
            if hasattr(self.di, 'index') and hasattr(self.di.index, 'values'):
                # This is a pd.DataFrame:
                return self.di.index.values
            if hasattr(self.di, '__len__') or hasattr(self.di, '__iter__'):
                return [i for i in range(len(self.di))]
            if hasattr(self.eo, '__len__') or hasattr(self.eo, '__iter__'):
                return [i for i in range(len(self.eo))]
        return self._ids

    @ids.setter
    def ids(self, ids: IDT):
        """
        Set ids.

        :param ids: ids, often a list of integers.
        :type ids: Iterable
        """
        self._ids = ids

    @property
    def di(self) -> DIT:
        """
        Get data inputs.

        :return: data inputs
        :rtype: Iterable
        """
        return self.data_inputs

    @di.setter
    def di(self, di: DIT):
        """
        Set data inputs.

        :param di: data inputs
        :type di: Iterable
        """
        self.data_inputs = di

    @property
    def eo(self) -> EOT:
        """
        Get expected outputs.

        If the expected outputs are None, will return a list of None of data_inputs length.

        :return: expected outputs
        :rtype: Iterable
        """
        if self.expected_outputs is None and (
            hasattr(self.data_inputs, '__len__') or hasattr(self.data_inputs, '__iter__')
        ):
            return [None] * len(self.data_inputs)
        return self.expected_outputs

    @eo.setter
    def eo(self, eo: EOT):
        """
        Set expected outputs.

        :param eo: expected outputs
        :type eo: Iterable
        """
        self.expected_outputs = eo

    @property
    def sdact(self) -> List['NamedDACTTuple']:
        """
        Get sub data containers.

        :return: sub data containers
        """
        return self.sub_data_containers

    @staticmethod
    def from_di(data_inputs: DIT) -> 'DACT[IDT, DIT, List[None]]':
        """
        Create a DataContainer (dact) from data inputs (di).
        """
        return DACT(
            ids=None,
            di=data_inputs,
            eo=None
        )

    @staticmethod
    def from_eo(expected_outputs: EOT) -> 'DACT[List[None], List[None], EOT]':
        """
        Create a DataContainer (dact) from expected outputs (eo).
        """
        return DACT(
            ids=None,
            di=None,
            eo=expected_outputs
        )

    def without_di(self) -> 'DACT[IDT, List[None], EOT]':
        return self.copy().set_data_inputs(None)

    def without_eo(self) -> 'DACT[IDT, DIT, List[None]]':
        return self.copy().set_expected_outputs(None)

    def with_di(self, di: DIT) -> 'DACT[IDT, DIT, EOT]':
        return self.copy().set_data_inputs(di)

    def with_eo(self, eo: EOT) -> 'DACT[IDT, DIT, EOT]':
        return self.copy().set_expected_outputs(eo)

    def with_ids(self, ids: DIT) -> 'DACT[IDT, DIT, EOT]':
        return self.copy().set_ids(ids)

    def set_ids(self, ids: IDT) -> 'DACT':
        """
        Set ids.

        :param ids: data inputs' ids. Often a range of integers.
        :return: self
        """
        self.ids = ids
        return self

    def set_data_inputs(self, data_inputs: DIT) -> 'DACT':
        """
        Set data inputs.

        :param data_inputs: data inputs
        :type data_inputs: Iterable
        :return: self
        """
        self.data_inputs = data_inputs
        return self

    def set_expected_outputs(self, expected_outputs: EOT) -> 'DACT':
        """
        Set expected outputs.

        :param expected_outputs: expected outputs
        :type expected_outputs: Iterable
        :return: self
        """
        self.expected_outputs: EOT = expected_outputs
        return self

    def get_ids_summary(self) -> Optional[str]:
        if self._ids is None:
            return None
        return ','.join([str(i) for i in self.ids if i is not None])

    def add_sub_data_container(self, name: str, data_container: 'DACT') -> 'DACT':
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

    def set_sub_data_containers(self, sub_data_containers: List['DACT']) -> 'DACT':
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
    ) -> Iterable['DACT[IDT, DIT, EOT]']:
        """
        Yields minibatches extracted from looping on the DataContainer's content with a batch_size and a certain behavior for the last batch when the batch_size is uneven with the total size.

        Note that the default value for IDs is None.

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
                default_value_data_inputs=StripAbsentValues()
            ):
                print(data_container_batch.data_inputs)
            # [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8]), array([9])]


        :param batch_size: number of elements to combine into a single batch
        :param keep_incomplete_batch: (Optional.) A bool representing
        whether the last batch should be dropped in the case it has fewer than
        `batch_size` elements; the default behavior is to keep the smaller
        batch.
        :param default_value_data_inputs: expected_outputs default fill value
        for padding and values outside iteration range, or :class:`~neuraxle.data_container.StripAbsentValues`
        to trim absent values from the batch
        :param default_value_expected_outputs: expected_outputs default fill value
        for padding and values outside iteration range, or :class:`~neuraxle.data_container.StripAbsentValues`
        to trim absent values from the batch
        :return: an iterator of DataContainer

        .. seealso::
            :class:`~neuraxle.data_container.StripAbsentValues`
        """
        for i in range(0, len(self.data_inputs), batch_size):
            data_container: DACT[IDT, DIT, EOT] = DACT(
                ids=self.ids[i:i + batch_size] if self._ids is not None else None,
                data_inputs=self.di[i:i + batch_size] if self.data_inputs is not None else None,
                expected_outputs=self.eo[i:i + batch_size] if self.expected_outputs is not None else None
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

    def copy(self) -> 'DACT[IDT, DIT, EOT]':
        return DACT(
            ids=self._ids,
            di=self.data_inputs,
            eo=self.expected_outputs,
            sub_data_containers=[(name, data_container.copy()) for name, data_container in self.sub_data_containers]
        )

    def __contains__(self, item: Union[str, int]) -> bool:
        """
        return true if sub container name is in the sub data containers, or if has ids of the good number.

        :param item: sub container name if string, id if int.
        :return: contains
        """
        if isinstance(item, str):
            contains = False
            for name, data_container in self.sub_data_containers:
                if name == item:
                    contains = True
            return contains
        else:
            if item in self.ids:
                return True
            else:
                return False
        raise NotImplementedError('DataContainer.__contains__ not implemented for this type: {}'.format(type(item)))

    def __getitem__(self, item: Union[str, int]) -> Union['DACT', Tuple[IDT, DIT, EOT]]:
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

    def tolist(self) -> 'DACT[List, List, List]':
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

    def tolistshallow(self) -> 'DACT[List, List, List]':
        return self.apply_conversion_func(list)

    def to_numpy(self) -> 'DACT[np.ndarray, np.ndarray, np.ndarray]':
        return self.apply_conversion_func(np.array)

    def apply_conversion_func(self, conversion_function: Callable[[Any], Any]) -> 'DACT':
        """
        Apply conversion function to data inputs, expected outputs, and ids,
        and set the new values in self. Returns self.
        Conversion function must be able to handle None values.
        """
        self.set_ids(
            conversion_function(self.ids)
            if self.ids is not None
            else None
        )
        self.set_data_inputs(
            conversion_function(self.data_inputs)
            if self.data_inputs is not None
            else None
        )
        self.set_expected_outputs(
            conversion_function(self.expected_outputs)
            if self.expected_outputs is not None
            else None
        )
        return self

    def unpack(self) -> Tuple[IDT, DIT, EOT]:
        """
        Unpack to a tuples of (ids, data input, expected output).

        :return: tuple of ids, data inputs, expected outputs
        """
        return self.ids, self.di, self.eo

    def __iter__(self) -> Iterator[Tuple[IDT, DIT, EOT]]:
        """
        Iter method returns a zip of all of the ids, data_inputs, and expected_outputs in the data container.

        :return: iterator of tuples containing ids, data_inputs, and expected outputs
        :rtype: Iterator[Tuple]
        """
        if self.data_inputs is None:
            return iter(())

        _ids: Optional[List[DACTData]] = self.ids
        _di: Optional[List[DACTData]] = self.di
        _eo: Optional[List[DACTData]] = self.eo
        if _ids is None or _di is None or _eo is None:
            return iter(())

        return zip(_ids, _di, _eo)

    def __repr__(self):
        return str(self)

    def __str__(self):
        ids = self._ids
        di = self.data_inputs
        eo = self.expected_outputs
        ids_rep = self._str_data(ids)
        di_rep = self._str_data(di)
        eo_rep = self._str_data(eo)
        return (
            f"{self.__class__.__name__}[{type(ids).__name__}, {type(di).__name__}, {type(eo).__name__}](\n"
            f"\tids={ids_rep},\n"
            f"\tdi={di_rep},\n"
            f"\teo={eo_rep}\n)"
        )

    def _str_data(self, _idata: DACTData) -> str:
        if _idata is None:
            return str(None)

        if len(_idata) > 10 and hasattr(_idata, '__getitem__'):
            _shortrepr = repr(_idata[:15])
        else:
            _shortrepr = repr(_idata)
        _shortrepr = _shortrepr[:70] + ("" if len(_shortrepr) < 70 else "...")

        _len = "len=?"
        if isinstance(_idata, pd.DataFrame):
            _len = f"shape={_idata.values.shape}"
        elif isinstance(_idata, np.ndarray):
            _len = f"shape={_idata.shape}"
        elif hasattr(_idata, "__len__"):
            _len = f"len={len(_idata)}"

        _len_rep = f"<`{_shortrepr}` of {_len}>"
        return _len_rep.replace("\n", " ").replace("\t", " ").replace("    ", " ")

    def __len__(self):
        return len(self.data_inputs)


DACT = DataContainer
TrainDACT = DACT[IDT, ARG_X_INPUTTED, ARG_Y_EXPECTED]  # training set input
ValidDACT = DACT[IDT, ARG_X_INPUTTED, ARG_Y_EXPECTED]  # validation set input
PredsDACT = DACT[IDT, ARG_Y_PREDICTD, Optional[EOT]]  # output after prediction

EvalEOTDACT = DACT[IDT, ARG_Y_PREDICTD, ARG_Y_EXPECTED]  # a merge of ValidDACT and PredsDACT: PRED Y and EXPECTED Y


class ExpandedDataContainer(DACT):
    """
    Sub class of DataContainer to expand data container dimension.
    This is akin from passing from `shape` to `[1, *shape]`
    when using :func:`ExpandedDataContainer.create_from`.

    .. seealso::
        :class:`DataContainer`
    """

    def __init__(self, data_inputs, ids, expected_outputs, _old_ids):
        DACT.__init__(
            self,
            ids=ids,
            di=data_inputs,
            eo=expected_outputs,
        )

        self._old_ids = _old_ids

    @staticmethod
    def create_from(data_container: DACT) -> 'ExpandedDataContainer':
        """
        Create ExpandedDataContainer with a summary id for the new id.
        This is akin from passing from `shape` to `[1, *shape]`.

        :param data_container: data container to transform
        :type data_container: DataContainer
        :return: expanded data container
        """
        return ExpandedDataContainer(
            ids=[data_container.get_ids_summary()],
            data_inputs=[data_container.data_inputs] if data_container.data_inputs is not None else None,
            expected_outputs=[data_container.expected_outputs] if data_container.expected_outputs is not None else None,
            _old_ids=data_container._ids
        )

    def reduce_dim(self) -> 'DACT':
        """
        Reduce DataContainer to its original shape with a list of multiple ids, data_inputs, and expected outputs.

        :return: reduced data container
        :rtype: DataContainer
        """
        if len(self.data_inputs) != 1:
            raise ValueError(
                'Invalid Expanded Data Container. Please create ExpandedDataContainer with ExpandedDataContainer.create_from(data_container) method.')

        return DACT(
            ids=self._old_ids,
            data_inputs=self.data_inputs[0] if self.data_inputs is not None else None,
            expected_outputs=self.expected_outputs[0] if self.expected_outputs is not None else None,
            sub_data_containers=self.sub_data_containers
        )


class ZipDataContainer(DACT):
    """
    Sub class of DataContainer to zip two data sources together.

    .. seealso::
        :class: `DataContainer`
    """

    @staticmethod
    def create_from(data_container: DACT, *other_data_containers: List[DACT], zip_expected_outputs: bool = False) -> 'ZipDataContainer':
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

        new_data_inputs = tuple(zip(*map(attrgetter("di"), [data_container] + list(other_data_containers))))
        if zip_expected_outputs:
            expected_outputs = tuple(
                zip(*map(attrgetter("eo"), [data_container] + list(other_data_containers))))
        else:
            expected_outputs = data_container.expected_outputs

        return ZipDataContainer(
            data_inputs=new_data_inputs,
            expected_outputs=expected_outputs,
            ids=data_container._ids,
            sub_data_containers=data_container.sub_data_containers
        )

    def concatenate_inner_features(self):
        """
        Concatenate inner features from zipped data inputs.
        Assumes each data_input entry is an iterable of numpy arrays.
        """
        new_data_inputs = [di[0] for di in self.di]

        for i, data_input in enumerate(self.di):
            new_data_inputs[i] = _inner_concatenate_np_array(list(data_input))

        self.set_data_inputs(new_data_inputs)


class ListDataContainer(DataContainer, Generic[IDT, DIT, EOT]):
    """
    Sub class of DataContainer to perform list operations.
    It allows to perform append, and concat operations on a DataContainer.

    .. seealso::
        :class:`DataContainer`
    """

    def __init__(self, data_inputs: Any, ids=None, expected_outputs: Any = None,
                 sub_data_containers=None):
        DACT.__init__(
            self,
            ids=ids,
            di=data_inputs,
            eo=expected_outputs,
            sub_data_containers=sub_data_containers
        )
        self.tolistshallow()

    @staticmethod
    def empty(original_data_container: DACT = None) -> 'ListDataContainer':
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
        self._ids.append(_id)
        self.data_inputs.append(data_input)
        self.expected_outputs.append(expected_output)

    def append_data_container_in_data_inputs(self, other: DACT) -> 'ListDataContainer':
        """
        Append a data container to the data inputs of this data container.

        :param other: data container
        :type other: DataContainer
        :return:
        """
        if not isinstance(other, DACT):
            raise ValueError(
                f"Expected data container of type {DACT.__name__}, "
                f"got {other.__class__.__name__}: {other}")
        self.data_inputs.append(other)
        self._ids.append(other.get_ids_summary())
        return self

    def append_data_container(self, other: DACT) -> 'ListDataContainer':
        """
        Append a data container to the DataContainer.

        :param other: data container
        :type other: DataContainer
        :return:
        """
        self._ids.append(other._ids)
        self.data_inputs.append(other.data_inputs)
        self.expected_outputs.append(other.expected_outputs)

        return self

    def extend(self, other: DACT):
        """
        Concat the given data container at the end of self so as to extend each IDs, DIs, and EOs.

        :param data_container: data container
        :type data_container: DataContainer
        :return:
        """
        other.tolistshallow()

        if self._ids is not None and other._ids is not None:
            self._ids.extend(other._ids)
        if self.data_inputs is not None and other.data_inputs is not None:
            self.data_inputs.extend(other.data_inputs)
        if self.expected_outputs is not None and other.expected_outputs is not None:
            self.expected_outputs.extend(other.expected_outputs)

        return self


def _pad_or_keep_incomplete_batch(
        data_container,
        batch_size,
        default_value_data_inputs,
        default_value_expected_outputs
) -> DACT:
    should_pad_right = not isinstance(
        default_value_data_inputs, StripAbsentValues
    ) or not isinstance(
        default_value_expected_outputs, StripAbsentValues)

    if should_pad_right:
        data_container = _pad_incomplete_batch(
            data_container,
            batch_size,
            default_value_data_inputs,
            default_value_expected_outputs
        )

    return data_container


def _pad_incomplete_batch(
        data_container: 'DACT',
        batch_size: int,
        default_value_data_inputs: Any,
        default_value_expected_outputs: Any
) -> DACT:
    pad_di = not isinstance(default_value_data_inputs, StripAbsentValues)
    pad_eo = not isinstance(default_value_expected_outputs, StripAbsentValues)
    pad_ids = pad_di and pad_eo

    data_container = DACT(
        ids=_pad_data(
            data_container.ids,
            default_value=None,
            batch_size=batch_size
        ) if pad_ids else data_container.ids,
        data_inputs=_pad_data(
            data_container.data_inputs,
            default_value=default_value_data_inputs,
            batch_size=batch_size
        ) if pad_di else data_container.data_inputs,
        expected_outputs=_pad_data(
            data_container.expected_outputs,
            default_value=default_value_expected_outputs,
            batch_size=batch_size
        ) if pad_eo else data_container.expected_outputs,
    )

    return data_container


def _pad_data(data: DACTData, default_value: Any, batch_size: int):
    if data is None:
        return None
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
    _temp = map(lambda x: list(x) + [0 for _ in range(target_n_dims - len(x))], _temp)
    target_dims_m1 = list(map(max, *list(_temp)))[:-1]

    for i in range(n_arrays):
        while len(np_arrays_to_concatenate[i].shape) < target_n_dims:
            np_arrays_to_concatenate[i] = np.expand_dims(np_arrays_to_concatenate[i], axis=-1)

        target_shape = tuple(target_dims_m1 + [np_arrays_to_concatenate[i].shape[-1]])
        np_arrays_to_concatenate[i] = np.broadcast_to(np_arrays_to_concatenate[i], target_shape)

    return np.concatenate(np_arrays_to_concatenate, axis=-1)
