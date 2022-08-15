"""
Validation
====================================
Classes for hyperparameter tuning, such as random search.

..
   Copyright 2022, Neuraxio Inc.

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
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Set, Tuple

import numpy as np
from neuraxle.base import ExecutionContext as CX
from neuraxle.data_container import DIT, EOT, IDT, DACTData
from neuraxle.data_container import DataContainer as DACT
from neuraxle.data_container import TrainDACT, ValidDACT

FoldsList = List  # A list over folds. Can contain DACTData or even DACTs or Tuples of DACTs.


class BaseValidationSplitter(ABC):

    def __init__(self, force_fixed_metric_expected_outputs: bool = False):
        """
        :param force_fixed_metric_expected_outputs: If True, the expected outputs provided at split time are used to compute the metric instead of their possibly modified version after passing through the pipeline. More info in the documentation of :func:`set_to_force_expected_outputs_for_scoring`.
        """
        self.force_fixed_metric_expected_outputs: bool = False

    def set_to_force_expected_outputs_for_scoring(self) -> 'BaseValidationSplitter':
        """
        Set self.force_fixed_metric_expected_outputs to True.

        Use this in case you do not want the pipeline to be able to
        affect the Y (expected_output) value throughout the fit or transform process. This is to have a way to
        force using the provided expected output for the calculation of metrics in the Trainer's epochs loop.

        Do not use this when the pipeline can change the expected_outputs, for instance within an autoencoder
        that would split a time series and set its own expected output inside the pipeline, such as where the
        initial expected_output would be none at split time, and then would be computed on the fly through the
        pipeline and would be expected to be used for the metrics after this computation.
        """
        self.force_fixed_metric_expected_outputs = True
        return self

    def split_dact(self, data_container: DACT, context: CX) -> FoldsList[Tuple[TrainDACT, ValidDACT]]:
        """
        Wrap a validation split function with a split data container function.
        A validation split function takes two arguments:  data inputs, and expected outputs.

        :param data_container: data container to split
        :return: a tuple of the train and validation data containers.
        """
        splits: FoldsList[Tuple[TrainDACT, ValidDACT]] = []

        data_folds: FoldsList[Tuple[DIT, EOT, IDT, DIT, EOT, IDT]] = list(zip(*self.split(
            data_container.data_inputs, data_container.ids, data_container.expected_outputs, context
        )))

        # Iterate on folds:
        for (train_di, train_eo, train_ids, valid_di, valid_eo, valid_ids) in data_folds:

            # TODO: use ListDACT?
            train_data_container_split: TrainDACT = TrainDACT(
                ids=train_ids,
                data_inputs=train_di,
                expected_outputs=train_eo
            )

            validation_data_container_split: ValidDACT = ValidDACT(
                ids=valid_ids,
                data_inputs=valid_di,
                expected_outputs=valid_eo
            )

            splits.append((train_data_container_split, validation_data_container_split))

        return splits

    @abstractmethod
    def split(
        self,
        data_inputs: DIT,
        ids: Optional[IDT] = None,
        expected_outputs: Optional[EOT] = None,
        context: Optional[CX] = None
    ) -> Tuple[FoldsList[DIT], FoldsList[EOT], FoldsList[IDT], FoldsList[DIT], FoldsList[EOT], FoldsList[IDT]]:
        """
        Train/Test split data inputs and expected outputs.

        :param data_inputs: data inputs
        :param ids: id associated with each data entry (optional)
        :param expected_outputs: expected outputs (optional)
        :param context: execution context (optional)
        :return: train_di, train_eo, train_ids, valid_di, valid_eo, valid_ids
        """
        pass


class ValidationSplitter(BaseValidationSplitter):
    """
    Create a function that splits data into a training, and a validation set.

    .. code-block:: python

        # create a validation splitter function with 80% train, and 20% validation
        validation_splitter(0.20)


    :param test_size: test size in float
    :return:
    """

    def __init__(self, validation_size: float):
        BaseValidationSplitter.__init__(self)
        self.validation_size = validation_size

    def split(
        self,
        data_inputs: DIT,
        ids: Optional[IDT] = None,
        expected_outputs: Optional[EOT] = None,
        context: Optional[CX] = None
    ) -> Tuple[FoldsList[DIT], FoldsList[EOT], FoldsList[IDT], FoldsList[DIT], FoldsList[EOT], FoldsList[IDT]]:

        return tuple([
            # The data goes from `DACTData` to `FoldsList[DACTData]`, as per the a single fold:
            [data] for data in self._full_validation_split(
                data_inputs=data_inputs,
                ids=ids,
                expected_outputs=expected_outputs
            )
        ])

    def _full_validation_split(
        self,
        data_inputs: Optional[DIT] = None,
        ids: Optional[IDT] = None,
        expected_outputs: Optional[EOT] = None
    ) -> Tuple[DIT, EOT, IDT, DIT, EOT, IDT]:
        """
        Split data inputs, and expected outputs into a single training set, and a single validation set.

        :param test_size: test size in float
        :param data_inputs: data inputs to split
        :param ids: ids associated with each data entry
        :param expected_outputs: expected outputs to split
        :return: train_di, train_eo, train_ids, valid_di, valid_eo, valid_ids
        """
        return (
            self._train_split(data_inputs),
            self._train_split(expected_outputs),
            self._train_split(ids),
            self._validation_split(data_inputs),
            self._validation_split(expected_outputs),
            self._validation_split(ids),
        )

    def _train_split(self, data_inputs: DACTData) -> DACTData:
        """
        Split training set.

        :param data_inputs: data inputs to split
        :return: train_data_inputs
        """
        if data_inputs is None:
            return None
        return data_inputs[0:self._get_index_split(data_inputs)]

    def _validation_split(self, data_inputs: DACTData) -> DACTData:
        """
        Split validation set.

        :param data_inputs: data inputs to split
        :return: validation_data_inputs
        """
        if data_inputs is None:
            return None
        return data_inputs[self._get_index_split(data_inputs):]

    def _get_index_split(self, data_inputs: DACTData) -> int:
        if self.validation_size < 0 or self.validation_size > 1:
            raise ValueError('validation_size must be a float in the range [0, 1].')
        return math.floor(len(data_inputs) * (1 - self.validation_size))


class KFoldCrossValidationSplitter(BaseValidationSplitter):
    """
    Create a function that splits data with K-Fold Cross-Validation resampling.

    .. code-block:: python

        # create a kfold cross validation splitter with 2 kfold
        kfold_cross_validation_split(0.20)


    :param k_fold: number of folds.
    :return:
    """

    def __init__(self, k_fold: int):
        BaseValidationSplitter.__init__(self)
        self._k_fold: int = k_fold

    def _get_k_fold(self, dact_data: DACTData = None) -> int:
        return self._k_fold

    def split(
        self,
        data_inputs: DIT,
        ids: Optional[IDT] = None,
        expected_outputs: Optional[EOT] = None,
        context: Optional[CX] = None
    ) -> Tuple[FoldsList[DIT], FoldsList[EOT], FoldsList[IDT], FoldsList[DIT], FoldsList[EOT], FoldsList[IDT]]:

        train_di, valid_di = self._kfold_cv_split(
            data_inputs)

        _n_folds = len(train_di)
        _empty_folds: Tuple[FoldsList] = [[None] * _n_folds] * 2

        train_ids, valid_ids = self._kfold_cv_split(
            ids) or copy.deepcopy(_empty_folds)

        train_eo, valid_eo = self._kfold_cv_split(
            expected_outputs) or copy.deepcopy(_empty_folds)

        return train_di, train_eo, train_ids, \
            valid_di, valid_eo, valid_ids

    def _kfold_cv_split(self, dact_data: DACTData) -> Tuple[FoldsList[DACTData], FoldsList[DACTData]]:
        """
        Split data with K-Fold Cross-Validation splitting.

        :param data_inputs: data inputs
        :param k_fold: number of folds
        :return: a tuple of lists of folds of train_data, and of lists of validation_data, each of length "k_fold".
        """
        if dact_data is None:
            return None

        train_splitted_data: List[DACTData] = []
        valid_splitted_data: List[DACTData] = []

        for fold_i in range(self._get_k_fold(dact_data)):
            train_slice, valid_slice = self._get_train_val_slices_at_fold_i(dact_data, fold_i)

            train_splitted_data.append(train_slice)
            valid_splitted_data.append(valid_slice)

        return train_splitted_data, valid_splitted_data

    def _get_train_val_slices_at_fold_i(self, dact_data: DACTData, fold_i: int) -> Tuple[DACTData, DACTData]:
        step = len(dact_data) / float(self._get_k_fold())
        a = int(step * fold_i)
        b = int(step * (fold_i + 1))
        b = min(b, len(dact_data))

        train_slice: DACTData = self._concat_fold_dact_data(dact_data[:a], dact_data[b:])
        valid_slice: DACTData = dact_data[a:b]  # held-out fold against the training data

        return train_slice, valid_slice

    def _concat_fold_dact_data(self, arr1: DACTData, arr2: DACTData) -> DACTData:
        if isinstance(arr1, (list, tuple)):
            return arr1 + arr2
        else:
            return np.concatenate((arr1, arr2), axis=0)


class AnchoredWalkForwardTimeSeriesCrossValidationSplitter(KFoldCrossValidationSplitter):
    """
    An anchored walk forward cross validation works by performing a forward rolling split.

    All training splits start at the beginning of the time series, and finish time varies.

    For the validation split it, will start after a certain time delay (if padding is set)
    after their corresponding training split.

    Data is expected to be an is a square nd.array of shape [batch_size, total_time_steps, ...].
    It can be N dimensions, such as 3D or more, but the time series axis is currently limited to `axis=1`.
    """

    def __init__(
        self,
        minimum_training_size,
        validation_window_size=None,
        padding_between_training_and_validation=0,
        drop_remainder=False,
    ):
        """
        Create a anchored walk forward time series cross validation object.

        The size of the validation split is defined by `validation_window_size`.
        The difference in start position between two consecutive validation split is also equal to
        `validation_window_size`.

        :param minimum_training_size: size of the smallest training split.
        :param validation_window_size: size of each validation split and also the time step taken between each
            forward roll, by default None. If None : It takes the value `minimum_training_size`.
        :param padding_between_training_and_validation: the size of the padding between the end of the training split
            and the start of the validation split, by default 0.
        :param drop_remainder: drop the last split if the last validation split does not coincide
            with a full validation_window_size, by default False.
        """
        self.minimum_training_size = minimum_training_size
        # If validation_window_size is None, we give the same value as training_window_size.
        self.validation_window_size = validation_window_size or self.minimum_training_size
        self.padding_between_training_and_validation = padding_between_training_and_validation
        self.drop_remainder = drop_remainder
        self._validation_initial_start = self.minimum_training_size + self.padding_between_training_and_validation

    def _get_k_fold(self, dact_data: DACTData = None) -> int:
        if self.drop_remainder:
            _round_func = math.floor
        else:
            _round_func = math.ceil
        k_folds = _round_func(
            (dact_data.shape[1] - self._validation_initial_start) / float(self.validation_window_size)
        )
        return k_folds

    def _get_train_val_slices_at_fold_i(self, dact_data: DACTData, fold_i: int) -> Tuple[DACTData, DACTData]:
        # dact_data of shape [batch_size, total_time_steps, ...].

        # first slice index is always 0 for anchored walk forward cross validation.
        a = self._get_beginning_at_fold_i(fold_i)
        b = int(fold_i * self.validation_window_size + self.minimum_training_size)
        b = min(b, dact_data.shape[1])
        train_slice: DACTData = dact_data[:, a:b]

        x = int(fold_i * self.validation_window_size + self._validation_initial_start)
        y = int(x + self.validation_window_size)
        y = min(y, dact_data.shape[1])
        valid_slice: DACTData = dact_data[:, x:y]  # held-out fold against the training data

        return train_slice, valid_slice

    def _get_beginning_at_fold_i(self, fold_i: int) -> int:
        """
        Get the start time of the training split at the given fold index.
        Here in the anchored splitter, it is always zero. This method is overwritten
        in the non-anchored version of the walk forward ts validation splitter
        """
        return 0


class WalkForwardTimeSeriesCrossValidationSplitter(AnchoredWalkForwardTimeSeriesCrossValidationSplitter):
    """
    Perform a classic walk forward cross validation by performing a forward rolling split.
    As opposed to the AnchoredWalkForwardTimeSeriesCrossValidationSplitter, this class
    has a train split that is always of the same size.

    All the training split have the same `validation_window_size` size. The start time and end time of each training
    split will increase identically toward the end at each forward split. Same principle apply with the validation
    split, where the start and end will increase in the same manner toward the end. Each validation split start after
    a certain time delay (if padding is set) after their corresponding training split.

    Notes: The data supported by this cross validation is nd.array of shape [batch_size, total_time_steps, n_features].
    The array can have an arbitrary number of dimension, but the time series axis is currently limited to `axis=1`.
    """

    def __init__(
        self,
        training_window_size,
        validation_window_size=None,
        padding_between_training_and_validation=0,
        drop_remainder=False
    ):
        """
        Create a classic walk forward time series cross validation object.

        The difference in start position between two consecutive validation split are equal to one
        `validation_window_size`.

        :param training_window_size: the window size of training split.
        :param validation_window_size: the window size of each validation split and also the time step taken between
            each forward roll, by default None. If None : It takes the value `training_window_size`.
        :param padding_between_training_and_validation: the size of the padding between the end of the training split
            and the start of the validation split, by default 0.
        :param drop_remainder: drop the last split if the last validation split does not coincide
            with a full validation_window_size, by default False.
        """
        AnchoredWalkForwardTimeSeriesCrossValidationSplitter.__init__(
            self,
            training_window_size,
            validation_window_size=validation_window_size,
            padding_between_training_and_validation=padding_between_training_and_validation,
            drop_remainder=drop_remainder
        )

    def _get_beginning_at_fold_i(self, fold_i: int) -> int:
        """
        Get the start time of the training split at the given fold index.
        """
        return int(fold_i * self.validation_window_size)
