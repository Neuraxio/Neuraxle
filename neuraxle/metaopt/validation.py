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
import operator
import random
import warnings
import itertools
from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import reduce
from typing import Any, List, Optional, Set, Tuple

import numpy as np
from neuraxle.base import ExecutionContext as CX
from neuraxle.data_container import DIT, EOT, IDT, DACTData
from neuraxle.data_container import DataContainer as DACT
from neuraxle.data_container import TrainDACT, ValidDACT
from neuraxle.hyperparams.distributions import (
    ContinuousHyperparameterDistribution, DiscreteHyperparameterDistribution)
from neuraxle.hyperparams.space import (FlatDict, HyperparameterSamples,
                                        HyperparameterSpace)
from neuraxle.metaopt.data.aggregates import Round
from neuraxle.metaopt.data.vanilla import BaseHyperparameterOptimizer


class RandomSearchSampler(BaseHyperparameterOptimizer):
    """
    AutoML Hyperparameter Optimizer that randomly samples the space of random variables.
    Please refer to :class:`AutoML` for a usage example.

    .. seealso::
        :class:`Trainer`,
        :class:`HyperparamsRepository`,
    """

    def __init__(self):
        BaseHyperparameterOptimizer.__init__(self)

    def find_next_best_hyperparams(self, round_scope: Round) -> HyperparameterSamples:
        """
        Randomly sample the next hyperparams to try.

        :param round_scope: round scope
        :return: next random hyperparams
        """
        return round_scope.hp_space.rvs()


class GridExplorationSampler(BaseHyperparameterOptimizer):
    """
    This hyperparameter space optimizer is similar to a grid search, however, it does
    try to greedily sample maximally different points in the space to explore it. This space
    optimizer has a fixed pseudorandom exploration method that makes the sampling reproductible.

    When over the expected_n_trials (if sampling too much), the sampler will turn
    to a non-seeded random search.

    It may be good for space exploration before a TPE or for unit tests.

    If the expected_n_trials is not set or set to 0, the sampler will guess its ideal
    sampling count and then switch to random search after that.
    """

    def __init__(self, expected_n_trials: int = 0, seed_i: int = 0):
        BaseHyperparameterOptimizer.__init__(self)
        self.expected_n_trials: int = expected_n_trials
        self._i: int = seed_i

    @staticmethod
    def estimate_ideal_n_trials(hp_space: HyperparameterSpace) -> int:
        _ges: GridExplorationSampler = GridExplorationSampler(expected_n_trials=1)
        _ges._reinitialize_grid(hp_space, [])

        _expected_n_trials = _ges.expected_n_trials
        # Then readjust the expected_n_trials to be a multiple of the number of hyperparameters:
        #     NOTE: TRIED A DOZEN OF POWS SQRT CUMSUMPROD AND SOPHISTICATED OTHER THINGS, AND FOUND THIS BEST ONE:
        _estimated_ideal_n_trials: int = math.ceil(0.29 * _expected_n_trials)
        # TODO: could add more samples to the grid lens to match the counts in _estimated_ideal_n_trials.

        return _estimated_ideal_n_trials

    def _reinitialize_grid(self, hp_space: HyperparameterSpace, previous_trials_hp: List[HyperparameterSamples]) -> HyperparameterSamples:
        """
        Update the grid exploration sampler.

        :param round_scope: round scope
        :return: next random hyperparams
        """
        self._n_sampled = 0
        self.flat_hp_grid_values: OrderedDict[str, List[Any]] = OrderedDict()
        self.flat_hp_grid_lens: List[int] = []
        # TODO: could make use of a ND array here to keep track of the grid exploration instead of using random too much. And a walk method picking the most L2-distant point, permutated over the last mod 3 samples to walk awkwardly like [mid, begin, fartest, side, other side] in the ND cube, also avoiding same-seen values.
        self._seen_hp_grid_values: Set[Tuple[int]] = set()

        self._generate_grid(hp_space)
        for flat_dict_sample in previous_trials_hp:
            self._reshuffle_grid()

            vals: Tuple[int] = tuple(flat_dict_sample.values())
            self._seen_hp_grid_values.add(vals)

    def find_next_best_hyperparams(self, round_scope: Round) -> HyperparameterSamples:
        """
        Sample the next hyperparams to try.

        :param round_scope: round scope
        :return: next hyperparams
        """
        self._reinitialize_grid(round_scope.hp_space, round_scope.get_all_hyperparams())

        _space_max = reduce(operator.mul, self.flat_hp_grid_lens, 1)

        if self._n_sampled >= max(self.expected_n_trials, _space_max):
            return RandomSearchSampler().find_next_best_hyperparams(round_scope)
        for _ in range(_space_max):
            i_grid_keys: Tuple[int] = tuple(self._gen_keys_for_grid())
            grid_values: OrderedDict[str, Any] = tuple(self[i_grid_keys].values())
            if grid_values in self._seen_hp_grid_values:
                self._reshuffle_grid()
                self._i += 1
            else:
                break

        # Second chance at picking unseen value yet.
        if grid_values in self._seen_hp_grid_values:
            # overwrite it:
            prod_values = list(itertools.product(*self.flat_hp_grid_values.values()))
            generator = random.Random(self._i)
            generator.shuffle(prod_values)
            for grid_values in prod_values:
                if grid_values not in self._seen_hp_grid_values:
                    break
            if grid_values in self._seen_hp_grid_values:
                grid_values = generator.choice(prod_values)

        # assert grid_values not in self._seen_hp_grid_values  # TODO: TMP.

        # value is finally chosen:
        self._seen_hp_grid_values.add(grid_values)
        _full_dict = OrderedDict([(a, b) for a, b in zip(
            self.flat_hp_grid_values.keys(),
            grid_values
        )])
        return HyperparameterSamples(_full_dict)

    def _generate_grid(self, hp_space: HyperparameterSpace):
        """
        Generate the grid of hyperparameters to pick from.

        :param hp_space: hyperparameter space
        """
        # Start with discrete params:
        for hp_name, hp_dist in hp_space.to_flat_dict().items():
            if isinstance(hp_dist, DiscreteHyperparameterDistribution):
                hp_samples: List[Any] = hp_dist.values()

                reordered_hp_samples = self._pseudo_shuffle_list(hp_samples)
                self.flat_hp_grid_values[hp_name] = reordered_hp_samples
                self.flat_hp_grid_lens.append(len(reordered_hp_samples))

        # Then fill the remaining continous params using the expected_n_trials:
        remainder: int = max(3, self.expected_n_trials)
        for hp_name, hp_dist in hp_space.to_flat_dict().items():
            if isinstance(hp_dist, ContinuousHyperparameterDistribution):
                hp_samples: List[Any] = [
                    hp_dist.mean(),
                    hp_dist.min() if hp_dist.is_discrete() else hp_dist._pseudo_min(),
                    hp_dist.max() if hp_dist.is_discrete() else hp_dist._pseudo_max(),
                    hp_dist.mean() + hp_dist.std(),
                    hp_dist.mean() - hp_dist.std(),
                    hp_dist.mean() + hp_dist.std() / 2,
                    hp_dist.mean() - hp_dist.std() / 2,
                    hp_dist.mean() + hp_dist.std() * 1.5,
                    hp_dist.mean() - hp_dist.std() * 1.5,
                    hp_dist.mean() + hp_dist.std() / 4,
                    hp_dist.mean() - hp_dist.std() / 4,
                    hp_dist.mean() + hp_dist.std() * 2.5,
                    hp_dist.mean() - hp_dist.std() * 2.5,
                ]

                def _ensure_unique(value: Any, _unique_set: Set[Any]) -> bool:
                    # remove duplicates such as when (or if) STD is of 0.
                    try:
                        ret = value not in _unique_set
                        _unique_set.add(value)
                        return ret
                    except BaseException:
                        return True
                _unique_set: Set[Any] = set()
                hp_samples: List[Any] = [
                    x for x in hp_samples[:remainder]
                    if x in hp_dist and not (math.isinf(x) or math.isnan(x)) and _ensure_unique(x, _unique_set)
                ]
                self.flat_hp_grid_values[hp_name] = hp_samples
                self.flat_hp_grid_lens.append(len(hp_samples))

        _estimated_ideal_n_trials: int = sum(self.flat_hp_grid_lens)
        _space_max = reduce(operator.mul, self.flat_hp_grid_lens, 1)

        if not (_estimated_ideal_n_trials <= self.expected_n_trials <= _space_max):
            _new_val = max(min(_space_max, self.expected_n_trials), _estimated_ideal_n_trials)
            warnings.warn(
                f"Warning: changed {self.__class__.__name__}.expected_n_trials from "
                f"{self.expected_n_trials} to {_new_val}. RandomSearch will be used "
                f"as a fallback past this point if needed.")
            self.expected_n_trials = _new_val

        self._i = 1

    @staticmethod
    def _pseudo_shuffle_list(x: List, seed: int = 0) -> list:
        """
        Shuffle a list to create a pseudo-random order that is interesting.
        """
        x = copy.copy(x)
        for i in reversed(range(len(x))):
            v = x[i]
            if (len(x) + i + seed) % 2 and (i + seed) != 0:
                del x[i]
                x.insert(1 - (seed % 2), v)
        return x

    def _gen_keys_for_grid(self) -> List[int]:
        """
        Generate the keys for the grid.

        :param i: index
        :return: keys
        """
        flat_idx: List[int] = []
        for _len in self.flat_hp_grid_lens:
            flat_idx.append(self._i % _len)
        return flat_idx

    def _reshuffle_grid(self, new_sample: FlatDict = None):
        """
        Reshuffling with pseudo-random seed the hyperparameters' values:
        """
        assert self._i != 0, "Cannot reshuffle the grid when _i is 0. Please generate grid first and increase _i."

        for seed, (k, v) in enumerate(self.flat_hp_grid_values.items()):
            if self._i % len(v) == 0:
                # reshuffling the grid for when it comes to the end of each of its sublist.
                # TODO: make a test for this to ensure that over an infinity of trials that the lists are shuffled evenly, or use a real and repeatable pseudorandom rng generator for it.
                new_v = self._pseudo_shuffle_list(v, seed % (self._i + 1) + self._i % (seed + 1))
                self.flat_hp_grid_values[k] = new_v

                if (seed + self._i / len(v)) % 3 == 0:
                    # reversing the grid at each 3 reshuffles + seed as it may sometimes skip the zero in the disorder.
                    self.flat_hp_grid_values[k] = list(reversed(self.flat_hp_grid_values[k]))
                    self._n_sampled += 1

        self._i += 1

    def __getitem__(self, i_grid_keys: List[int]) -> 'OrderedDict[str, Any]':
        """
        Access the keys for the grid.

        :param i_grid_keys: keys
        :return: hyperparams
        """
        flat_result: OrderedDict[str, Any] = OrderedDict()
        for j, hp_name in enumerate(self.flat_hp_grid_values.keys()):
            flat_result[hp_name] = self.flat_hp_grid_values[hp_name][i_grid_keys[j]]
        return flat_result


FoldsList = List  # A list over folds. Can contain DACTData or even DACTs or Tuples of DACTs.


class BaseValidationSplitter(ABC):
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
