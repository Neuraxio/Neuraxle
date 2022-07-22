"""
Neuraxle's Hyperparameter Optimizer Base Classes
====================================================

Not all hyperparameter optimizers are there, but the base can be found here.


.. seealso::
    :class:`~neuraxle.metaopt.hyperopt.tpe.TreeParzenEstimator`,


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


"""

import copy
import itertools
import math
import operator
import random
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import reduce
from typing import Any, List, Set, Tuple

from neuraxle.hyperparams.distributions import (
    ContinuousHyperparameterDistribution, DiscreteHyperparameterDistribution)
from neuraxle.hyperparams.space import (FlatDict, HyperparameterSamples,
                                        HyperparameterSpace)
from neuraxle.metaopt.data.reporting import RoundReport


class BaseHyperparameterOptimizer(ABC):

    @abstractmethod
    def find_next_best_hyperparams(self, _round: RoundReport, hp_space: HyperparameterSpace) -> HyperparameterSamples:
        """
        Find the next best hyperparams using previous trials, that is the
        whole :class:`neuraxle.metaopt.data.aggregate.Round`.

        :param round: a :class:`neuraxle.metaopt.data.aggregate.Round`
        :return: next hyperparameter samples to train on
        """
        raise NotImplementedError()


class HyperparameterSamplerStub(BaseHyperparameterOptimizer):

    def __init__(self, preconfigured_hp_samples: HyperparameterSamples):
        self.preconfigured_hp_samples = preconfigured_hp_samples

    def find_next_best_hyperparams(self, _round: RoundReport, hp_space: HyperparameterSpace) -> HyperparameterSamples:
        return self.preconfigured_hp_samples


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

    def find_next_best_hyperparams(self, _round: RoundReport, hp_space: HyperparameterSpace) -> HyperparameterSamples:
        """
        Randomly sample the next hyperparams to try.

        :param round: round report
        :return: next random hyperparams
        """
        return hp_space.rvs()


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

    def find_next_best_hyperparams(self, _round: RoundReport, hp_space: HyperparameterSpace) -> HyperparameterSamples:
        """
        Sample the next hyperparams to try.

        :param round_scope: round scope
        :return: next hyperparams
        """
        self._reinitialize_grid(hp_space, _round.get_all_hyperparams())

        _space_max = reduce(operator.mul, self.flat_hp_grid_lens, 1)

        if self._n_sampled >= max(self.expected_n_trials, _space_max):
            return RandomSearchSampler().find_next_best_hyperparams(_round, hp_space)
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
