"""
Hyperparameter Distributions
====================================
Here you'll find a few hyperparameter distributions. It's also possible to create yours by inheriting
from the base class. Each distribution must override the method ``rvs``, which will return a sampled value from
the distribution.

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
import random
import sys
from abc import abstractmethod, ABCMeta
from typing import List

import math
import numpy as np


class HyperparameterDistribution(metaclass=ABCMeta):
    """Base class for other hyperparameter distributions."""

    def __init__(self):
        """
        Create a HyperparameterDistribution. This method should still be called with super if it gets overriden.
        """
        self.first_id = id(self)

    @abstractmethod
    def rvs(self):
        """
        Sample the random variable.

        :return: The randomly sampled value.
        """
        pass

    def narrow_space_from_best_guess(self, best_guess, kept_space_ratio: float = 0.0) -> 'HyperparameterDistribution':
        """
        Takes a value that is estimated to be the best one of the space, and restrict the space near that value.
        By default, this function will completely replace the returned value by the new guess if not overriden.

        :param best_guess: the value towards which we want to narrow down the space.
        :param kept_space_ratio: what proportion of the space is kept. Should be between 0.0 and 1.0. Default is to keep only the best_guess (0.0).
        :return: a new HyperparameterDistribution object that has been narrowed down.
        """
        return FixedHyperparameter(best_guess).was_narrowed_from(kept_space_ratio, self)

    def was_narrowed_from(
            self, kept_space_ratio: float, original_hp: 'HyperparameterDistribution'
    ) -> 'HyperparameterDistribution':
        """
        Keep track of the original distribution to restore it.

        :param kept_space_ratio: the ratio which made the current object narrower than the ``original_hp``.
        :param original_hp: The original HyperparameterDistribution, which will be kept in a private variable for an eventual restore.
        :return: self.
        """
        self.kept_space_ratio_trace = (
                self.get_current_narrowing_value() *
                kept_space_ratio *
                original_hp.get_current_narrowing_value()
        )
        self.original_hp: HyperparameterDistribution = original_hp.unnarrow()
        return self

    def get_current_narrowing_value(self):
        if not hasattr(self, 'kept_space_ratio_trace'):
            self.kept_space_ratio_trace: float = 1.0
        return self.kept_space_ratio_trace

    def unnarrow(self) -> 'HyperparameterDistribution':
        """
        Return the original distribution before narrowing of the distribution. If the distribution was never narrowed,
        will return a copy of self.

        :return: the original HyperparameterDistribution before narrowing, or else self if the distribution is virgin.
        """
        if not hasattr(self, 'original_hp'):
            return copy.deepcopy(self)
        return copy.deepcopy(self.original_hp.unnarrow())

    def __eq__(self, other):
        return self.first_id == other.first_id


class FixedHyperparameter(HyperparameterDistribution):
    """This is an hyperparameter that won't change again, but that is still expressed as a distribution."""

    def __init__(self, value):
        """
        Create a still hyperparameter

        :param value: what will be returned by calling ``.rvs()``.
        """
        self.value = value
        super(FixedHyperparameter, self).__init__()

    def rvs(self):
        """
        Sample the non-random anymore value.

        :return: the value given at creation.
        """
        return self.value


# TODO: Mixin this or something:
# class DelayedAdditionOf(MalleableDistribution):
#     """A HyperparameterDistribution (MalleableDistribution mixin) that """
#
#     def __init__(self, *dists):
#         self.dists = dists
#
#     def rvs(self):
#         rvss = [d.rvs if hasattr(d, 'rvs') else d for d in self.dists]
#         return sum(rvss)
#
#
# class MalleableDistribution(metaclass=ABCMeta):
#     """An hyperparameter distribution to which it's possible to do additional math using defaut python operators."""
#
#     def __add__(self, other):
#         return DelayedAdditionOf(self, other)
#
# max min + - / * % ** // == != < > <= >=
#


class Boolean(HyperparameterDistribution):
    """Get a random boolean hyperparameter."""

    def rvs(self):
        """
        Get a random True or False.

        :return: True or False (random).
        """
        return random.choice([True, False])


class Choice(HyperparameterDistribution):
    """Get a random value from a choice list of possible value for this hyperparameter.

    When narrowed, the choice will only collapse to a single element when narrowed enough.
    For example, if there are 4 items in the list, only at a narrowing value of 0.25 that
    the first item will be kept alone.
    """

    def __init__(self, choice_list: List):
        """
        Create a random choice hyperparameter from the given list.

        :param choice_list: a list of values to sample from.
        """
        self.choice_list = choice_list
        super(Choice, self).__init__()

    def rvs(self):
        """
        Get one of the items randomly.

        :return: one of the items of the list.
        """
        return random.choice(self.choice_list)

    def narrow_space_from_best_guess(self, best_guess, kept_space_ratio: float = 0.0) -> HyperparameterDistribution:
        """
        Will narrow the space. If the cumulative kept_space_ratio gets to be under or equal to 1/len(choice_list),
        then the list is crunched to a single item as a FixedHyperparameter to reflect this narrowing.
        So once a small enough kept_space_ratio is reached, the list becomes a fixed unique item from the best guess.
        Otherwise, a deepcopy of self is returned.

        :param best_guess: the best item of the list to keep if truly narrowing.
        :param kept_space_ratio: the ratio of the space to keep.
        :return: a deepcopy of self, or else a FixedHyperparameter of the best_guess.
        """
        new_narrowing = self.get_current_narrowing_value() * kept_space_ratio

        if len(self.choice_list) == 0 or len(self.choice_list) == 1 or new_narrowing <= 1.0 / len(self.choice_list):
            return FixedHyperparameter(best_guess).was_narrowed_from(kept_space_ratio, self)

        return copy.deepcopy(self).was_narrowed_from(kept_space_ratio, self)

    def __len__(self):
        """
        Return the number of choices.

        :return: the number of choices.
        """
        return len(self.choice_list)


class PriorityChoice(HyperparameterDistribution):
    """Get a random value from a choice list of possible value for this hyperparameter.

    The first parameters are kept until the end when the list is narrowed (it is narrowed progressively),
    unless there is a best guess that surpasses some of the top choices.
    """

    def __init__(self, choice_list: List):
        """
        Create a random choice hyperparameter from the given list (choice_list).
        The first parameters in the choice_list will be kept longer when narrowing the space.

        :param choice_list: a list of values to sample from. First placed, first kept when space is narrowed.
        """
        self.choice_list = choice_list
        super(PriorityChoice, self).__init__()

    def rvs(self):
        """
        Get one of the items randomly.

        :return: one of the items of the list.
        """
        return random.choice(self.choice_list)

    def narrow_space_from_best_guess(self, best_guess, kept_space_ratio: float = 0.0) -> HyperparameterDistribution:
        """
        Will narrow the space. If the cumulative kept_space_ratio gets to be under or equal to 1-1/len(choice_list),
        then the list is crunched to discard the last items to reflect this narrowing.
        After a few narrowing (or a big one), the list may become a FixedHyperparameter.
        Otherwise if the list is unchanged, a deepcopy of self is returned.

        :param best_guess: the best item of the list, which will be brought back as the first item.
        :param kept_space_ratio: the ratio of the space to keep.
        :return: a deepcopy of self, or a subchoice of self, or else a FixedHyperparameter of the best_guess.
        """
        new_size = int(len(self) * kept_space_ratio + sys.float_info.epsilon)
        if (
                len(self.choice_list) == 0
                or len(self.choice_list) == 1
                or new_size <= 1
                or kept_space_ratio <= 1.0 / len(self.choice_list)
        ):
            return FixedHyperparameter(best_guess).was_narrowed_from(kept_space_ratio, self)

        # Bring best_guess to front
        idx = self.choice_list.index(best_guess)
        del self.choice_list[idx]
        self.choice_list = [best_guess] + self.choice_list

        # Narrowing of the list.
        maybe_reduced_list = self.choice_list[:new_size]
        return PriorityChoice(maybe_reduced_list).was_narrowed_from(kept_space_ratio, self)

    def __len__(self):
        """
        Return the number of choices.

        :return: the number of choices.
        """
        return len(self.choice_list)


class WrappedHyperparameterDistributions(HyperparameterDistribution):
    def __init__(self, hd: HyperparameterDistribution = None, hds: List[HyperparameterDistribution] = None):
        """
        Create a wrapper that will surround another HyperparameterDistribution.
        The wrapper might use one (hd) and/or many (hds) HyperparameterDistribution depending on the argument(s) used.

        :param hd: the other HyperparameterDistribution to wrap.
        :param hds: the others HyperparameterDistribution to wrap.
        """
        self.hd: HyperparameterDistribution = hd
        self.hds: List[HyperparameterDistribution] = hds
        super(WrappedHyperparameterDistributions, self).__init__()

    def __repr__(self):
        return self.__class__.__name__ + "(" + repr(self.hd) + ", hds=" + repr(self.hds) + ")"

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.hd) + ", hds=" + str(self.hds) + ")"


class Quantized(WrappedHyperparameterDistributions):
    """A quantized wrapper for another distribution: will round() the rvs number."""

    def rvs(self) -> int:
        """
        Will return an integer, rounded from the output of the previous distribution.

        :return: an integer.
        """
        return round(self.hd.rvs())

    def narrow_space_from_best_guess(self, best_guess, kept_space_ratio: float = 0.5) -> 'Quantized':
        """
        Will narrow the underlying distribution and re-wrap it under a Quantized.

        :param best_guess: the value towards which we want to narrow down the space.
        :param kept_space_ratio: what proportion of the space is kept. Default is to keep half the space (0.5).
        :return:
        """
        return Quantized(
            self.hd.narrow_space_from_best_guess(best_guess, kept_space_ratio)
        ).was_narrowed_from(kept_space_ratio, self)


class RandInt(HyperparameterDistribution):
    """Get a random integer within a range"""

    def __init__(self, min_included: int, max_included: int):
        """
        Create a quantized random uniform distribution.
        A random integer between the two values inclusively will be returned.

        :param min_included: minimum integer, included.
        :param max_included: maximum integer, included.
        """
        self.min_included = min_included
        self.max_included = max_included
        super(RandInt, self).__init__()

    def rvs(self) -> int:
        """
        Will return an integer in the specified range as specified at creation.

        :return: an integer.
        """
        return random.randint(self.min_included, self.max_included)

    def narrow_space_from_best_guess(self, best_guess, kept_space_ratio: float = 0.5) -> HyperparameterDistribution:
        """
        Will narrow the underlying distribution towards the best guess.

        :param best_guess: the value towards which we want to narrow down the space. Should be between 0.0 and 1.0.
        :param kept_space_ratio: what proportion of the space is kept. Default is to keep half the space (0.5).
        :return: a new HyperparameterDistribution that has been narrowed down.
        """
        lost_space_ratio = 1.0 - kept_space_ratio
        new_min_included = round(self.min_included * kept_space_ratio + best_guess * lost_space_ratio)
        new_max_included = round(self.max_included * kept_space_ratio + best_guess * lost_space_ratio)
        if new_max_included <= new_min_included or kept_space_ratio == 0.0:
            return FixedHyperparameter(best_guess).was_narrowed_from(kept_space_ratio, self)
        return RandInt(new_min_included, new_max_included).was_narrowed_from(kept_space_ratio, self)


class Uniform(HyperparameterDistribution):
    """Get a uniform distribution."""

    def __init__(self, min_included: float, max_included: float):
        """
        Create a random uniform distribution.
        A random float between the two values somehow inclusively will be returned.

        :param min_included: minimum integer, included.
        :param max_included: maximum integer, might be included - for more info, see https://docs.python.org/2/library/random.html#random.uniform
        """
        self.min_included: float = min_included
        self.max_included: float = max_included
        super(Uniform, self).__init__()

    def rvs(self) -> float:
        """
        Will return a float value in the specified range as specified at creation.

        :return: a float.
        """
        return random.random() * (self.max_included - self.min_included) + self.min_included

    def narrow_space_from_best_guess(self, best_guess, kept_space_ratio: float = 0.5) -> HyperparameterDistribution:
        """
        Will narrow the underlying distribution towards the best guess.

        :param best_guess: the value towards which we want to narrow down the space. Should be between 0.0 and 1.0.
        :param kept_space_ratio: what proportion of the space is kept. Default is to keep half the space (0.5).
        :return: a new HyperparameterDistribution that has been narrowed down.
        """
        lost_space_ratio = 1.0 - kept_space_ratio
        new_min_included = self.min_included * kept_space_ratio + best_guess * lost_space_ratio
        new_max_included = self.max_included * kept_space_ratio + best_guess * lost_space_ratio
        if new_max_included <= new_min_included or kept_space_ratio == 0.0:
            return FixedHyperparameter(best_guess).was_narrowed_from(kept_space_ratio, self)
        return Uniform(new_min_included, new_max_included).was_narrowed_from(kept_space_ratio, self)


class LogUniform(HyperparameterDistribution):
    """Get a LogUniform distribution.

    For example, this is good for neural networks' learning rates: that vary exponentially."""

    def __init__(self, min_included: float, max_included: float):
        """
        Create a quantized random log uniform distribution.
        A random float between the two values inclusively will be returned.

        :param min_included: minimum integer, should be somehow included.
        :param max_included: maximum integer, should be somehow included.
        """
        self.log2_min_included = math.log2(min_included)
        self.log2_max_included = math.log2(max_included)
        super(LogUniform, self).__init__()

    def rvs(self) -> float:
        """
        Will return a float value in the specified range as specified at creation.

        :return: a float.
        """
        return 2 ** random.uniform(self.log2_min_included, self.log2_max_included)

    def narrow_space_from_best_guess(self, best_guess, kept_space_ratio: float = 0.5) -> HyperparameterDistribution:
        """
        Will narrow, in log space, the distribution towards the new best_guess.

        :param best_guess: the value towards which we want to narrow down the space. Should be between 0.0 and 1.0.
        :param kept_space_ratio: what proportion of the space is kept. Default is to keep half the space (0.5).
        :return: a new HyperparameterDistribution that has been narrowed down.
        """
        log2_best_guess = math.log2(best_guess)
        lost_space_ratio = 1.0 - kept_space_ratio
        new_min_included = self.log2_min_included * kept_space_ratio + log2_best_guess * lost_space_ratio
        new_max_included = self.log2_max_included * kept_space_ratio + log2_best_guess * lost_space_ratio
        if new_max_included <= new_min_included or kept_space_ratio == 0.0:
            return FixedHyperparameter(best_guess).was_narrowed_from(kept_space_ratio, self)
        return LogUniform(2 ** new_min_included, 2 ** new_max_included).was_narrowed_from(kept_space_ratio, self)


class Normal(HyperparameterDistribution):
    """Get a normal distribution."""

    def __init__(self, mean: float, std: float,
                 hard_clip_min: float = None, hard_clip_max: float = None):
        """
        Create a normal distribution from mean and standard deviation.

        :param mean: the most common value to pop
        :param std: the standard deviation (that is, the sqrt of the variance).
        :param hard_clip_min: if not none, rvs will return max(result, hard_clip_min).
        :param hard_clip_max: if not none, rvs will return min(result, hard_clip_min).
        """
        self.mean = mean,
        self.std = std
        self.hard_clip_min = hard_clip_min
        self.hard_clip_max = hard_clip_max
        super(Normal, self).__init__()

    def rvs(self) -> float:
        """
        Will return a float value in the specified range as specified at creation.

        :return: a float.
        """
        result = float(np.random.normal(self.mean, self.std))
        if not math.isfinite(result):
            return self.rvs()
        # TODO: replace hard_clip with malleable max and min? also remove in doc if so (search for "hard clip").
        if self.hard_clip_max is not None:
            result = min(result, self.hard_clip_max)
        if self.hard_clip_min is not None:
            result = max(result, self.hard_clip_min)
        return float(result)

    def narrow_space_from_best_guess(self, best_guess, kept_space_ratio: float = 0.5) -> HyperparameterDistribution:
        """
        Will narrow the distribution towards the new best_guess.
        The mean will move towards the new best guess, and the standard deviation
        will be multiplied by the kept_space_ratio.
        The hard clip limit is unchanged.

        :param best_guess: the value towards which we want to narrow down the space's mean. Should be between 0.0 and 1.0.
        :param kept_space_ratio: what proportion of the space is kept. Default is to keep half the space (0.5).
        :return: a new HyperparameterDistribution that has been narrowed down.
        """
        lost_space_ratio = 1.0 - kept_space_ratio
        if isinstance(self.mean, tuple):
            self.mean = self.mean[0]
        new_mean = self.mean * kept_space_ratio + best_guess * lost_space_ratio
        new_std = self.std * kept_space_ratio
        if new_std <= 0.0:
            return FixedHyperparameter(best_guess).was_narrowed_from(kept_space_ratio, self)
        return Normal(
            new_mean, new_std, self.hard_clip_min, self.hard_clip_max
        ).was_narrowed_from(kept_space_ratio, self)


class LogNormal(HyperparameterDistribution):
    """Get a LogNormal distribution."""

    def __init__(self, log2_space_mean: float, log2_space_std: float,
                 hard_clip_min: float = None, hard_clip_max: float = None):
        """
        Create a LogNormal distribution. 

        :param log2_space_mean: the most common value to pop, but before taking 2**value.
        :param log2_space_std: the standard deviation of the most common value to pop, but before taking 2**value.
        :param hard_clip_min: if not none, rvs will return max(result, hard_clip_min). This value is not checked in logspace (so it is checked after the exp).
        :param hard_clip_max: if not none, rvs will return min(result, hard_clip_min). This value is not checked in logspace (so it is checked after the exp).
        """
        self.log2_space_mean = log2_space_mean
        self.log2_space_std = log2_space_std
        self.hard_clip_min = hard_clip_min
        self.hard_clip_max = hard_clip_max
        super(LogNormal, self).__init__()

    def rvs(self) -> float:
        """
        Will return a float value in the specified range as specified at creation.
        Note: the range at creation was in log space. The return value is after taking an exponent.

        :return: a float.
        """
        result = 2 ** float(np.random.normal(self.log2_space_mean, self.log2_space_std))
        if not math.isfinite(result):
            return self.rvs()
        if self.hard_clip_max is not None:
            result = min(result, self.hard_clip_max)
        if self.hard_clip_min is not None:
            result = max(result, self.hard_clip_min)
        return float(result)

    def narrow_space_from_best_guess(self, best_guess, kept_space_ratio: float = 0.5) -> HyperparameterDistribution:
        """
        Will narrow the distribution towards the new best_guess.
        The log2_space_mean (log space mean) will move, in log space, towards the new best guess, and the
        log2_space_std (log space standard deviation) will be multiplied by the kept_space_ratio.

        :param best_guess: the value towards which we want to narrow down the space's mean. Should be between 0.0 and 1.0.
        :param kept_space_ratio: what proportion of the space is kept. Default is to keep half the space (0.5).
        :return: a new HyperparameterDistribution that has been narrowed down.
        """
        log2_best_guess = math.log2(best_guess)
        lost_space_ratio = 1.0 - kept_space_ratio
        new_mean = self.log2_space_mean * kept_space_ratio + log2_best_guess * lost_space_ratio
        new_std = self.log2_space_std * kept_space_ratio
        if new_std <= 0.0:
            return FixedHyperparameter(best_guess).was_narrowed_from(kept_space_ratio, self)
        return Normal(
            new_mean, new_std, self.hard_clip_min, self.hard_clip_max
        ).was_narrowed_from(kept_space_ratio, self)
