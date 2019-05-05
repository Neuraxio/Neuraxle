"""
Hyperparameter Distributions
====================================
Here you'll find a few hyperparameter distributions. It's also possible to create yours by inheriting
from the base class. Each distribution must override the method `rvs`, which will return a sampled value from
the distribution.
"""

import math
import random
from abc import abstractmethod, ABCMeta
from typing import List

import numpy as np


class HyperparameterDistribution(metaclass=ABCMeta):
    """Base class for other hyperparameter distributions."""

    @abstractmethod
    def rvs(self):
        """
        Sample the random variable.

        :return: The randomly sampled value.
        """
        pass


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

    def __init__(self):
        """
        Create a random boolean hyperparameter.
        """
        pass

    def rvs(self):
        """
        Get a random True or False.

        :return: True or False (random).
        """
        return random.choice([True, False])


class Choice(HyperparameterDistribution):
    """Get a random value from a choice list of possible value for this hyperparameter."""

    def __init__(self, choice_list: List):
        """
        Create a random choice hyperparameter from the given list.

        :param choice_list: a list of values to sample from.
        """
        self.choice_list = choice_list

    def rvs(self):
        """
        Get one of the items randomly.

        :return: one of the items of the list.
        """
        return random.choice(self.choice_list)


class Quantized(HyperparameterDistribution):
    """A quantized wrapper for another distribution: will round() the rvs number."""

    def __init__(self, hd: HyperparameterDistribution):
        """
        Create a quantized distribution.
        This objects wrap another HyperparameterDistribution, rounding to the nearest integer.

        :param hd: the other HyperparameterDistribution to round-wrap.
        """
        self.hd: HyperparameterDistribution = hd

    def rvs(self) -> int:
        """
        Will return an integer, rounded from the output of the previous distribution.

        :return: an integer.
        """
        return round(self.hd.rvs())

    def __repr__(self):
        return self.__class__.__name__ + "(" + repr(self.hd) + ")"

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.hd) + ")"


class Uniform(HyperparameterDistribution):
    """Get a uniform distribution."""

    def __init__(self, min_included: int, max_included: int):
        """
        Create a random uniform distribution.
        A random float between the two values somehow inclusively will be returned.

        :param min_included: minimum integer, included.
        :param max_included: maximum integer, might be included - for more info, see https://docs.python.org/2/library/random.html#random.uniform
        """
        self.min_included = min_included
        self.max_included = max_included

    def rvs(self) -> float:
        """
        Will return a float value in the specified range as specified at creation.

        :return: a float.
        """
        return random.random() * (self.max_included - self.min_included) + self.min_included


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

    def rvs(self) -> int:
        """
        Will return an integer in the specified range as specified at creation.

        :return: an integer.
        """
        return random.randint(self.min_included, self.max_included)


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

    def rvs(self) -> float:
        """
        Will return a float value in the specified range as specified at creation.

        :return: a float.
        """
        return 2 ** random.uniform(self.log2_min_included, self.log2_max_included)


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

    def rvs(self) -> float:
        """
        Will return a float value in the specified range as specified at creation.

        :return: a float.
        """
        result = np.random.normal(self.mean, self.std)
        # TODO: replace hard_clip with malleable max and min?
        if self.hard_clip_max is not None:
            result = min(result, self.hard_clip_max)
        if self.hard_clip_min is not None:
            result = max(result, self.hard_clip_min)
        return result


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

    def rvs(self) -> float:
        """
        Will return a float value in the specified range as specified at creation.
        Note: the range at creation was in log space. The return value is after taking an exponent.

        :return: a float.
        """
        result = 2 ** np.random.normal(self.log2_space_mean, self.log2_space_std)
        if self.hard_clip_max is not None:
            result = min(result, self.hard_clip_max)
        if self.hard_clip_min is not None:
            result = max(result, self.hard_clip_min)
        return result
