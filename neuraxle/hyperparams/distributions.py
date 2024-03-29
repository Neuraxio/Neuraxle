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

..
    Thanks to Umaneo Technologies Inc. for their contributions to this Machine Learning
    project, visit https://www.umaneo.com/ for more information on Umaneo Technologies Inc.

"""
import math
import warnings
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, List, Optional

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, truncnorm


HPSampledValue = Any


class HyperparameterDistribution(metaclass=ABCMeta):
    """
    Base class for other hyperparameter distributions.

    This is the abstract class of the abstract classes.
    Please inherit from these mid-level abstract classes:

    - :class:`~neuraxle.hyperparams.distributions.ContinuousHyperparameterDistribution`
    - :class:`~neuraxle.hyperparams.distributions.DiscreteHyperparameterDistribution`
    - :class:`~neuraxle.hyperparams.distributions.OrdinalHyperparameterDistribution`
    """

    def __init__(self, null_default_value: Any = None):
        """
        Create a HyperparameterDistribution. This method should still be called with super if it gets overriden.
        """
        self.first_id = id(self)
        self.null_default_value = null_default_value

    @abstractmethod
    def is_discrete(self) -> bool:
        raise NotImplementedError("Please use a concrete class.")

    @abstractmethod
    def rvs(self) -> HPSampledValue:
        """
        Sample the random variable.

        :return: The randomly sampled value.
        """
        pass

    def rvs_many(self, size: int) -> List[HPSampledValue]:
        return [self.rvs() for _ in range(size)]

    def nullify(self) -> HPSampledValue:
        return self.null_default_value

    @abstractmethod
    def pdf(self, x: HPSampledValue) -> float:
        """
        Abstract method for probability distribution function value at `x`.

        :param x: value where the probability distribution function is evaluated.

        :return: The probability distribution function value.
        """
        pass

    @abstractmethod
    def cdf(self, x: HPSampledValue) -> float:
        """
        Abstract method for cumulative distribution function value at `x`.
        Therefore, `p = cdf(x)`.

        :param x: value where the cumulative distribution function is evaluated.

        :return: The cumulative distribution function value.
        """
        pass

    @abstractmethod
    def icdf(self, p: float) -> HPSampledValue:
        """
        This is the inverse of the cumulative distribution function,
        where x = icdf(p).

        :param x: value where the cumulative distribution function is evaluated.

        :return: The cumulative distribution function value.
        """
        pass

    @abstractmethod
    def min(self) -> float:
        """
        Abstract method for obtaining minimum value that can sampled in distribution.

        :return: minimal value that can be sampled from distribution.
        """
        pass

    @abstractmethod
    def max(self) -> float:
        """
        Abstract method for obtaining maximal value that can sampled in distribution.

        :return: maximal value that can be sampled from distribution.
        """
        pass

    @abstractmethod
    def mean(self) -> float:
        """
        Abstract method for calculating distribution mean value.

        :return: distribution mean value.
        """
        pass

    def std(self) -> float:
        """
        Base method to calculate distribution std value by taking sqrt of variance value.

        :return: distribution std value.
        """
        return math.sqrt(self.var())

    @abstractmethod
    def var(self) -> float:
        """
        Abstract method for calculate distribution variance value.

        : return: distribution variance value.
        """
        pass

    def __eq__(self, other):
        return self.first_id == other.first_id

    def __contains__(self, item) -> bool:
        """
        Check if item is in the distribution.
        The item is expected to be the result of a call to :func:`HyperparameterDistribution.rvs`.
        """
        return (
            item == self.null_default_value or
            (self.min() <= item <= self.max())
        )

    def __str__(self) -> str:
        return self.__class__.__name__ + f"<min={self.min()}, max={self.max()}>"


class ContinuousHyperparameterDistribution(HyperparameterDistribution):
    """
    Continuous distributions.

    .. seealso::
        :class:`~neuraxle.hyperparams.distributions.DiscreteHyperparameterDistribution`
        :class:`~neuraxle.hyperparams.distributions.OrdinalHyperparameterDistribution`
    """

    def __init__(self, null_default_value=None):
        HyperparameterDistribution.__init__(self, null_default_value=null_default_value)

    def is_discrete(self) -> bool:
        return False

    def icdf(self, p: float) -> HPSampledValue:
        """
        This is the inverse of the cumulative distribution function,
        where x = icdf(p).
        """
        _tolerance = 2e-11
        _tolerance *= (self._pseudo_max() - self._pseudo_min())  # tolerance is relative to range.
        x: HPSampledValue = self._icdf(p, self._pseudo_min(), self._pseudo_max(), _tolerance=2e-11)
        return x  # x is float since it's continuous.

    def _icdf(self, p: float, x_min: float, x_max: float, _tolerance=2e-11, _maxiters=100) -> float:

        if p <= 0.0:
            return x_min
        elif p >= 1.0:
            return x_max

        for _ in range(0, max(_maxiters, 1)):

            # Calculating the mean of the current interval
            x_mid = (x_min + x_max) / 2.0
            p_mid = self.cdf(x_mid)
            p_diff = p - p_mid

            # Exit condition
            if abs(p_diff) < _tolerance:
                return x_mid

            # Update rule for next iter: if p is greater than p_mid: increase x_min. Otherwise decrease x_max.
            if p_diff > 0:
                x_min = x_mid
            else:
                x_max = x_mid

        warnings.warn(
            f"Could not find a solution for icdf({p}): reached at most `{x_min} < x < {x_max}`. "
            f"returning midpoint: {x_mid}. ")
        return x_mid

    def _pseudo_min(self) -> float:
        """
        This is like the function min, but in case it is infinite, there will be a clipping to at least 4 sigmas.
        """
        x = self.min()
        if math.isinf(x) or math.isnan(x):
            x = self.mean() - 4 * self.std()
        return x

    def _pseudo_max(self) -> float:
        """
        This is like the function max, but in case it is infinite, there will be a clipping to at most 4 sigmas.
        """
        x = self.max()
        if math.isinf(x) or math.isnan(x):
            x = self.mean() + 4 * self.std()
        return x


class DiscreteHyperparameterDistribution(HyperparameterDistribution):
    """
    Discrete distributions.

    Ineritors will have to implement the following method at least:
    :func:`DiscreteHyperparameterDistribution.values`.

    .. seealso::
        :class:`~neuraxle.hyperparams.distributions.ContinuousHyperparameterDistribution`
        :class:`~neuraxle.hyperparams.distributions.OrdinalHyperparameterDistribution`
    """

    def __init__(self, null_default_value=None):
        HyperparameterDistribution.__init__(self, null_default_value=null_default_value)

    def is_discrete(self) -> bool:
        return True

    @abstractmethod
    def values(self) -> List[Any]:
        return [i for i in range(int(self.min()), int(self.max()) + 1)]

    def icdf(self, p: float) -> HPSampledValue:
        """
        This is the inverse of the cumulative distribution function,
        where x = icdf(p).
        """
        probas = np.array([self.cdf(i) for i in self.values()])
        closest_idx: int = np.argmin(np.abs(probas - p))
        x: HPSampledValue = self.values()[closest_idx]
        return x

    def probabilities(self) -> List[float]:
        return [self.pdf(i) for i in self.values()]

    def __contains__(self, item) -> bool:
        return item == self.null_default_value or item in self.values()


class OrdinalDiscreteHyperparameterDistribution(DiscreteHyperparameterDistribution):
    """
    Inherit from this class to represent a :class:`DiscreteHyperparameterDistribution`
    that contains values in :func:`DiscreteHyperparameterDistribution.values` that are
    in order such that they would sort the same way, either ascending or descending.

    The usage of this class indicates the AutoML wheter it should consider the different
    values like a categorical or a numerical variable.

    .. seealso::
        :class:`~neuraxle.hyperparams.distributions.ContinuousHyperparameterDistribution`
        :class:`~neuraxle.hyperparams.distributions.DiscreteHyperparameterDistribution`
    """

    def __init__(self, null_default_value=None):
        DiscreteHyperparameterDistribution.__init__(self, null_default_value=null_default_value)


class LogSpaceDistributionMixin:
    """
    Use this mixin when your distribution samples from a log-space distribution
    to identify it.
    """

    def __contains__(self, item) -> bool:
        return item != 0.0 and super().__contains__(item)


class FixedHyperparameter(DiscreteHyperparameterDistribution):
    """This is an hyperparameter that won't change again, but that is still expressed as a distribution."""

    def __init__(self, value, null_default_value=None):
        """
        Create a still hyperparameter

        :param value: what will be returned by calling ``.rvs()``.
        """
        DiscreteHyperparameterDistribution.__init__(self, null_default_value)
        self.value = value

    def rvs(self):
        """
        Sample the non-random anymore value.

        :return: the value given at creation.
        """
        return self.value

    def pdf(self, x) -> float:
        """
        Probability distribution function value at `x`.
        Since the parameter is fixed, the value return is 1 when x == value and 0 otherwise.

        :param x: value where the probability distribution function is evaluated.

        :return: The probability distribution function value.
        """
        if x == self.value:
            return 1.
        return 0.

    def cdf(self, x) -> float:
        """
        Cumulative distribution function value at `x`.
        Since the parameter is fixed, the value return is 1 if x>= value and 0 otherwise.

        :param x: value where the cumulative distribution function is evaluated.

        :return: The cumulative distribution function value.
        """
        if x >= self.value:
            return 1.

        return 0.

    def min(self):
        """
        Calculate minimum value that can be sampled in a fixed distribution.

        :return: minimal value return from distribution.
        """
        return self.value

    def max(self):
        """
        Calculate maximum value that can be sampled in a fixed distribution.

        :return: maximum value return from distribution.
        """
        return self.value

    def mean(self):
        """
        Calculate mean value (also called esperance) of the random variable.

        :return: mean value of the random variable.
        """
        return self.value

    def var(self):
        """
        Calculate variance value of the random variable.

        :return: variance value of the random variable.
        """
        return 0

    def __str__(self):
        return self.__class__.__name__ + f"(value={self.value}: {type(self.value)})"

    def values(self) -> List[Any]:
        return [self.value]


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


class Boolean(DiscreteHyperparameterDistribution):
    """Get a random boolean hyperparameter."""

    def __init__(self, proba_is_true: Optional[float] = None, null_default_value=False):
        """
        Create a boolean hyperparameter with given probability.

        Boolean distribution is in fact a Bernouilli distribution where given probability
        set occurance probability of having value 1. (1 - probability) gives occurance probability of having value 0.

        :param proba: a float corresponding to proportion of 1 over 0.
        :type proba: float
        :param null_default_value: default value for distribution
        :type null_default_value: default choice value. if None, default choice value will be the first choice
        """
        DiscreteHyperparameterDistribution.__init__(self, null_default_value)

        if proba_is_true is None:
            proba_is_true = 0.5

        if not (0 <= proba_is_true <= 1):
            raise ValueError("Probability must be between 0 and 1 (inclusively).")

        self.proba_is_true = proba_is_true

    def probabilities(self) -> List[float]:
        return [self.proba_is_true, 1 - self.proba_is_true]

    def values(self):
        return [True, False]

    def rvs(self) -> bool:
        """
        Get a random True or False.

        :return: True or False (random).
        """
        return bool(np.random.choice([False, True], p=[1 - self.proba_is_true, self.proba_is_true]))

    def pdf(self, x) -> float:
        """
        Calculate the boolean probability mass function value at position `x`.
        :param x: value where the probability mass function is evaluated.
        :return: value of the probability mass function.
        """
        if (x is True) or (x == 1):
            return self.proba_is_true

        if (x is False) or (x == 0):
            return 1 - self.proba_is_true

        return 0.

    def cdf(self, x) -> float:
        """
        Calculate the boolean cumulative distribution function value at position `x`.
        :param x: value where the cumulative distribution function is evaluated.
        :return: value of the cumulative distribution function.
        """
        if x < 0:
            return 0.

        if (0 <= x < 1) or (x is False):
            return 1 - self.proba_is_true

        if x >= 1 or (x is True):
            return 1.

        return 0.

    def min(self):
        """
        Calculate minimum value that can be sampled in a boolean distribution.

        :return: minimal value return from distribution.
        """
        return 0

    def max(self):
        """
        Calculate maximum value that can be sampled in a boolean distribution.

        :return: maximum value return from distribution.
        """
        return 1

    def mean(self):
        """
        Calculate mean value (also called esperance) of the random variable.

        :return: mean value of the random variable.
        """
        return self.proba_is_true

    def var(self):
        """
        Calculate variance value of the random variable.

        :return: variance value of the random variable.
        """
        return self.proba_is_true * (1 - self.proba_is_true)

    def __repr__(self):
        return self.__class__.__name__ + f"(proba_is_true={self.proba_is_true})"


class Choice(DiscreteHyperparameterDistribution):
    """Get a random value from a choice list of possible value for this hyperparameter.

    When narrowed, the choice will only collapse to a single element when narrowed enough.
    For example, if there are 4 items in the list, only at a narrowing value of 0.25 that
    the first item will be kept alone.
    """

    def __init__(self, choice_list: List[Any], probas: Optional[List[float]] = None, null_default_value=None):
        """
        Create a random choice hyperparameter from the given list.

        :param choice_list: a list of values to sample from.
        :type choice_list: List
        :param null_default_value: default value for distribution
        :type null_default_value: default choice value. if None, default choice value will be the first choice
        """
        DiscreteHyperparameterDistribution.__init__(self, null_default_value)

        self.choice_list: List[Any] = choice_list

        if probas is None:
            probas = [1 / len(self.choice_list) for _ in self.choice_list]

        # Normalize probas juste in case sum is not equal to one.
        self.probas: List[float] = (np.array(probas) / np.sum(probas)).tolist()

    def probabilities(self) -> List[float]:
        return self.probas

    def values(self) -> List[Any]:
        return self.choice_list

    def rvs(self):
        """
        Get one of the items randomly.

        :return: one of the items of the list.
        """
        choice_index = np.arange(0, len(self), 1)
        rvs_index = np.random.choice(choice_index, p=self.probas)
        return self.choice_list[rvs_index]

    def pdf(self, x) -> float:
        """
        Calculate the choice probability mass function value at position `x`.
        :param x: value where the probability mass function is evaluated.
        :return: value of the probability mass function.
        """
        try:
            x_in_choice = x in self.choice_list
        except (TypeError, ValueError, AttributeError):
            raise ValueError(
                "Item not find in list. Make sure the item is in the choice list and a correct method  __eq__ is defined for all item in the list.")
        else:
            if x_in_choice:
                index = get_index_in_list_with_bool(self.choice_list, x)
                return self.probas[index]

        return 0.

    def cdf(self, x) -> float:
        """
        Calculate the choice probability cumulative distribution function value at position `x`.
        The index in the list is used to know how the choice is performed.
        :param x: value where the cumulative distribution function is evaluated.
        :return: value of the cumulative distribution function.
        """
        try:
            index = get_index_in_list_with_bool(self.choice_list, x)
        except ValueError:
            raise ValueError(
                "Item not found in list. Make sure the item is in the choice list and a correct method  __eq__ is defined for all item in the list.")
        except (NotImplementedError, NotImplemented):
            raise ValueError("A correct method for __eq__ should be defined for all item in the list.")
        except AttributeError:
            raise ValueError("choice_list param should be a list.")
        else:
            probas = np.array(self.probas)
            return np.sum(probas[0:index + 1])

    def __len__(self):
        """
        Return the number of choices.

        :return: the number of choices.
        """
        return len(self.choice_list)

    def min(self):
        """
        Calculate minimum value that can be sampled in a choice distribution.

        Here the minimal index in the list is return. In this case it returns 0.

        :return: minimal value return from distribution.
        """
        return 0.

    def max(self):
        """
        Calculate maximal value that can be sampled in a choice distribution.

        Here the maximal index in the list is return. In this case it returns the length value of the list.

        :return: maximal value return from distribution.
        """
        return len(self) - 1

    def mean(self):
        """
        Calculate mean value (also called esperance) of the random variable.

        :return: mean value of the random variable.
        """
        choice_index = np.arange(0, len(self), 1)
        probas = np.array(self.probas)
        mean = np.sum(choice_index * probas)
        return mean

    def var(self):
        """
        Calculate variance value of the random variable.

        :return: variance value of the random variable.
        """
        choice_index = np.arange(0, len(self), 1)
        probas = np.array(self.probas)
        mean = np.sum(choice_index * probas)
        second_moment = np.sum(choice_index**2 * probas)
        var = second_moment - mean**2
        return var

    def __str__(self):
        return self.__class__.__name__ + \
            f"({'choice_list='+', '.join(self.choice_list) if len(self.choice_list) < 4 else f'<a {len(self.choice_list)} elements list>'})"


class PriorityChoice(OrdinalDiscreteHyperparameterDistribution):
    """
    Get a random value from a choice list of possible value for this hyperparameter.

    This is a choice that is ordinal and where the first choice is the most important one, as well as the default.

    The first parameters are kept until the end when the list is narrowed (it is narrowed progressively),
    unless there is a best guess that surpasses some of the top choices.
    """

    def __init__(self, choice_list: List[Any], probas: Optional[List[float]] = None, null_default_value=None):
        """
        Create a random choice hyperparameter from the given list (choice_list).
        The first parameters in the choice_list will be kept longer when narrowing the space.

        :param choice_list: a list of values to sample from.
        :type choice_list: List
        :param null_default_value: default value for distribution
        :type null_default_value: default choice value. if None, default choice value will be the first choice
        """
        if null_default_value is None:
            null_default_value = choice_list[0]
        elif null_default_value in choice_list:
            null_default_value = null_default_value
        else:
            raise ValueError(
                'invalid default value {0} not in choice list : {1}'.format(null_default_value, choice_list))

        OrdinalDiscreteHyperparameterDistribution.__init__(self, null_default_value)
        self.choice_list: List[Any] = choice_list

        if probas is None:
            probas = [1 / len(self.choice_list) for _ in self.choice_list]

        # Normalize probas juste in case sum is not equal to one.
        self.probas: List[float] = list(np.array(probas) / np.sum(probas))

    def probabilities(self) -> List[float]:
        return self.probas

    def values(self) -> List:
        return self.choice_list

    def rvs(self):
        """
        Get one of the items randomly.

        :return: one of the items of the list.
        """
        choice_index = np.arange(0, len(self), 1)
        rvs_index = np.random.choice(choice_index, p=self.probas)
        return self.choice_list[rvs_index]

    def pdf(self, x) -> float:
        """
        Calculate the choice probability mass function value at position `x`.
        :param x: value where the probability mass function is evaluated.
        :return: value of the probability mass function.
        """
        try:
            x_in_choice = x in self.choice_list
        except (TypeError, ValueError, AttributeError):
            raise ValueError(
                "Item not find in list. Make sure the item is in the choice list and a correct method  __eq__ is defined for all item in the list.")
        else:
            if x_in_choice:
                index = get_index_in_list_with_bool(self.choice_list, x)
                return self.probas[index]

        return 0.

    def cdf(self, x) -> float:
        """
        Calculate the choice probability cumulative distribution function value at position `x`.
        The index in the list is used to know how the choice is performed.
        :param x: value where the cumulative distribution function is evaluated.
        :return: value of the cumulative distribution function.
        """
        try:

            index = get_index_in_list_with_bool(self.choice_list, x)
        except ValueError:
            raise ValueError(
                "Item not find in list. Make sure the item is in the choice list and a correct method  __eq__ is defined for all item in the list.")
        except (NotImplementedError, NotImplemented):
            raise ValueError("A correct method for __eq__ should be defined for all item in the list.")
        except AttributeError:
            raise ValueError("choice_list param should be a list.")
        else:
            return np.sum(self.probas[0:index + 1])

    def __len__(self):
        """
        Return the number of choices.

        :return: the number of choices.
        """
        return len(self.choice_list)

    def min(self):
        """
        Calculate minimum value that can be sampled in a priority choice distribution.

        Here the minimal index in the list is return. In this case it returns 0.

        :return: minimal value return from distribution.
        """
        return 0.

    def max(self):
        """
        Calculate maximal value that can be sampled in a priority choice distribution.

        Here the maximal index in the list is return. In this case it returns the length value of the list.

        :return: maximal value return from distribution.
        """
        return len(self) - 1

    def mean(self):
        """
        Calculate mean value (also called esperance) of the random variable.

        :return: mean value of the random variable.
        """
        choice_index = np.arange(0, len(self), 1)
        probas = np.array(self.probas)
        mean = np.sum(choice_index * probas)
        return mean

    def var(self):
        """
        Calculate variance value of the random variable.

        :return: variance value of the random variable.
        """
        choice_index = np.arange(0, len(self), 1)
        probas = np.array(self.probas)
        mean = np.sum(choice_index * probas)
        second_moment = np.sum(choice_index ** 2 * probas)
        var = second_moment - mean ** 2
        return var


class WrappedDiscreteHyperparameterDistribution(DiscreteHyperparameterDistribution):
    def __init__(self, hd: HyperparameterDistribution = None, hds: List[HyperparameterDistribution] = None,
                 null_default_value=None):
        """
        Create a wrapper that will surround another HyperparameterDistribution.
        The wrapper might use one (hd) and/or many (hds) HyperparameterDistribution depending on the argument(s) used.

        :param hd: the other HyperparameterDistribution to wrap.
        :param hds: the others HyperparameterDistribution to wrap.
        """
        DiscreteHyperparameterDistribution.__init__(self, null_default_value)
        self.hd: HyperparameterDistribution = hd

    def __repr__(self):
        return self.__class__.__name__ + "(" + repr(self.hd) + ")"

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.hd) + ")"


class Quantized(WrappedDiscreteHyperparameterDistribution):
    """
    A quantized wrapper for another distribution: will round() the returned :func:`HyperparameterDistribution.rvs` number.
    """

    def rvs(self) -> int:
        """
        Will return an integer, rounded from the output of the previous distribution.

        :return: an integer.
        """
        return round(self.hd.rvs())

    def pdf(self, x) -> float:
        """
        Calculate the Quantized probability mass function value at position `x` of a continuous distribution.
        :param x: value where the probability mass function is evaluated.
        :return: value of the probability mass function.
        """
        # In order to calculate the pdf for any quantized distribution,
        # we have to perform the integral from x-0.5 to x+0.5 (because of round).
        if isinstance(x, int) or (isinstance(x, float) and x.is_integer()):
            return quad(self.hd.pdf, x - 0.5, x + 0.5)[0]
        return 0.

    def cdf(self, x) -> float:
        """
        Calculate the Quantized cumulative distribution function at position `x` of a continuous distribution.
        :param x: value where the cumulative distribution function is evaluated.
        :return: value of the cumulative distribution function.
        """
        # In order to calculate the cdf for any quantized distribution, we have to take the cdf at x + 0.5.
        return self.hd.cdf(math.floor(x) + 0.5)

    def min(self):
        """
        Calculate minimum value that can be sampled in the quanitzed version of the distribution.

        :return: minimal value return from distribution.
        """
        hd_min = self.hd.min()
        if np.isneginf(hd_min):
            return hd_min
        return round(hd_min)

    def max(self):
        """
        Calculate maximal value that can be sampled in the quantized version of the distribution.

        :return: maximal value return from distribution.
        """
        hd_max = self.hd.max()
        if np.isposinf(hd_max):
            return hd_max
        return round(hd_max)

    def mean(self) -> float:
        """
        Calculate mean value (also called esperance) of the random variable.

        :return: mean value of the random variable.
        """
        # We will perform two sum starting from floor(mean of original distribution) and from ceil(mean of orignal  distribution).
        original_mean = self.hd.mean()

        # First sum
        starting_decreasing_value = math.floor(original_mean)
        decreasing_sum = _calculate_sum(lambda x: x * self.pdf(x), [starting_decreasing_value, -np.inf])

        # Second sum
        starting_increasing_value = math.ceil(original_mean)
        if (starting_increasing_value - starting_decreasing_value) < 1e-10:
            starting_increasing_value += 1

        increasing_sum = _calculate_sum(lambda x: x * self.pdf(x), [starting_increasing_value, np.inf])

        return decreasing_sum + increasing_sum

    def var(self) -> float:
        """
        Calculate variance value of the random variable.

        :return: variance value of the random variable.
        """
        # first moment
        first_moment = self.mean()

        # Calculate second moment
        # We will perform two sum starting from floor(mean of original distribution) and from ceil(mean of orignal  distribution).
        original_mean = self.hd.mean()
        # First sum
        starting_decreasing_value = math.floor(original_mean)
        decreasing_sum = _calculate_sum(lambda x: x ** 2 * self.pdf(x), [starting_decreasing_value, -np.inf])
        # Second sum
        starting_increasing_value = math.ceil(original_mean)
        if (starting_increasing_value - starting_decreasing_value) < 1e-10:
            starting_increasing_value += 1
        increasing_sum = _calculate_sum(lambda x: x ** 2 * self.pdf(x), [starting_increasing_value, np.inf])

        second_moment = decreasing_sum + increasing_sum

        return second_moment - first_moment ** 2

    def values(self) -> List[Any]:
        return [i for i in range(int(self.min()), int(self.max()) + 1)]


class RandInt(DiscreteHyperparameterDistribution):
    """Get a random integer within a range"""

    def __init__(self, min_included: int, max_included: int, null_default_value: int = None):
        """
        Create a quantized random uniform distribution.
        A random integer between the two values inclusively will be returned.

        :param min_included: minimum integer, included.
        :param max_included: maximum integer, included.
        :param null_default_value: null default value for distribution. if None, take the min_included
        :type null_default_value: int
        """
        if null_default_value is None:
            DiscreteHyperparameterDistribution.__init__(self, min_included)
        else:
            DiscreteHyperparameterDistribution.__init__(self, null_default_value)

        self.min_included = min_included
        self.max_included = max_included

    def probabilities(self):
        values = self.values()
        return [1 / len(values) for _ in values]

    def values(self):
        return [i for i in range(self.min_included, self.max_included + 1)]

    def rvs(self) -> int:
        """
        Will return an integer in the specified range as specified at creation.

        :return: an integer.
        """
        return np.random.randint(self.min_included, self.max_included + 1)

    def pdf(self, x) -> float:
        """
        Calculate the random int mass function value at position `x`.
        :param x: value where the probability mass function is evaluated.
        :return: value of the probability mass function.
        """

        possible_values = set(range(self.min_included, self.max_included + 1))
        if (isinstance(x, int) or x.is_integer()) and x in possible_values:
            return 1 / (self.max_included - self.min_included + 1)

        return 0.

    def cdf(self, x) -> float:
        """
        Calculate the random int cumulative distribution function value at position `x`.
        :param x: value where the cumulative distribution function is evaluated.
        :return: value of the cumulative distribution function.
        """
        if x < self.min_included:
            return 0.

        if x > self.max_included:
            return 1.

        return (math.floor(x) - self.min_included + 1) / (self.max_included - self.min_included + 1)

    def min(self):
        """
        Calculate minimum value that can be sampled in the randint distribution.

        :return: minimal value return from distribution.
        """
        return self.min_included

    def max(self):
        """
        Calculate maximal value that can be sampled in the randint distribution.

        :return: maximal value return from distribution.
        """
        return self.max_included

    def mean(self):
        """
        Calculate mean value (also called esperance) of the random variable.

        :return: mean value of the random variable.
        """
        return (self.max_included + self.min_included) / 2

    def var(self):
        """
        Calculate variance value of the random variable.

        :return: variance value of the random variable.
        """
        return ((self.max_included - self.min_included + 1) ** 2 - 1) / 12

    def __str__(self):
        return self.__class__.__name__ + "(" + "max_included=" + str(self.max_included) + ", min_included=" + str(self.min_included) + ")"


class Uniform(ContinuousHyperparameterDistribution):
    """Get a uniform distribution."""

    def __init__(self, min_included: float, max_included: float, null_default_value=None):
        """
        Create a random uniform distribution.
        A random float between the two values somehow inclusively will be returned.

        :param min_included: minimum integer, included.
        :type min_included: float
        :param max_included: maximum integer, might be included - for more info, see `examples <https://docs.python.org/2/library/random.html#random.uniform>`__
        :type max_included: float
        :param null_default_value: null default value for distribution. if None, take the min_included
        :type null_default_value: int
        """
        if null_default_value is None:
            ContinuousHyperparameterDistribution.__init__(self, min_included)
        else:
            ContinuousHyperparameterDistribution.__init__(self, null_default_value)

        self.min_included: float = min_included
        self.max_included: float = max_included

    def rvs(self) -> float:
        """
        Will return a float value in the specified range as specified at creation.

        :return: a float.
        """
        return np.random.random() * (self.max_included - self.min_included) + self.min_included

    def pdf(self, x):
        """
        Calculate the Uniform probability distribution value at position `x`.

        :param x: value where the probability distribution function is evaluated.
        :return: value of the probability distribution function.
        """

        if self.min_included == self.max_included and (x == self.min_included):
            return 1.

        if (x >= self.min_included) and (x <= self.max_included):
            return 1 / (self.max_included - self.min_included)

        # Manage case where it is outside the probability distribution function.
        return 0.

    def cdf(self, x):
        """
        Calculate the Uniform cumulative distribution value at position `x`.
        :param x: value where the cumulative distribution function is evaluated.
        :return: value of the cumulative distribution function.
        """

        if self.min_included == self.max_included and (x == self.min_included):
            return 1.

        if x < self.min_included:
            return 0.

        if (x >= self.min_included) and (x <= self.max_included):
            return (x - self.min_included) / (self.max_included - self.min_included)

        # Manage the case where x_value > self.max_included
        return 1.

    def min(self):
        """
        Calculate minimum value that can be sampled in the uniform distribution.

        :return: minimal value return from distribution.
        """
        return self.min_included

    def max(self):
        """
        Calculate maximal value that can be sampled in the uniform distribution.

        :return: maximal value return from distribution.
        """
        return self.max_included

    def mean(self):
        """
        Calculate mean value (also called esperance) of the random variable.

        :return: mean value of the random variable.
        """
        return (self.min_included + self.max_included) / 2

    def var(self):
        """
        Calculate variance value of the random variable.

        :return: variance value of the random variable.
        """
        return (self.max_included - self.min_included) ** 2 / 12

    def __str__(self):
        return self.__class__.__name__ + "(" + "max_included=" + str(self.max_included) + ", min_included=" + str(self.min_included) + ")"


class LogUniform(LogSpaceDistributionMixin, ContinuousHyperparameterDistribution):
    """Get a LogUniform distribution.

    For example, this is good for neural networks' learning rates: that vary exponentially."""

    def __init__(self, min_included: float, max_included: float, null_default_value=None):
        """
        Create a quantized random log uniform distribution.
        A random float between the two values inclusively will be returned.

        :param min_included: minimum integer, should be somehow included.
        :param max_included: maximum integer, should be somehow included.
        :param null_default_value: null default value for distribution. if None, take the min_included
        :type null_default_value: int
        """
        if null_default_value is None:
            ContinuousHyperparameterDistribution.__init__(self, math.log2(min_included))
        else:
            ContinuousHyperparameterDistribution.__init__(self, math.log2(null_default_value))

        self.min_included: float = min_included
        self.max_included: float = max_included
        self.log2_min_included = math.log2(min_included)
        self.log2_max_included = math.log2(max_included)

    def rvs(self) -> float:
        """
        Will return a float value in the specified range as specified at creation.

        :return: a float.
        """
        return 2 ** np.random.uniform(self.log2_min_included, self.log2_max_included)

    def pdf(self, x) -> float:
        """
        Calculate the logUniform probability distribution value at position `x`.
        :param x: value where the probability distribution function is evaluated.
        :return: value of the probability distribution function.
        """
        if (self.log2_min_included == self.log2_max_included) and x == 2 ** self.log2_min_included:
            return 1.

        if (x >= 2 ** self.log2_min_included) and (x <= 2 ** self.log2_max_included):
            return 1 / (x * math.log(2) * (self.log2_max_included - self.log2_min_included))

        return 0.

    def cdf(self, x) -> float:
        """
        Calculate the logUniform cumulative distribution value at position `x`.
        :param x: value where the cumulative distribution function is evaluated.
        :return: value of the cumulative distribution function.
        """
        if x < 2 ** self.log2_min_included:
            return 0.

        if (self.log2_min_included == self.log2_max_included) and x == 2 ** self.log2_min_included:
            return 1.

        if (x >= 2 ** self.log2_min_included) and (x <= 2 ** self.log2_max_included):
            return (math.log2(x) - self.log2_min_included) / (self.log2_max_included - self.log2_min_included)

        # Manage the case x > 2**self.log2_max_included
        return 1.

    def min(self):
        """
        Calculate minimum value that can be sampled in the LogUniform distribution.

        :return: minimal value return from distribution.
        """
        return self.min_included

    def max(self):
        """
        Calculate maximal value that can be sampled in the LogUniform distribution.

        :return: maximal value return from distribution.
        """
        return self.max_included

    def mean(self):
        """
        Calculate mean value (also called esperance) of the random variable.

        :return: mean value of the random variable.
        """
        if self.log2_min_included == self.log2_max_included:
            return self.log2_max_included

        return (self.max_included - self.min_included) / (
            math.log(2) * (self.log2_max_included - self.log2_min_included))

    def var(self):
        """
        Calculate variance value of the random variable.

        :return: variance value of the random variable.
        """
        if self.log2_min_included == self.log2_max_included:
            return 0.

        esperance_squared = (self.max_included ** 2 - self.min_included ** 2) / (
            2 * math.log(2) * (self.log2_max_included - self.log2_min_included))
        return esperance_squared - (self.mean() ** 2)

    def __str__(self):
        return self.__class__.__name__ + "(" + "max_included=" + str(self.max_included) + ", min_included=" + str(self.min_included) + ")"


class Normal(ContinuousHyperparameterDistribution):
    """Get a normal distribution."""

    def __init__(self, mean: float, std: float,
                 hard_clip_min: float = None, hard_clip_max: float = None, null_default_value: float = None):
        """
        Create a normal distribution from mean and standard deviation.

        :param mean: the most common value to pop
        :type mean: float
        :param std: the standard deviation (that is, the sqrt of the variance).
        :type std: float
        :param hard_clip_min: if not none, rvs will return max(result, hard_clip_min).
        :type hard_clip_min: float
        :param hard_clip_max: if not none, rvs will return min(result, hard_clip_min).
        :type hard_clip_max: float
        :param null_default_value: if none, null default value will be set to hard_clip_min.
        :type null_default_value: float
        """
        if null_default_value is None:
            ContinuousHyperparameterDistribution.__init__(self, hard_clip_min)
        else:
            ContinuousHyperparameterDistribution.__init__(self, null_default_value)

        self._mean = mean
        self._std = std
        self.hard_clip_min = hard_clip_min
        self.hard_clip_max = hard_clip_max

    def rvs(self) -> float:
        """
        Will return a float value in the specified range as specified at creation.

        :return: a float.
        """
        if self.hard_clip_min is None and self.hard_clip_max is None:
            result = float(np.random.normal(self._mean, self._std))
        else:
            a = -np.inf
            b = np.inf

            if self.hard_clip_min is not None:
                a = (self.hard_clip_min - self._mean) / self._std

            if self.hard_clip_max is not None:
                b = (self.hard_clip_max - self._mean) / self._std

            result = truncnorm.rvs(a=a, b=b, loc=self._mean, scale=self._std)

        if not math.isfinite(result):
            return self.rvs()
        return float(result)

    def pdf(self, x) -> float:
        """
        Calculate the Normal probability distribution value at position `x`.
        :param x: value where the probability distribution function is evaluated.
        :return: value of the probability distribution function.
        """

        if self.hard_clip_min is not None and (x < self.hard_clip_min):
            return 0.

        if self.hard_clip_max is not None and (x > self.hard_clip_max):
            return 0.

        if self.hard_clip_min is not None or self.hard_clip_max is not None:
            a = -np.inf
            b = np.inf

            if self.hard_clip_min is not None:
                a = (self.hard_clip_min - self._mean) / self._std

            if self.hard_clip_max is not None:
                b = (self.hard_clip_max - self._mean) / self._std

            return truncnorm.pdf(x, a=a, b=b, loc=self._mean, scale=self._std)

        return norm.pdf(x, loc=self._mean, scale=self._std)

    def cdf(self, x) -> float:
        """
        Calculate the Normal cumulative distribution value at position `x`.
        :param x: value where the cumulative distribution function is evaluated.
        :return: value of the cumulative distribution function.
        """
        if self.hard_clip_min is not None and (x < self.hard_clip_min):
            return 0.

        if self.hard_clip_max is not None and (x > self.hard_clip_max):
            return 1.

        if self.hard_clip_min is not None or self.hard_clip_max is not None:
            a = -np.inf
            b = np.inf

            if self.hard_clip_min is not None:
                a = (self.hard_clip_min - self._mean) / self._std

            if self.hard_clip_max is not None:
                b = (self.hard_clip_max - self._mean) / self._std

            return truncnorm.cdf(x, a=a, b=b, loc=self._mean, scale=self._std)

        return norm.cdf(x, loc=self._mean, scale=self._std)

    def min(self):
        """
        Calculate minimum value that can be sampled in the Normal distribution.

        :return: minimal value return from distribution.
        """
        return self.hard_clip_min if self.hard_clip_min is not None else -1 * np.inf

    def max(self):
        """
        Calculate minimum value that can be sampled in the Normal distribution.

        :return: minimal value return from distribution.
        """
        return self.hard_clip_max if self.hard_clip_max is not None else np.inf

    def mean(self):
        """
        Calculate mean value (also called esperance) of the random variable.

        :return: mean value of the random variable.
        """
        if self.hard_clip_min is None and self.hard_clip_max is None:
            return self._mean

        if self.hard_clip_min is None:
            alpha = -np.inf
        else:
            alpha = (self.hard_clip_min - self._mean) / self._std

        if self.hard_clip_max is None:
            beta = np.inf
        else:
            beta = (self.hard_clip_max - self._mean) / self._std

        Z = norm.cdf(beta) - norm.cdf(alpha)

        mean = self._mean + (norm.pdf(alpha) - norm.pdf(beta)) / Z * self._std
        return mean

    def std(self):
        """
        Calculate standard deviation value of the random variable.

        :return: standard deviation value of the random variable.
        """
        if self.hard_clip_min is None and self.hard_clip_max is None:
            return self._std

        return math.sqrt(self.var())

    def var(self):
        """
        Calculate variance value of the random variable.

        :return: variance value of the random variable.
        """
        if self.hard_clip_min is None and self.hard_clip_max is None:
            return self._std ** 2

        if self.hard_clip_min is None:
            alpha = -np.inf
        else:
            alpha = (self.hard_clip_min - self._mean) / self._std

        if self.hard_clip_max is None:
            beta = np.inf
        else:
            beta = (self.hard_clip_max - self._mean) / self._std

        Z = norm.cdf(beta) - norm.cdf(alpha)

        if self.hard_clip_max is None:
            # Case of one side truncated gaussian (lower tail truncated).
            variance = self._std ** 2 * (1 + alpha * norm.pdf(alpha) / Z - (norm.pdf(alpha) / Z) ** 2)
            return variance

        if self.hard_clip_min is None:
            # Case of one side truncated gaussian (upper tail truncated).
            variance = self._std ** 2 * (1 + - beta * norm.pdf(beta) / Z - (norm.pdf(beta) / Z) ** 2)
            return variance

        # Case of both side truncated gaussian.
        variance = self._std ** 2 * (1 + (alpha * norm.pdf(alpha) - beta * norm.pdf(beta)) / Z - (
            (norm.pdf(alpha) - norm.pdf(beta)) / Z) ** 2)
        return variance

    def __str__(self):
        return self.__class__.__name__ + f"(_mean={str(self._mean)}, _std={str(self._std)}" \
                                         f", hard_clip_min={self.hard_clip_min}, hard_clip_max={self.hard_clip_max})"


class LogNormal(LogSpaceDistributionMixin, ContinuousHyperparameterDistribution):
    """Get a LogNormal distribution."""

    @staticmethod
    def from_regular_mean_std(
        mean: float,
        std: float,
        hard_clip_min: float = None,
        hard_clip_max: float = None
    ):
        # See some theory here::
        # https://en.wikipedia.org/wiki/Log-normal_distribution#Generation_and_parameters
        # https://en.wikipedia.org/wiki/Log-normal_distribution#Arithmetic_moments

        # See these resources for more information on how to compute things:
        # https://www.wolframalpha.com/input?i=x%5E2%3De%5E%282*y%2B2*z%5E2%29-e%5E%282*y%2Bz%5E2%29
        # https://www.wolframalpha.com/input?i=sqrt%28log%281%2F2+e%5E%28-2+y%29+%28e%5E%282+y%29+%2B+sqrt%28e%5E%284+y%29+%2B+4+e%5E%282+y%29+x%5E2%29%29%29%29
        # https://www.wolframalpha.com/input?i=sqrt%28log%281%2F2+e%5E%28-2+y%29+%28e%5E%282+y%29+%2B+sqrt%28e%5E%284+y%29+%2B+4+e%5E%282+y%29+x%5E2%29%29%29%29%2C+y%3D1.6%2C+x%3D+180.601+
        # https://homepage.divms.uiowa.edu/~mbognar/applets/lognormal.html
        # https://stats.stackexchange.com/questions/568768/estimating-the-log-space-sigma-std-parameter-of-a-lognormal-distribution-from
        log2_space_mean = math.log2(mean)
        # log2_space_std = (math.log2(mean - std) + math.log2(mean - std)) / 2

        y = log2_space_mean
        x = std
        # The conversion to base 2 as follow was made just by replacing the log by log2 and the e by 2:
        # sqrt(log(1/2 e^(-2 y) (e^(2 y) + sqrt(e^(4 y) + 4 e^(2 y) x^2))))
        log2_space_std = math.sqrt(math.log2(
            0.5 * 2**(-2 * y) * (2**(2 * y) + math.sqrt(2**(4 * y) + 4 * 2**(2 * y) * x**2))
        ))

        return LogNormal(
            log2_space_mean=log2_space_mean,
            log2_space_std=log2_space_std,
            hard_clip_min=hard_clip_min,
            hard_clip_max=hard_clip_max
        )

    def __init__(self, log2_space_mean: float, log2_space_std: float,
                 hard_clip_min: float = None, hard_clip_max: float = None, null_default_value=None):
        """
        Create a LogNormal distribution.

        :param log2_space_mean: the most common value to pop, but before taking 2**value.
        :param log2_space_std: the standard deviation of the most common value to pop, but before taking 2**value.
        :param hard_clip_min: if not none, rvs will return max(result, hard_clip_min). This value is not checked in logspace (so it is checked after the exp).
        :param hard_clip_max: if not none, rvs will return min(result, hard_clip_min). This value is not checked in logspace (so it is checked after the exp).
        :param null_default_value: null default value for distribution. if None, take the hard_clip_min
        :type null_default_value: int
        """
        if null_default_value is None:
            ContinuousHyperparameterDistribution.__init__(self, hard_clip_min)
        else:
            ContinuousHyperparameterDistribution.__init__(self, null_default_value)
        self.log2_space_mean = log2_space_mean
        self.log2_space_std = log2_space_std
        if hard_clip_min is not None and hard_clip_min <= 0:
            hard_clip_min = None
        self.hard_clip_min = hard_clip_min
        self.hard_clip_max = hard_clip_max

    def rvs(self) -> float:
        """
        Will return a float value in the specified range as specified at creation.
        Note: the range at creation was in log space. The return value is after taking an exponent.

        :return: a float.
        """
        if self.hard_clip_min is None and self.hard_clip_max is None:
            result = 2 ** float(np.random.normal(self.log2_space_mean, self.log2_space_std))
        else:
            a = -np.inf
            b = np.inf

            if self.hard_clip_min is not None:
                a = (math.log2(self.hard_clip_min) - self.log2_space_mean) / self.log2_space_std

            if self.hard_clip_max is not None:
                b = (math.log2(self.hard_clip_max) - self.log2_space_mean) / self.log2_space_std

            result = 2 ** float(truncnorm.rvs(a=a, b=b, loc=self.log2_space_mean, scale=self.log2_space_std))

        if not math.isfinite(result):
            return self.rvs()

        return float(result)

    def pdf(self, x) -> float:
        """
        Calculate the LogNormal probability distribution value at position `x`.
        :param x: value where the probability distribution function is evaluated.
        :return: value of the probability distribution function.
        """

        if self.hard_clip_min is not None and (x < self.hard_clip_min):
            return 0.

        if self.hard_clip_max is not None and (x > self.hard_clip_max):
            return 0.

        if x <= 0:
            return 0.

        cdf_min = 0.
        cdf_max = 1.

        if self.hard_clip_min is not None:
            cdf_min = norm.cdf(math.log2(self.hard_clip_min), loc=self.log2_space_mean, scale=self.log2_space_std)

        if self.hard_clip_max is not None:
            cdf_max = norm.cdf(math.log2(self.hard_clip_max), loc=self.log2_space_mean, scale=self.log2_space_std)

        pdf_x = 1 / (x * math.log(2) * self.log2_space_std * math.sqrt(2 * math.pi)) * math.exp(
            -(math.log2(x) - self.log2_space_mean) ** 2 / (2 * self.log2_space_std ** 2))
        return pdf_x / (cdf_max - cdf_min)

    def cdf(self, x) -> float:
        """
        Calculate the LogNormal cumulative distribution value at position `x`.
        :param x: value where the cumulative distribution function is evaluated.
        :return: value of the cumulative distribution function.
        """
        if self.hard_clip_min is not None and (x < self.hard_clip_min):
            return 0.

        if self.hard_clip_max is not None and (x >= self.hard_clip_max):
            return 1.

        if x <= 0:
            return 0.

        cdf_min = 0.
        cdf_max = 1.

        if self.hard_clip_min is not None:
            cdf_min = norm.cdf(math.log2(self.hard_clip_min), loc=self.log2_space_mean, scale=self.log2_space_std)

        if self.hard_clip_max is not None:
            cdf_max = norm.cdf(math.log2(self.hard_clip_max), loc=self.log2_space_mean, scale=self.log2_space_std)

        cdf_x = norm.cdf(math.log2(x), loc=self.log2_space_mean, scale=self.log2_space_std)
        return (cdf_x - cdf_min) / (cdf_max - cdf_min)

    def min(self):
        """
        Calculate minimum value that can be sampled in the LogNormal distribution.

        :return: minimal value return from distribution.
        """
        return self.hard_clip_min if self.hard_clip_min is not None else 0.

    def max(self):
        """
        Calculate maximal value that can be sampled in the LogNormal distribution.

        :return: maximal value return from distribution.
        """
        return self.hard_clip_max if self.hard_clip_max is not None else np.inf

    def _pseudo_min(self) -> float:
        """
        This is like the function min, but in case it is of 0, there will be a clipping to at least 4 sigmas.
        """
        x = self.min()
        if math.isinf(x) or math.isnan(x):
            x = self.mean() - 4 * self.std()
        elif x == 0:
            x = self._icdf(0.0001, 0.0, self.mean())
        return x

    def mean(self):
        """
        Calculate mean value (also called esperance) of the random variable.

        :return: mean value of the random variable.
        """
        # More info: https://en.wikipedia.org/wiki/Log-normal_distribution#Generation_and_parameters
        # More info: https://en.wikipedia.org/wiki/Log-normal_distribution#Arithmetic_moments
        const = math.exp(math.log(2) * self.log2_space_mean + math.log(2) ** 2 * self.log2_space_std ** 2 / 2)

        if self.hard_clip_min is None and self.hard_clip_max is None:
            mean = const
            return mean

        if self.hard_clip_min == self.hard_clip_max:
            return self.hard_clip_min

        mean_to_calculate_cdf = self.log2_space_mean + math.log(2) * self.log2_space_std ** 2

        cdf_min = 0.
        log_normal_cdf_min = 0
        if self.hard_clip_min is not None:
            log_normal_cdf_min = norm.cdf(
                math.log2(self.hard_clip_min), loc=mean_to_calculate_cdf, scale=self.log2_space_std)
            cdf_min = norm.cdf(
                math.log2(self.hard_clip_min), loc=self.log2_space_mean, scale=self.log2_space_std)

        cdf_max = 1.
        log_normal_cdf_max = 1
        if self.hard_clip_max is not None:
            log_normal_cdf_max = norm.cdf(
                math.log2(self.hard_clip_max), loc=mean_to_calculate_cdf, scale=self.log2_space_std)
            cdf_max = norm.cdf(
                math.log2(self.hard_clip_max), loc=self.log2_space_mean, scale=self.log2_space_std)

        const_norm = const / (cdf_max - cdf_min)
        mean = const_norm * (log_normal_cdf_max - log_normal_cdf_min)
        return mean

    def var(self):
        """
        Calculate variance value (also called esperance) of the random variable.

        :return: variance value of the random variable.
        """
        # Use the mean**2 to calculate variance, with variance = moment_2 - mean**2.
        mean_squared = self.mean() ** 2
        const_moment_2 = math.exp(
            2 * math.log(2) * self.log2_space_mean + 2 * math.log(2) ** 2 * self.log2_space_std ** 2)
        if self.hard_clip_min is None and self.hard_clip_max is None:
            variance = const_moment_2 - mean_squared
            return variance

        if self.hard_clip_min == self.hard_clip_max:
            return 0.

        mean_to_calculate_cdf_for_variance = self.log2_space_mean + 2 * math.log(2) * self.log2_space_std ** 2

        if self.hard_clip_min is None:
            cdf_min = 0.
            log_normal_cdf_min = 0
        else:
            cdf_min = norm.cdf(
                math.log2(self.hard_clip_min), loc=self.log2_space_mean, scale=self.log2_space_std)
            log_normal_cdf_min = norm.cdf(
                math.log2(self.hard_clip_min), loc=mean_to_calculate_cdf_for_variance, scale=self.log2_space_std)

        if self.hard_clip_max is None:
            cdf_max = 1.
            log_normal_cdf_max = 1
        else:
            cdf_max = norm.cdf(
                math.log2(self.hard_clip_max), loc=self.log2_space_mean, scale=self.log2_space_std)
            log_normal_cdf_max = norm.cdf(
                math.log2(self.hard_clip_max), loc=mean_to_calculate_cdf_for_variance, scale=self.log2_space_std)

        const_norm = const_moment_2 / (cdf_max - cdf_min)
        moment_2 = const_norm * (log_normal_cdf_max - log_normal_cdf_min)
        variance = moment_2 - mean_squared
        return variance


class DistributionMixture(DiscreteHyperparameterDistribution):
    """Get a mixture of multiple distribution"""

    def __init__(self, distributions: List[HyperparameterDistribution],
                 distribution_amplitudes: List[float]):
        """
        Create a mixture of multiple distributions.

        Distribution amplitude are normalized to make sure that the sum equals one.
        This normalization ensure to keep a random variable at the end (0 < probability < 1).

        :param distributions: list of multiple instantiated distribution.
        :param distribution_amplitudes: list of float representing the amplitude in the probability distribution function for each distribution.
        """
        DiscreteHyperparameterDistribution.__init__(self, null_default_value=None)

        self.distributions: List[HyperparameterDistribution] = distributions

        # Normalize distribution amplitude
        distribution_amplitudes = np.array(distribution_amplitudes)
        distribution_amplitudes = distribution_amplitudes / np.sum(distribution_amplitudes)
        self.distribution_amplitudes: List[float] = distribution_amplitudes

    @staticmethod
    def build_gaussian_mixture(
            distribution_amplitudes: List[float],
            means: List[float],
            stds: List[float],
            distributions_mins: List[float],
            distributions_max: List[float],
            use_logs: bool = False,
            use_quantized_distributions: bool = False
    ) -> 'DistributionMixture':
        """
        Create a gaussian mixture.

         Will create a mixture distribution which consist of multiple gaussians of different amplitudes, means, standard deviations, mins and max.

        :param distribution_amplitudes: list of different amplitudes to infer to the mixture. The amplitudes are normalized to sum to 1.
        :param means: list of means to infer mean to each gaussian.
        :param stds: list of standard deviations to infer standard deviation to each gaussian.
        :param distributions_mins: list of minimum value that will clip each gaussian. If it is -Inf or None, it will not be clipped.
        :param distributions_max: list of maximal value that will clip each gaussian. If it is +Inf or None, it will not be clipped.
        :param distributions_max: bool weither to use a quantized version or not.
        :param use_logs: use logs boolean
        :param use_quantized_distributions: use quantized distributions boolean

        :return DistributionMixture instance
        """

        distribution_ctor: Callable[[float, float, float, float], HyperparameterDistribution] = \
            LogNormal.from_regular_mean_std if use_logs else Normal

        distributions = []

        for mean, std, single_min, single_max in zip(means, stds, distributions_mins, distributions_max):

            if single_min is None or np.isneginf(single_min):
                single_min = None

            if single_max is None or np.isposinf(single_max):
                single_max = None

            distribution_instance = distribution_ctor(
                mean=mean, std=std, hard_clip_min=single_min, hard_clip_max=single_max)

            if use_quantized_distributions:
                distribution_instance = Quantized(distribution_instance)

            distributions.append(distribution_instance)

        return DistributionMixture(distributions, distribution_amplitudes)

    def rvs(self) -> float:
        """
        Will return a float value drawned from the distribution mixture.

        :return: a float.
        """
        # Draw from which distribution to draw first using the amplitude as probabilities on how to draw from each.
        distribution_index = np.random.choice(len(self.distributions), p=self.distribution_amplitudes, replace=True)

        # Draw from distribution.
        return self.distributions[distribution_index].rvs()

    def pdf(self, x) -> float:
        """
        Calculate the mixture probability distribution value at position `x`.

        :param x: value where the probability distribution function is evaluated.
        :return: value of the probability distribution function.
        """
        pdf_result = 0

        for distribution_amplitude, distribution in zip(self.distribution_amplitudes, self.distributions):
            pdf_result += (distribution_amplitude * distribution.pdf(x))

        return pdf_result

    def cdf(self, x) -> float:
        """
        Calculate the mixture cumulative distribution value at position `x`.

        :param x: value where the cumulative distribution function is evaluated.
        :return: value of the cumulative distribution function.
        """
        cdf_result = 0

        for distribution_amplitude, distribution in zip(self.distribution_amplitudes, self.distributions):
            cdf_result += (distribution_amplitude * distribution.cdf(x))

        return cdf_result

    def mean(self) -> float:
        """
        Calculate mean of the mixture.

        Mean of the distribution mixture is calculated using the following equation:

        .. math:: mean = \sum_{i=1}^n w_i * \mu_i,

        where :math:`w_i` and :math:`\mu_i` are respectively the amplitude and mean of each distribution.

        :return: mean value of the mixture.
        """
        mean_result = 0

        for distribution_amplitude, distribution in zip(self.distribution_amplitudes, self.distributions):
            mean_result += (distribution_amplitude * distribution.mean())

        return mean_result

    def var(self) -> float:
        """
        Calculate variance of the mixture.

        Variance of a mixture is calculated using the following equation:

        .. math:: variance = (\sum_{i=1}^n w_i * \sigma_i * \mu_i) - \mu^2,

        where :math:`w_i`, :math:`\sigma_i` and :math:`\mu_i` are respectively the amplitude,
        standard deviation and mean of each distribution and :math:`\mu` is the mean of the whole mixture.

        :return: standard deviation value of the mixtture.
        """
        second_moment = 0

        for distribution_amplitude, distribution in zip(self.distribution_amplitudes, self.distributions):
            second_moment += distribution_amplitude * (distribution.var() + distribution.mean() ** 2)

        var_result = second_moment - self.mean() ** 2
        return var_result

    def std(self) -> float:
        """
        Calculate standard deviation of the mixture.

        :return: standard deviation value of the mixtture.
        """
        return math.sqrt(self.var())

    def min(self) -> float:
        """
        Calculate minimal domain value of the mixture.

        :return: minimal domain value of the mixtture.
        """
        return min([distribution.min() for distribution in self.distributions])

    def max(self) -> float:
        """
        Calculate minimal domain value of the mixture.

        :return: minimal domain value of the mixture.
        """
        return max([distribution.max() for distribution in self.distributions])

    def is_discrete(self) -> bool:
        """
        Check if the distribution is discrete.

        :return: True if the distribution is discrete, False otherwise.
        """
        return all([distribution.is_discrete() for distribution in self.distributions])

    def values(self) -> List[float]:
        """
        Get all the values of the distribution.

        :return: list of all the values of the distribution.
        """
        if self.is_discrete():
            return sum([distribution.values() for distribution in self.distributions], [])
        else:
            raise RuntimeError("The distribution is not discrete.")

    def probabilities(self) -> List[float]:
        """
        Get all the probabilities of the distribution.

        :return: list of all the probabilities of the distribution.
        """
        if self.is_discrete():
            return sum([distribution.probabilities() for distribution in self.distributions], [])


class LogDistributionMixture(DistributionMixture):

    @abstractmethod
    def rvs(self) -> float:
        # return super().rvs()
        raise NotImplementedError("TODO: must code this for the 'use logs' method.")

    @staticmethod
    def vals_to_log_wrap(vals: List[float]) -> List[float]:
        return


def _calculate_sum(
        eval_func,
        limits: List[float],
        value_step: float = 1.,
        tol: float = 1e-10,
        number_value_before_stop: int = 5
):
    method, starting_value, stop_value = _get_sum_starting_info(limits)

    current_x = starting_value
    number_negligible_result = 0
    total_sum = 0

    if method.lower() == "increasing":
        comparison = np.greater
    else:
        comparison = np.less

    while True:
        reach_end = comparison(current_x, stop_value)

        if reach_end:
            break

        current_result = eval_func(current_x)
        total_sum += current_result

        is_negligible = abs(current_result) < tol

        if is_negligible:
            number_negligible_result += 1
        else:
            number_negligible_result = 0

        if is_negligible and number_negligible_result > number_value_before_stop:
            break

        if method.lower() == "increasing":
            current_x += value_step
        else:
            current_x -= value_step

    return total_sum


def _get_sum_starting_info(limits):
    if np.isinf(limits[0]) and np.isinf(limits[1]):
        raise ValueError("Cannot calculate a sum on infinite terms.")
    if np.isposinf(limits[0]):
        starting_value = limits[1]
        stop_value = limits[0]
        method = "increasing"
    elif np.isposinf(limits[1]):
        starting_value = limits[0]
        stop_value = limits[1]
        method = "increasing"
    elif np.isneginf(limits[0]):
        starting_value = limits[1]
        stop_value = limits[0]
        method = "decreasing"
    elif np.isneginf(limits[1]):
        starting_value = limits[0]
        stop_value = limits[1]
        method = "decreasing"
    elif np.greater(limits[1], limits[0]):
        starting_value = limits[0]
        stop_value = limits[1]
        method = "increasing"
    else:
        starting_value = limits[1]
        stop_value = limits[0]
        method = "increasing"
    return method, starting_value, stop_value


def get_index_in_list_with_bool(choice_list: List[Any], value: Any) -> int:
    for choice_index, choice in enumerate(choice_list):
        if choice == value and not isinstance(choice, bool) and not isinstance(value, bool):
            index = choice_index
            return index

        if choice is value:
            index = choice_index
            return index

    raise ValueError("{} is not in list".format(value))
