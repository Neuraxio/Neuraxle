import math
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from neuraxle.hyperparams.distributions import (
    ContinuousHyperparameterDistribution, DiscreteHyperparameterDistribution,
    LogSpaceDistributionMixin)
from scipy.special import factorial
from scipy.stats import rv_continuous, rv_discrete, rv_histogram


def scipy_method(func):
    def wrapper(*args, **kwargs):
        self_in_args: ScipyDistributionWrapper = args[0]
        self_in_args._override_scipy_methods()
        return func(*args, **kwargs)

    return wrapper


class ScipyDistributionWrapper(ABC):
    """
    Base class for a distribution that wraps a scipy distribution.

    Usage example:

    .. code-block:: python

        distribution = ScipyDistributionWrapper(
            scipy_distribution=rv_histogram(histogram=histogram),
            null_default_value=null_default_value
        )

    .. seealso::
        :class:`HyperparameterDistribution`
    """

    def __init__(self, scipy_distribution, **scipy_distribution_arguments):
        self.scipy_distribution = scipy_distribution
        self.scipy_distribution_arguments = scipy_distribution_arguments

    def _override_scipy_methods(self):
        return

    @scipy_method
    def rvs(self, *args, **kwargs) -> float:
        return self.scipy_distribution.rvs(*args, **kwargs, **self.scipy_distribution_arguments)

    @scipy_method
    def rvs_many(self, size: int, *args, **kwargs) -> List:
        return self.scipy_distribution.rvs(size=size, *args, **kwargs, **self.scipy_distribution_arguments)

    @scipy_method
    def pdf(self, x, *args, **kwargs) -> float:
        if hasattr(self.scipy_distribution, 'pdf'):
            return self.scipy_distribution.pdf(x, *args, **kwargs, **self.scipy_distribution_arguments)
        else:
            return self.scipy_distribution.pmf(x, *args, **kwargs, **self.scipy_distribution_arguments)

    @scipy_method
    def cdf(self, x, *args, **kwargs) -> float:
        """
        Cumulative distribution function of the given x.

        :param x:
        :param args:
        :param kwargs:
        :return:
        """
        return self.scipy_distribution.cdf(x, *args, **kwargs, **self.scipy_distribution_arguments)

    @scipy_method
    def entropy(self, *args, **kwargs):
        """
        Differential entropy of the RV.

        :return:
        """
        return self.scipy_distribution.entropy(*args, **kwargs, **self.scipy_distribution_arguments)

    @scipy_method
    def expect(self, *args, **kwargs):
        """
        Calculate expected value of a function with respect to the distribution by numerical integration.

        :param args:
        :param kwargs:
        :return:
        """
        return self.scipy_distribution.expect(*args, **kwargs, **self.scipy_distribution_arguments)

    @scipy_method
    def fit(self, data, *args, **kwargs):
        """
        Return MLEs for shape( if applicable), location, and scale parameters from data.

        :param data:
        :param args:
        :param kwargs:
        :return:
        """
        return self.scipy_distribution.fit(data, *args, **kwargs, **self.scipy_distribution_arguments)

    @scipy_method
    def fit_loc_scale(self, data, *args):
        """
        Estimate loc and scale parameters from data using 1st and 2nd moments.
        """
        return self.scipy_distribution.fit_loc_scale(data, *args, **self.scipy_distribution_arguments)

    @scipy_method
    def freeze(self, *args, **kwargs):
        """
        Freeze the distribution for the given arguments.

        :return:
        """
        return self.scipy_distribution.freeze(*args, **kwargs, **self.scipy_distribution_arguments)

    @scipy_method
    def interval(self, alpha, *args, **kwargs):
        """
        Confidence interval with equal areas around the median.

        :param alpha:
        :return:
        """
        return self.scipy_distribution.interval(alpha, *args, **kwargs, **self.scipy_distribution_arguments)

    @scipy_method
    def isf(self, q, *args, **kwargs):
        """
        Inverse survival function(inverse of sf) at q of the given RV.

        :param q:
        :return:
        """
        return self.scipy_distribution.isf(q, *args, **kwargs, **self.scipy_distribution_arguments)

    @scipy_method
    def logcdf(self, x, *args, **kwargs):
        """
        Log of the cumulative distribution function at x of the given RV.

        :param x:
        :param args:
        :param kwargs:
        :return:
        """
        return self.scipy_distribution.logcdf(x, *args, **kwargs, **self.scipy_distribution_arguments)

    @scipy_method
    def logpdf(self, x, *args, **kwargs):
        """
        Log of the probability density function at x of the given RV.

        :param x:
        :param args:
        :param kwargs:
        :return:
        """
        return self.scipy_distribution.logpdf(x, *args, **kwargs, **self.scipy_distribution_arguments)

    @scipy_method
    def logsf(self, x, *args, **kwargs):
        """
        Log of the survival function of the given RV.

        :param x:
        :return:
        """
        return self.scipy_distribution.logsf(x, *args, **kwargs, **self.scipy_distribution_arguments)

    @scipy_method
    def min(self):
        return self.scipy_distribution.a

    @scipy_method
    def max(self):
        return self.scipy_distribution.b

    @scipy_method
    def mean(self, *args, **kwargs):
        """
        Mean of the distribution.

        :return:
        """
        return self.scipy_distribution.mean(*args, **kwargs)

    @scipy_method
    def median(self, *args, **kwargs):
        """
        Median of the distribution.

        :return:
        """
        return self.scipy_distribution.median(*args, **kwargs, **self.scipy_distribution_arguments)

    @scipy_method
    def moment(self, n, *args, **kwargs):
        """
        n-th order non-central moment of distribution.

        :param n:
        :return:
        """
        return self.scipy_distribution.moment(n, *args, **kwargs, **self.scipy_distribution_arguments)

    @scipy_method
    def nnlf(self, theta, x):
        """
        Return negative loglikelihood function.

        :param theta:
        :param x:
        :return:
        """
        return self.scipy_distribution.nnlf(theta, x)

    @scipy_method
    def ppf(self, q, *args, **kwargs):
        """
        Percent point function(inverse of cdf) at q of the given RV.

        :param q:
        :return:
        """
        return self.scipy_distribution.ppf(q, *args, **kwargs)

    @scipy_method
    def sf(self, x, *args, **kwargs):
        """
        Survival function(1 - cdf) at x of the given RV.

        :param x:
        :return:
        """
        return self.scipy_distribution.sf(x, *args, **kwargs, **self.scipy_distribution_arguments)

    @scipy_method
    def stats(self, *args, **kwargs):
        """
        Some statistics of the given RV.

        :return:
        """
        return self.scipy_distribution.stats(*args, **kwargs, **self.scipy_distribution_arguments)

    @scipy_method
    def std(self, *args, **kwargs):
        """
        Standard deviation of the distribution.

        :param args:
        :param kwargs:
        :return:
        """
        return self.scipy_distribution.std(*args, **kwargs, **self.scipy_distribution_arguments)

    @scipy_method
    def support(self, *args, **kwargs):
        """
        Return the support of the distribution.

        :param args:
        :param kwargs:
        :return:
        """
        return self.scipy_distribution.support(*args, **kwargs, **self.scipy_distribution_arguments)

    @scipy_method
    def var(self, *args, **kwargs):
        """
        Variance of the distribution.

        :return:
        """
        return self.scipy_distribution.var(*args, **kwargs, **self.scipy_distribution_arguments)

    def to_sk_learn(self):
        return self.scipy_distribution


class ScipyContinuousDistributionWrapper(ScipyDistributionWrapper, ContinuousHyperparameterDistribution):
    def __init__(self, scipy_distribution, null_default_value=None, **kwargs):
        ScipyDistributionWrapper.__init__(
            self,
            scipy_distribution=scipy_distribution,
            **kwargs
        )
        ContinuousHyperparameterDistribution.__init__(self, null_default_value=null_default_value)


class ScipyDiscreteDistributionWrapper(ScipyDistributionWrapper, DiscreteHyperparameterDistribution):
    def __init__(self, scipy_distribution, null_default_value=None, **kwargs):
        ScipyDistributionWrapper.__init__(
            self,
            scipy_distribution=scipy_distribution,
            **kwargs
        )
        DiscreteHyperparameterDistribution.__init__(self, null_default_value=null_default_value)

    def values(self):
        if not hasattr(self, "precomp"):
            self.precomp = list(sorted(list(set(
                self.rvs() for _ in range(5000)
            ))))
        return self.precomp


class BaseCustomDiscreteScipyDistribution(ScipyDiscreteDistributionWrapper):
    def __init__(self, name, min_included, max_included, null_default_value, **kwargs):
        scipy_dist_obj = rv_discrete(
            name=name,
            a=min_included,
            b=max_included,
            **kwargs
        )
        scipy_dist_obj._pmf = self._pmf
        ScipyDiscreteDistributionWrapper.__init__(
            self,
            scipy_distribution=scipy_dist_obj,
            null_default_value=null_default_value
        )

    def _override_scipy_methods(self):
        self.scipy_distribution._pmf = self._pmf

    @abstractmethod
    def _pmf(self, x, *args):
        pass


class Distribution(rv_continuous):
    def _pdf(self, x):
        return 0.0


class BaseCustomContinuousScipyDistribution(ScipyContinuousDistributionWrapper):
    def __init__(self, name, min_included, max_included, null_default_value, **kwargs):
        scipy_distribution = Distribution(
            name=name,
            a=min_included,
            b=max_included,
            **kwargs
        )
        scipy_distribution._pdf = self._pdf

        ScipyContinuousDistributionWrapper.__init__(
            self,
            scipy_distribution=scipy_distribution,
            null_default_value=null_default_value
        )

    def _override_scipy_methods(self):
        self.scipy_distribution._pdf = self._pdf

    @abstractmethod
    def _pdf(self, x, *args):
        pass


class ScipyLogUniform(LogSpaceDistributionMixin, BaseCustomContinuousScipyDistribution):
    """
    Get a LogUniform distribution.

    Refer to: :class:`scipy.stats.loguniform`.

    .. seealso::
        :func:`~neuraxle.base.BaseStep.set_hyperparams_space`,
        :class:`ScipyDistributionWrapper`,
    """

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
            null_default_value = math.log2(min_included)
        else:
            null_default_value = math.log2(null_default_value)

        self.min_included: float = min_included
        self.max_included: float = max_included
        self.log2_min_included = math.log2(min_included)
        self.log2_max_included = math.log2(max_included)

        super().__init__(
            name='log_uniform',
            null_default_value=null_default_value,
            min_included=self.log2_min_included,
            max_included=self.log2_max_included
        )

    def _pdf(self, x) -> float:
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


class StdMeanLogNormal(LogSpaceDistributionMixin, BaseCustomContinuousScipyDistribution):
    """
    Get a normal distribution from std and min.

    .. seealso::
        :class:`NormalScipyDistribution`,
        :func:`~neuraxle.base.BaseStep.set_hyperparams_space`,
        :class:`HyperparameterDistribution`,
        :class:`ScipyDistributionWrapper`,
        :class:`neuraxle.hyperparams.space.HyperparameterSamples`,
        :class:`neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`neuraxle.base.BaseStep`
    """

    def __init__(
            self,
            log2_space_mean: float,
            log2_space_std: float,
            hard_clip_min: float,
            hard_clip_max: float,
            null_default_value: float = None
    ):
        """
        Create a normal distribution from mean and standard deviation.

        :param log2_space_mean: the most common value to pop
        :type log2_space_mean: float
        :param log2_space_std: the standard deviation (that is, the sqrt of the variance).
        :type log2_space_std: float
        :param hard_clip_min: if not none, rvs will return max(result, hard_clip_min).
        :type hard_clip_min: float
        :param hard_clip_max: if not none, rvs will return min(result, hard_clip_min).
        :type hard_clip_max: float
        :param null_default_value: if none, null default value will be set to hard_clip_min.
        :type null_default_value: float
        """
        if null_default_value is None:
            null_default_value = hard_clip_min

        if hard_clip_min is None:
            hard_clip_min = np.nan

        if hard_clip_max is None:
            hard_clip_max = np.nan

        self.log2_space_mean = log2_space_mean
        self.log2_space_std = log2_space_std

        super().__init__(
            name='log_normal',
            min_included=hard_clip_min,
            max_included=hard_clip_max,
            null_default_value=null_default_value
        )

    def _pdf(self, x):
        if x <= 0:
            return 0.

        cdf_min = 0.
        cdf_max = 1.

        pdf_x = 1 / (x * math.log(2) * self.log2_space_std * math.sqrt(2 * math.pi)) * math.exp(
            -(math.log2(x) - self.log2_space_mean) ** 2 / (2 * self.log2_space_std ** 2))
        return pdf_x / (cdf_max - cdf_min)


class Gaussian(BaseCustomContinuousScipyDistribution):
    """
    Gaussian distribution that inherits from `scipy.stats.rv_continuous <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous>`_

    Example usage :

    .. code-block:: python

        gaussian_distribution = Gaussian(
            min_included=0,
            max_included=10,
            null_default_value=0.0
        )

        assert 0.0 <= gaussian_distribution.rvs() <= 10.0
        assert gaussian_distribution.pdf(10) < 0.001
        assert gaussian_distribution.pdf(0) < 0.42
        assert 0.55 > gaussian_distribution.cdf(5.0) > 0.45
        assert gaussian_distribution.cdf(0) == 0.0


    .. seealso::
        :class:`GaussianScipyDistribution`,
        :class:`ScipyDistributionWrapper`,
        :func:`~neuraxle.base.BaseStep.set_hyperparams_space`,
        :class:`HyperparameterDistribution`,
        :class:`neuraxle.hyperparams.space.HyperparameterSamples`,
        :class:`neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`neuraxle.base.BaseStep`
    """

    def __init__(self, min_included: int, max_included: int, null_default_value: float = None):
        self.max_included = max_included
        self.min_included = min_included

        BaseCustomContinuousScipyDistribution.__init__(
            self,
            name='gaussian',
            min_included=min_included,
            max_included=max_included,
            null_default_value=null_default_value
        )

    def _pdf(self, x):
        return math.exp(-x ** 2 / 2.) / np.sqrt(2.0 * np.pi)


class Poisson(BaseCustomDiscreteScipyDistribution):
    """
    Poisson distribution that inherits from `scipy.stats.rv_discrete <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_histogram.html#scipy.stats.rv_histogram>`_

    Example usage :

    .. code-block:: python

        poisson_distribution = Poisson(
            min_included=0.0,
            max_included=10.0,
            null_default_value=0.0,
            mu=5.0
        )

        rvs = [poisson_distribution.rvs() for i in range(10)]
        assert not all(x == rvs[0] for x in rvs)
        assert 0.0 <= poisson_distribution.rvs() <= 10.0
        assert poisson_distribution.pdf(10) == 0.01813278870782187
        assert np.isclose(poisson_distribution.pdf(0), 0.006737946999085467)
        assert poisson_distribution.cdf(5.0) == 0.6159606548330632
        assert poisson_distribution.cdf(0) == 0.006737946999085467


    .. seealso::
        :func:`~neuraxle.base.BaseStep.set_hyperparams_space`,
        :class:`PoissonScipyDistribution`,
        :class:`HyperparameterDistribution`,
        :class:`ScipyDistributionWrapper`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`~neuraxle.base.BaseStep`
    """

    def __init__(self, min_included: float, max_included: float, null_default_value: float = None, mu=0.6):
        super().__init__(
            min_included=min_included,
            max_included=max_included,
            name='poisson',
            null_default_value=null_default_value
        )
        self.mu = mu

    def _pmf(self, x):
        return math.exp(-self.mu) * self.mu ** x / factorial(x)


class Histogram(ScipyDiscreteDistributionWrapper):
    """
    Histogram distribution that inherits from `scipy.stats.rv_histogram <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_histogram.html#scipy.stats.rv_histogram>`_

    Example usage :

    .. code-block:: python

        hist_dist = Histogram(
             histogram=np.histogram(data, bins=100),
             null_default_value=0.0
        )

        assert min(data) <= hist_dist.rvs() <= max(data)
        assert 1.0 > hist_dist.pdf(x=1.0) > 0.0
        assert hist_dist.pdf(x=np.max(data)) == 0.0
        assert hist_dist.pdf(x=np.min(data)) < 0.001
        assert hist_dist.cdf(x=np.max(data)) == 1.0
        assert 0.55 > hist_dist.cdf(x=np.median(data)) > 0.45
        assert hist_dist.cdf(x=np.min(data)) == 0.0


    .. seealso::
        :class:`HyperparameterDistribution`,
        :class:`ScipyDistributionWrapper`,
        :class:`Poisson`,
        :class:`Gaussian`,
        :func:`~neuraxle.base.BaseStep.set_hyperparams_space`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`~neuraxle.base.BaseStep`
    """

    def __init__(self, histogram: np.histogram, null_default_value: float = None, **kwargs):
        scipy_distribution = rv_histogram(histogram=histogram, **kwargs)
        ScipyDiscreteDistributionWrapper.__init__(
            self,
            scipy_distribution=scipy_distribution,
            null_default_value=null_default_value
        )
