import math
from abc import abstractmethod, ABC
from typing import Optional, List, Any

import numpy as np
from scipy.integrate import quad
from scipy.special import factorial
from scipy.stats import rv_continuous, norm, rv_discrete, rv_histogram, truncnorm, randint

from neuraxle.hyperparams.distributions import HyperparameterDistribution, WrappedHyperparameterDistributions, \
    DiscreteHyperparameterDistribution, ContinuousHyperparameterDistrbution


def scipy_method(func):
    def wrapper(*args, **kwargs):
        self_in_args = args[0]
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
        if kwargs is None:
            kwargs = dict()
        kwargs = dict(kwargs)
        kwargs.update(**self.scipy_distribution_arguments)
        return self.scipy_distribution.rvs(*args, **kwargs)

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
        Estimate loc and scale parameters from data using 1 st and 2 nd moments. 
        
        :param data: 
        :return: 
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


class ScipyContinuousDistributionWrapper(ScipyDistributionWrapper, HyperparameterDistribution):
    def __init__(self, scipy_distribution, null_default_value=None, **kwargs):
        ScipyDistributionWrapper.__init__(
            self,
            scipy_distribution=scipy_distribution,
            **kwargs
        )
        ContinuousHyperparameterDistrbution.__init__(self, null_default_value=null_default_value)


class ScipyDiscreteDistributionWrapper(ScipyDistributionWrapper, DiscreteHyperparameterDistribution):
    def __init__(self, scipy_distribution, null_default_value=None, **kwargs):
        ScipyDistributionWrapper.__init__(
            self,
            scipy_distribution=scipy_distribution,
            **kwargs
        )
        DiscreteHyperparameterDistribution.__init__(self, null_default_value=null_default_value)


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


class RandInt(ScipyDiscreteDistributionWrapper):
    """
    Rand int scipy distribution. Check out `scipy.stats.randint for more info <https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.randint.html>`_
    """

    def __init__(self, min_included: int, max_included: int, **kwargs):
        self.min_included = min_included
        self.max_included = max_included
        kwargs.update(low=min_included, high=max_included)

        super().__init__(
            scipy_distribution=randint,
            **kwargs,
        )

    def min(self):
        return self.min_included

    def mean(self, *args, **kwargs):
        return (self.min_included + self.max_included) / 2

    def median(self, *args, **kwargs):
        return self.mean()

    def max(self):
        return self.max_included


class Uniform(BaseCustomContinuousScipyDistribution):
    """
    Get a uniform distribution.

    .. seealso::
        :class:`UniformScipyDistribution`,
        :func:`~neuraxle.base.BaseStep.set_hyperparams_space`,
        :class:`HyperparameterDistribution`,
        :class:`ScipyDistributionWrapper`,
        :class:`neuraxle.hyperparams.space.HyperparameterSamples`,
        :class:`neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`neuraxle.base.BaseStep`
    """

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
            null_default_value = min_included

        self.min_included: float = min_included
        self.max_included: float = max_included

        super().__init__(
            name='uniform',
            min_included=min_included,
            max_included=max_included,
            null_default_value=null_default_value
        )

    def _pdf(self, x):
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


class LogUniform(BaseCustomContinuousScipyDistribution):
    """
    Get a LogUniform distribution.

    .. seealso::
        :class:`LogUniformScipyDistribution`,
        :func:`~neuraxle.base.BaseStep.set_hyperparams_space`,
        :class:`HyperparameterDistribution`,
        :class:`ScipyDistributionWrapper`,
        :class:`neuraxle.hyperparams.space.HyperparameterSamples`,
        :class:`neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`neuraxle.base.BaseStep`
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


class Normal(BaseCustomContinuousScipyDistribution):
    """
    LogNormal distribution that wraps a `continuous scipy distribution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous>`_

    Example usage :

    .. code-block:: python

        distribution = LogNormal(
            min_included=0,
            max_included=10,
            null_default_value=0.0
        )

        assert 0.0 <= distribution.rvs() <= 10.0
        assert distribution.pdf(10) < 0.001
        assert distribution.pdf(0) < 0.42
        assert 0.55 > distribution.cdf(5.0) > 0.45
        assert distribution.cdf(0) == 0.0


    .. seealso::
        :class:`LogNormalScipyDistribution`,
        :func:`~neuraxle.base.BaseStep.set_hyperparams_space`,
        :class:`HyperparameterDistribution`,
        :class:`ScipyDistributionWrapper`,
        :class:`neuraxle.hyperparams.space.HyperparameterSamples`,
        :class:`neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`neuraxle.base.BaseStep`
    """

    def __init__(
            self,
            mean: float,
            std: float,
            hard_clip_min: float = None,
            hard_clip_max: float = None,
            null_default_value: float = None
    ):
        super().__init__(
            name='normal',
            min_included=hard_clip_min,
            max_included=hard_clip_max,
            null_default_value=null_default_value
        )

        self.hard_clip_min = hard_clip_min
        self.hard_clip_max = hard_clip_max
        self.mean = mean
        self.std = std

    def _pdf(self, x) -> float:
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
                a = (self.hard_clip_min - self.mean) / self.std

            if self.hard_clip_max is not None:
                b = (self.hard_clip_max - self.mean) / self.std

            return truncnorm.pdf(x, a=a, b=b, loc=self.mean, scale=self.std)

        return norm.pdf(x, loc=self.mean, scale=self.std)


class LogNormal(BaseCustomContinuousScipyDistribution):
    """
    Get a normal distribution.

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


class Histogram(ScipyDistributionWrapper):
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
        ScipyDistributionWrapper.__init__(
            self,
            scipy_distribution=rv_histogram(histogram=histogram, **kwargs)
        )
        ContinuousHyperparameterDistrbution.__init__(self, null_default_value=null_default_value)


class FixedHyperparameter(BaseCustomDiscreteScipyDistribution):
    """This is an hyperparameter that won't change again, but that is still expressed as a distribution."""

    def __init__(self, value, null_default_value=None):
        """
        Create a still hyperparameter

        :param value: what will be returned by calling ``.rvs()``.
        """
        self.value = value

        super().__init__(
            name='fixed',
            min_included=value,
            max_included=value,
            null_default_value=null_default_value
        )

    def _pdf(self, x):
        if x == self.value:
            return 1.
        return 0.


class Boolean(BaseCustomDiscreteScipyDistribution):
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
        if proba_is_true is None:
            proba_is_true = 0.5

        if not (0 <= proba_is_true <= 1):
            raise ValueError("Probability must be between 0 and 1 (inclusively).")

        self.proba_is_true = proba_is_true

        super().__init__(
            name='boolean',
            min_included=0,
            max_included=1,
            null_default_value=null_default_value
        )

    def _pmf(self, x):
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


def get_index_in_list_with_bool(choice_list: List[Any], value: Any) -> int:
    for choice_index, choice in enumerate(choice_list):
        if choice == value and not isinstance(choice, bool) and not isinstance(value, bool):
            index = choice_index
            return index

        if choice is value:
            index = choice_index
            return index

    raise ValueError("{} is not in list".format(value))


class Choice(BaseCustomDiscreteScipyDistribution):
    """Get a random value from a choice list of possible value for this hyperparameter.

    When narrowed, the choice will only collapse to a single element when narrowed enough.
    For example, if there are 4 items in the list, only at a narrowing value of 0.25 that
    the first item will be kept alone.
    """

    def __init__(self, choice_list: List, probas: Optional[List[float]] = None, null_default_value=None):
        """
        Create a random choice hyperparameter from the given list.

        :param choice_list: a list of values to sample from.
        :type choice_list: List
        :param null_default_value: default value for distribution
        :type null_default_value: default choice value. if None, default choice value will be the first choice
        """
        self.choice_list = choice_list

        if probas is None:
            probas = [1 / len(self.choice_list) for _ in self.choice_list]

        # Normalize probas juste in case sum is not equal to one.
        self.probas = np.array(probas) / np.sum(probas)

        if null_default_value is None:
            null_default_value = choice_list[0]
        elif null_default_value in choice_list:
            null_default_value = null_default_value
        else:
            raise ValueError(
                'invalid default value {0} not in choice list : {1}'.format(null_default_value, choice_list))

        super().__init__(
            name='choice',
            min_included=0,
            max_included=len(choice_list) - 1,
            null_default_value=null_default_value
        )

    def probabilities(self):
        return self.probas

    def values(self):
        return self.choice_list

    def rvs(self, *args, **kwargs):
        sample_choice_index = super().rvs(*args, **kwargs)
        if isinstance(sample_choice_index, np.ndarray):
            return [self.choice_list[int(i)] for i in sample_choice_index]
        return self.choice_list[int(sample_choice_index)]

    def pdf(self, x, *args, **kwargs):
        choice_index = [str(choice) for choice in self.choice_list].index(str(x))
        return self.scipy_distribution.pmf(choice_index, *args, **kwargs, **self.scipy_distribution_arguments)

    def cdf(self, x, *args, **kwargs):
        try:
            index = [str(choice) for choice in self.choice_list].index(str(x))
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

    def min(self):
        return 0

    def max(self):
        return len(self.choice_list) - 1

    def std(self):
        return np.std([i for i, _ in enumerate(self.choice_list)])

    def pmf(self, x):
        """
        Calculate the choice probability mass function value at position `x`.

        :param x: value where the probability mass function is evaluated.
        :return: value of the probability mass function.
        """
        if len(self.choice_list) - 1 >= x[-1] >= 0:
            return sum([self.probas[int(i)] for i in x])
        else:
            return 0.

    def _pmf(self, x):
        """
        Calculate the choice probability mass function value at position `x`.

        :param x: value where the probability mass function is evaluated.
        :return: value of the probability mass function.
        """
        if len(self.choice_list) - 1 >= x[-1] >= 0:
            return sum([self.probas[int(i)] for i in x])
        else:
            return 0.

    def __len__(self):
        return len(self.choice_list)


class Quantized(WrappedHyperparameterDistributions, BaseCustomContinuousScipyDistribution):
    """A quantized wrapper for another distribution: will round() the rvs number."""

    def __init__(self, hd: HyperparameterDistribution = None, hds: List[HyperparameterDistribution] = None,
                 null_default_value=None):
        WrappedHyperparameterDistributions.__init__(
            self,
            hd=hd,
            hds=hds,
            null_default_value=null_default_value
        )

        hd_min = self.hd.min()
        if np.isneginf(hd_min):
            min_included = hd_min
        else:
            min_included = round(hd_min)

        hd_max = self.hd.max()
        if np.isposinf(hd_max):
            max_included = hd_max
        else:
            max_included = round(hd_max)

        BaseCustomContinuousScipyDistribution.__init__(
            self,
            name='quantized',
            min_included=min_included,
            max_included=max_included,
            null_default_value=self.null_default_value,
            is_continuous=True
        )

    def _pdf(self, x):
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
