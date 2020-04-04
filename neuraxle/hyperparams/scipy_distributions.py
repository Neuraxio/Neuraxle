import math

import numpy as np
from scipy.special import factorial
from scipy.stats import rv_continuous, norm, rv_discrete, rv_histogram, truncnorm, randint

from neuraxle.hyperparams.distributions import HyperparameterDistribution


class BaseWrappedScipyDistribution(HyperparameterDistribution):
    """
    Base class for a distribution that wraps a scipy distribution.

    Usage example:

    .. code-block:: python

        distribution = BaseWrappedScipyDistribution(
            scipy_distribution=rv_histogram(histogram=histogram),
            null_default_value=null_default_value
        )

    .. seealso::
        :class:`HyperparameterDistribution`
    """

    def __init__(self, scipy_distribution, null_default_value, **kwargs):
        self.kwargs = kwargs
        self.sk_learn_distribution = scipy_distribution
        HyperparameterDistribution.__init__(self, null_default_value=null_default_value)

    def rvs(self, *args, **kwargs) -> float:
        return self.sk_learn_distribution.rvs(*args, **self.kwargs, **kwargs)

    def pdf(self, x, *args, **kwargs) -> float:
        if hasattr(self.sk_learn_distribution, 'pdf'):
            return self.sk_learn_distribution.pdf(x, *args, **self.kwargs, **kwargs)
        else:
            return self.sk_learn_distribution.pmf(x, *args, **self.kwargs, **kwargs)

    def cdf(self, x, *args, **kwargs) -> float:
        return self.sk_learn_distribution.cdf(x, *args, **self.kwargs, **kwargs)

    def to_sk_learn(self):
        return self.sk_learn_distribution


class RandInt(BaseWrappedScipyDistribution):
    def __init__(self, min_included: int, max_included: int, null_default_value: float = None):
        BaseWrappedScipyDistribution.__init__(
            self,
            scipy_distribution=randint(low=min_included, high=max_included),
            null_default_value=null_default_value
        )


class UniformScipyDistribution(rv_continuous):
    def _pdf(self, x, min_included, max_included):
        """
        Calculate the Uniform probability distribution value at position `x`.

        :param x: value where the probability distribution function is evaluated.
        :return: value of the probability distribution function.
        """

        if min_included == max_included and (x == min_included):
            return 1.

        if (x >= min_included) and (x <= max_included):
            return 1 / (max_included - min_included)

        # Manage case where it is outside the probability distribution function.
        return 0.


class Uniform(BaseWrappedScipyDistribution):
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
            null_default_value = min_included

        self.min_included: float = min_included
        self.max_included: float = max_included

        BaseWrappedScipyDistribution.__init__(
            self,
            scipy_distribution=UniformScipyDistribution(
                name='uniform',
                a=min_included,
                b=max_included
            ),
            null_default_value=null_default_value,
            min_included=min_included,
            max_included=max_included
        )


class LogUniformScipyDistribution(rv_continuous):
    def _pdf(self, x, log2_min_included, log2_max_included) -> float:
        """
        Calculate the logUniform probability distribution value at position `x`.
        :param x: value where the probability distribution function is evaluated.
        :return: value of the probability distribution function.
        """
        if (log2_min_included == log2_max_included) and x == 2 ** log2_min_included:
            return 1.

        if (x >= 2 ** log2_min_included) and (x <= 2 ** log2_max_included):
            return 1 / (x * math.log(2) * (log2_max_included - log2_min_included))

        return 0.


class LogUniform(BaseWrappedScipyDistribution):
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

        BaseWrappedScipyDistribution.__init__(
            self,
            scipy_distribution=NormalScipyDistribution(
                name='log_uniform',
                a=self.min_included,
                b=self.max_included
            ),
            null_default_value=null_default_value,
            min_included=self.min_included,
            max_included=self.max_included,
            log2_min_included=self.log2_min_included,
            log2_max_included=self.log2_max_included,
        )


class NormalScipyDistribution(rv_continuous):
    def _pdf(self, x, hard_clip_min, hard_clip_max, mean, std) -> float:
        """
        Calculate the Normal probability distribution value at position `x`.
        :param x: value where the probability distribution function is evaluated.
        :return: value of the probability distribution function.
        """

        if hard_clip_min is not None and (x < hard_clip_min):
            return 0.

        if hard_clip_max is not None and (x > hard_clip_max):
            return 0.

        if hard_clip_min is not None or hard_clip_max is not None:
            a = -np.inf
            b = np.inf

            if hard_clip_min is not None:
                a = (hard_clip_min - mean) / std

            if hard_clip_max is not None:
                b = (hard_clip_max - mean) / std

            return truncnorm.pdf(x, a=a, b=b, loc=mean, scale=std)

        return norm.pdf(x, loc=mean, scale=std)


class Normal(BaseWrappedScipyDistribution):
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
            null_default_value = hard_clip_min

        BaseWrappedScipyDistribution.__init__(
            self,
            scipy_distribution=LogNormalScipyDistribution(
                name='log_normal',
                a=hard_clip_min,
                b=hard_clip_max
            ),
            null_default_value=null_default_value,
            hard_clip_min=hard_clip_min,
            hard_clip_max=hard_clip_max,
            mean=mean,
            std=std
        )


class LogNormalScipyDistribution(rv_continuous):
    def _pdf(self, x, hard_clip_min, hard_clip_max, log2_space_mean, log2_space_std, *args):
        if hard_clip_min is not None and (x < hard_clip_min):
            return 0.

        if hard_clip_max is not None and (x > hard_clip_max):
            return 0.

        if x <= 0:
            return 0.

        cdf_min = 0.
        cdf_max = 1.

        if hard_clip_min is not None:
            cdf_min = norm.cdf(math.log2(hard_clip_min), loc=log2_space_mean, scale=log2_space_std)

        if hard_clip_max is not None:
            cdf_max = norm.cdf(math.log2(hard_clip_max), loc=log2_space_mean, scale=log2_space_std)

        pdf_x = 1 / (x * math.log(2) * log2_space_std * math.sqrt(2 * math.pi)) * math.exp(
            -(math.log2(x) - log2_space_mean) ** 2 / (2 * log2_space_std ** 2))
        return pdf_x / (cdf_max - cdf_min)


class LogNormal(BaseWrappedScipyDistribution):
    """
    LogNormal distribution that wraps a `continuous scipy distribution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous>`_

    Example usage :

    .. code-block:: python

        gaussian_distribution = GaussianContinuous(
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
        :func:`~neuraxle.base.BaseStep.set_hyperparams_space`,
        :class:`Discrete`,
        :class:`Continuous`,
        :class:`HyperparameterDistribution`,
        :class:`neuraxle.hyperparams.space.HyperparameterSamples`,
        :class:`neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`neuraxle.base.BaseStep`
    """

    def __init__(self, min_included: int, max_included: int, null_default_value: float = None):
        BaseWrappedScipyDistribution.__init__(
            self,
            scipy_distribution=LogNormalScipyDistribution(
                name='log_normal',
                a=min_included,
                b=max_included
            ),
            null_default_value=null_default_value
        )


class GaussianScipyDistribution(rv_continuous):
    def _pdf(self, x):
        return math.exp(-x ** 2 / 2.) / np.sqrt(2.0 * np.pi)


class Gaussian(BaseWrappedScipyDistribution):
    """
    Gaussian distribution that inherits from `scipy.stats.rv_continuous <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous>`_

    Example usage :

    .. code-block:: python

        gaussian_distribution = GaussianContinuous(
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
        :func:`~neuraxle.base.BaseStep.set_hyperparams_space`,
        :class:`Discrete`,
        :class:`Continuous`,
        :class:`HyperparameterDistribution`,
        :class:`neuraxle.hyperparams.space.HyperparameterSamples`,
        :class:`neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`neuraxle.base.BaseStep`
    """

    def __init__(self, min_included: int, max_included: int, null_default_value: float = None):
        BaseWrappedScipyDistribution.__init__(
            self,
            scipy_distribution=GaussianScipyDistribution(
                name='gaussian',
                a=min_included,
                b=max_included
            ),
            null_default_value=null_default_value
        )


class PoissonScipyDistribution(rv_discrete):
    def _pmf(self, k, mu):
        return math.exp(-mu) * mu ** k / factorial(k)


class Poisson(BaseWrappedScipyDistribution):
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
        :class:`Discrete`,
        :class:`Continuous`,
        :class:`HyperparameterDistribution`,
        :class:`neuraxle.hyperparams.space.HyperparameterSamples`,
        :class:`neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`neuraxle.base.BaseStep`
    """

    def __init__(self, min_included: float, max_included: float, null_default_value: float = None, mu=0.6):
        super().__init__(
            scipy_distribution=PoissonScipyDistribution(
                a=min_included,
                b=max_included,
                name='poisson'
            ),
            null_default_value=null_default_value,
            mu=mu
        )

        self.mu = mu


class Histogram(BaseWrappedScipyDistribution, HyperparameterDistribution):
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
        :class:`BaseWrappedScipyDistribution`,
        :class:`Poisson`,
        :class:`Gaussian`,
        :func:`~neuraxle.base.BaseStep.set_hyperparams_space`,
        :class:`neuraxle.hyperparams.space.HyperparameterSamples`,
        :class:`neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`neuraxle.base.BaseStep`
    """

    def __init__(self, histogram: np.histogram, null_default_value: float = None, **kwargs):
        BaseWrappedScipyDistribution.__init__(
            self,
            scipy_distribution=rv_histogram(histogram=histogram, **kwargs),
            null_default_value=null_default_value
        )
