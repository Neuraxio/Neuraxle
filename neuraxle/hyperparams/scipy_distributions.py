import math
from typing import Optional, List, Any, Union, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.special import factorial
from scipy.stats import rv_continuous, norm, rv_discrete, rv_histogram, truncnorm, randint

from neuraxle.hyperparams.distributions import HyperparameterDistribution, WrappedHyperparameterDistributions


class ScipyDistributionWrapper(HyperparameterDistribution):
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
        """
        Cumulative distribution function of the given x.

        :param x:
        :param args:
        :param kwargs:
        :return:
        """
        return self.sk_learn_distribution.cdf(x, *args, **self.kwargs, **kwargs)

    def entropy(self, *args, **kwargs):
        """
        Differential entropy of the RV.
        
        :return: 
        """
        return self.sk_learn_distribution.entropy(*args, **self.kwargs, **kwargs)

    def expect(self, *args, **kwargs):
        """
        Calculate expected value of a function with respect to the distribution by numerical integration.

        :param args:
        :param kwargs:
        :return:
        """
        return self.sk_learn_distribution.expect(*args, **self.kwargs, **kwargs)

    def fit(self, data, *args, **kwargs):
        """
        Return MLEs for shape( if applicable), location, and scale parameters from data.

        :param data:
        :param args:
        :param kwargs:
        :return:
        """
        return self.sk_learn_distribution.fit(data, *args, **self.kwargs, **kwargs)

    def fit_loc_scale(self, data, *args):
        """
        Estimate loc and scale parameters from data using 1 st and 2 nd moments. 
        
        :param data: 
        :return: 
        """
        return self.sk_learn_distribution.fit_loc_scale(data, *args)

    def freeze(self, *args, **kwargs):
        """
        Freeze the distribution for the given arguments.

        :return:
        """
        return self.sk_learn_distribution.freeze(*args, **kwargs)

    def interval(self, alpha, *args, **kwargs):
        """
        Confidence interval with equal areas around the median.

        :param alpha:
        :return:
        """
        return self.sk_learn_distribution.interval(alpha, *args, **kwargs)

    def isf(self, q, *args, **kwargs):
        """
        Inverse survival function(inverse of sf) at q of the given RV.

        :param q:
        :return:
        """
        return self.sk_learn_distribution.isf(q, *args, **kwargs)

    def logcdf(self, x, *args, **kwargs):
        """
        Log of the cumulative distribution function at x of the given RV.

        :param x:
        :param args:
        :param kwargs:
        :return:
        """
        return self.sk_learn_distribution.logcdf(x, *args, **kwargs)

    def logpdf(self, x, *args, **kwargs):
        """
        Log of the probability density function at x of the given RV.

        :param x:
        :param args:
        :param kwargs:
        :return:
        """
        return self.sk_learn_distribution.logpdf(x, *args, **kwargs)

    def logsf(self, x, *args, **kwargs):
        """
        Log of the survival function of the given RV.

        :param x:
        :return:
        """
        return self.sk_learn_distribution.logsf(x, *args, **kwargs)

    def mean(self, *args, **kwargs):
        """
        Mean of the distribution.

        :return:
        """
        return self.sk_learn_distribution.mean(*args, **kwargs)

    def median(self, *args, **kwargs):
        """
        Median of the distribution.

        :return:
        """
        return self.sk_learn_distribution.median(*args, **kwargs)

    def moment(self, n, *args, **kwargs):
        """
        n-th order non-central moment of distribution.

        :param n:
        :return:
        """
        return self.sk_learn_distribution.moment(n, *args, **kwargs)

    def nnlf(self, theta, x):
        """
        Return negative loglikelihood function.

        :param theta:
        :param x:
        :return:
        """
        return self.sk_learn_distribution.nnlf(theta, x)

    def ppf(self, q, *args, **kwargs):
        """
        Percent point function(inverse of cdf) at q of the given RV.

        :param q:
        :return:
        """
        return self.sk_learn_distribution.ppf(q, *args, **kwargs)

    def sf(self, x, *args, **kwargs):
        """
        Survival function(1 - cdf) at x of the given RV.

        :param x:
        :return:
        """
        return self.sk_learn_distribution.sf(x, *args, **kwargs)

    def stats(self, *args, **kwargs):
        """
        Some statistics of the given RV.

        :return:
        """
        return self.sk_learn_distribution.stats(*args, **kwargs)

    def std(self, *args, **kwargs):
        """
        Standard deviation of the distribution.

        :param args:
        :param kwargs:
        :return:
        """
        return self.sk_learn_distribution.std(*args, **kwargs)

    def support(self, *args, **kwargs):
        """
        Return the support of the distribution.

        :param args:
        :param kwargs:
        :return:
        """
        return self.sk_learn_distribution.support(*args, **kwargs)

    def var(self, *args, **kwargs):
        """
        Variance of the distribution.

        :return:
        """
        return self.sk_learn_distribution.var(*args, **kwargs)

    def to_sk_learn(self):
        return self.sk_learn_distribution


class RandInt(ScipyDistributionWrapper):
    """
    Rand int scipy distribution. Check out `scipy.stats.randint for more info <https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.randint.html>`_
    """

    def __init__(self, min_included: int, max_included: int, null_default_value: float = None):
        super().__init__(
            scipy_distribution=randint,
            low=min_included,
            high=max_included,
            null_default_value=null_default_value
        )


class UniformScipyDistribution(rv_continuous):
    """
    Uniform scipy distribution. Check out `scipy.stats.rv_continuous for more info <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous>`_
    """

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

    def _argcheck(self, *args):
        cond = 1
        for arg in args:
            cond = np.logical_and(
                cond,
                isinstance(arg, np.ndarray) and \
                (arg.dtype == np.float or arg.dtype == np.int)
            )
        return cond


class Uniform(ScipyDistributionWrapper):
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

        ScipyDistributionWrapper.__init__(
            self,
            scipy_distribution=UniformScipyDistribution(name='uniform', a=min_included, b=max_included),
            min_included=min_included,
            max_included=max_included,
            null_default_value=null_default_value
        )


class LogUniformScipyDistribution(rv_continuous):
    """
    Log Uniform scipy distribution. Check out `scipy.stats.rv_continuous for more info <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous>`_
    """

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

    def _argcheck(self, *args):
        cond = 1
        for arg in args:
            cond = np.logical_and(
                cond,
                isinstance(arg, np.ndarray) and \
                (arg.dtype == np.float or arg.dtype == np.int)
            )
        return cond


class LogUniform(ScipyDistributionWrapper):
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

        ScipyDistributionWrapper.__init__(
            self,
            scipy_distribution=LogUniformScipyDistribution(
                a=self.min_included,
                b=self.max_included,
                name='log_uniform'
            ),
            null_default_value=null_default_value,
            log2_min_included=self.log2_min_included,
            log2_max_included=self.log2_max_included,
        )


class NormalScipyDistribution(rv_continuous):
    """
    Normal scipy distribution. Check out `scipy.stats.rv_continuous for more info <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous>`_
    """

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

    def _argcheck(self, *args):
        cond = 1
        for arg in args:
            cond = np.logical_and(
                cond,
                isinstance(arg, np.ndarray) and \
                (arg.dtype == np.float or arg.dtype == np.int)
            )
        return cond


class Normal(ScipyDistributionWrapper):
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
            scipy_distribution=NormalScipyDistribution(
                name='normal',
                a=hard_clip_min,
                b=hard_clip_max
            ),
            mean=mean,
            std=std,
            hard_clip_min=hard_clip_min,
            hard_clip_max=hard_clip_max,
            null_default_value=null_default_value
        )


class LogNormalScipyDistribution(rv_continuous):
    """
    Log Normal scipy distribution. Check out `scipy.stats.rv_continuous for more info <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous>`_
    """

    def _pdf(self, x, log2_space_mean, log2_space_std):
        if x <= 0:
            return 0.

        cdf_min = 0.
        cdf_max = 1.

        pdf_x = 1 / (x * math.log(2) * log2_space_std * math.sqrt(2 * math.pi)) * math.exp(
            -(math.log2(x) - log2_space_mean) ** 2 / (2 * log2_space_std ** 2))
        return pdf_x / (cdf_max - cdf_min)

    def _argcheck(self, *args):
        cond = 1
        for arg in args:
            cond = np.logical_and(
                cond,
                isinstance(arg, np.ndarray) and \
                (arg.dtype == np.float or arg.dtype == np.int or arg == np.array(None))
            )
        return cond


class LogNormal(ScipyDistributionWrapper):
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

        super().__init__(
            scipy_distribution=LogNormalScipyDistribution(
                name='log_normal',
                a=hard_clip_min,
                b=hard_clip_max
            ),
            null_default_value=null_default_value,
            log2_space_mean=log2_space_mean,
            log2_space_std=log2_space_std
        )


class GaussianScipyDistribution(rv_continuous):
    """
    Gaussian scipy distribution. Check out `scipy.stats.rv_continuous for more info <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous>`_
    """

    def _pdf(self, x):
        return math.exp(-x ** 2 / 2.) / np.sqrt(2.0 * np.pi)


class Gaussian(ScipyDistributionWrapper):
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
        ScipyDistributionWrapper.__init__(
            self,
            scipy_distribution=GaussianScipyDistribution(
                name='gaussian',
                a=min_included,
                b=max_included
            ),
            null_default_value=null_default_value
        )


class PoissonScipyDistribution(rv_discrete):
    """
    Poisson scipy distribution. Check out `scipy.stats.rv_discrete for more info <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html#scipy.stats.rv_discrete>`_
    """

    def _pmf(self, k, mu):
        return math.exp(-mu) * mu ** k / factorial(k)


class Poisson(ScipyDistributionWrapper):
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
            scipy_distribution=PoissonScipyDistribution(
                a=min_included,
                b=max_included,
                name='poisson'
            ),
            null_default_value=null_default_value,
            mu=mu
        )

        self.mu = mu


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
            scipy_distribution=rv_histogram(histogram=histogram, **kwargs),
            null_default_value=null_default_value
        )


class FixedScipyDistribution(rv_continuous):
    def _pdf(self, x, value):
        if x == value:
            return 1.
        return 0.


class FixedHyperparameter(ScipyDistributionWrapper):
    """This is an hyperparameter that won't change again, but that is still expressed as a distribution."""

    def __init__(self, value, null_default_value=None):
        """
        Create a still hyperparameter

        :param value: what will be returned by calling ``.rvs()``.
        """
        self.value = value
        ScipyDistributionWrapper.__init__(
            self,
            scipy_distribution=FixedScipyDistribution(),
            value=value,
            null_default_value=null_default_value
        )


class BooleanScipyDistribution(rv_continuous):
    def _pdf(self, x, proba_is_true):
        """
        Calculate the boolean probability mass function value at position `x`.
        :param x: value where the probability mass function is evaluated.
        :return: value of the probability mass function.
        """
        if (x is True) or (x == 1):
            return proba_is_true

        if (x is False) or (x == 0):
            return 1 - proba_is_true

        return 0.


class Boolean(ScipyDistributionWrapper):
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
        ScipyDistributionWrapper.__init__(
            self,
            BooleanScipyDistribution(),
            null_default_value=null_default_value,
            proba_is_true=proba_is_true
        )


class ChoiceScipyDistribution(rv_discrete):
    def _pmf(self, x, choice_list, probas):
        """
        Calculate the choice probability mass function value at position `x`.
        :param x: value where the probability mass function is evaluated.
        :return: value of the probability mass function.
        """
        try:
            x_in_choice = x in choice_list
        except (TypeError, ValueError, AttributeError):
            raise ValueError(
                "Item not find in list. Make sure the item is in the choice list and a correct method  __eq__ is defined for all item in the list.")
        else:
            if x_in_choice:
                index = get_index_in_list_with_bool(choice_list, x)
                return probas[index]

        return 0.


class PriorityChoiceScipyDistribution(rv_discrete):
    def _pmf(self, x, choice_list, probas):
        """
        Calculate the choice probability mass function value at position `x`.
        :param x: value where the probability mass function is evaluated.
        :return: value of the probability mass function.
        """
        try:
            x_in_choice = x in choice_list
        except (TypeError, ValueError, AttributeError):
            raise ValueError(
                "Item not find in list. Make sure the item is in the choice list and a correct method  __eq__ is defined for all item in the list.")
        else:
            if x_in_choice:
                index = get_index_in_list_with_bool(choice_list, x)
                return probas[index]

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


class Choice(ScipyDistributionWrapper):
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
            ScipyDistributionWrapper.__init__(
                self,
                scipy_distribution=ChoiceScipyDistribution(),
                choice_list=choice_list,
                probas=probas,
                null_default_value=choice_list[0]
            )
        elif null_default_value in choice_list:
            ScipyDistributionWrapper.__init__(
                self,
                scipy_distribution=ChoiceScipyDistribution(),
                choice_list=choice_list,
                probas=probas,
                null_default_value=null_default_value
            )
        else:
            raise ValueError(
                'invalid default value {0} not in choice list : {1}'.format(null_default_value, choice_list))


class PriorityChoice(ScipyDistributionWrapper):
    """Get a random value from a choice list of possible value for this hyperparameter.

    The first parameters are kept until the end when the list is narrowed (it is narrowed progressively),
    unless there is a best guess that surpasses some of the top choices.
    """

    def __init__(self, choice_list: List, probas: Optional[List[float]] = None, null_default_value=None):
        """
        Create a random choice hyperparameter from the given list (choice_list).
        The first parameters in the choice_list will be kept longer when narrowing the space.

        :param choice_list: a list of values to sample from.
        :type choice_list: List
        :param null_default_value: default value for distribution
        :type null_default_value: default choice value. if None, default choice value will be the first choice
        """
        self.choice_list = choice_list

        # Normalize probas juste in case sum is not equal to one.
        self.probas = np.array(probas) / np.sum(probas)

        if probas is None:
            probas = [1 / len(self.choice_list) for _ in self.choice_list]

        if null_default_value is None:
            ScipyDistributionWrapper.__init__(
                self,
                scipy_distribution=PriorityChoiceScipyDistribution(),
                probas=probas,
                choice_list=choice_list,
                null_default_value=choice_list[0]
            )
        elif null_default_value in choice_list:
            ScipyDistributionWrapper.__init__(
                self,
                scipy_distribution=PriorityChoiceScipyDistribution(),
                probas=probas,
                choice_list=choice_list,
                null_default_value=null_default_value
            )
        else:
            raise ValueError(
                'invalid default value {0} not in choice list : {1}'.format(null_default_value, choice_list))


class QuantizedScipyDistribution(rv_continuous):
    def _pdf(self, x, hd):
        """
        Calculate the Quantized probability mass function value at position `x` of a continuous distribution.
        :param x: value where the probability mass function is evaluated.
        :return: value of the probability mass function.
        """
        # In order to calculate the pdf for any quantized distribution,
        # we have to perform the integral from x-0.5 to x+0.5 (because of round).
        if isinstance(x, int) or (isinstance(x, float) and x.is_integer()):
            return quad(hd.pdf, x - 0.5, x + 0.5)[0]
        return 0.


class Quantized(WrappedHyperparameterDistributions, ScipyDistributionWrapper):
    """A quantized wrapper for another distribution: will round() the rvs number."""

    def __init__(self, hd: HyperparameterDistribution = None, hds: List[HyperparameterDistribution] = None, null_default_value = None):
        WrappedHyperparameterDistributions.__init__(self, hd=hd, hds=hds, null_default_value=null_default_value)
        ScipyDistributionWrapper.__init__(
            self,
            scipy_distribution=QuantizedScipyDistribution(),
            hd=self.hd,
            null_default_value=self.null_default_value
        )


class DistributionMixtureScipyDistribution(rv_continuous):
    def _pdf(self, x, distributions, distribution_amplitudes):
        """
        Calculate the mixture probability distribution value at position `x`.

        :param x: value where the probability distribution function is evaluated.
        :return: value of the probability distribution function.
        """
        pdf_result = 0

        for distribution_amplitude, distribution in zip(distribution_amplitudes, distributions):
            pdf_result += (distribution_amplitude * distribution.pdf(x))

        return pdf_result


class DistributionMixture(ScipyDistributionWrapper):
    """Get a mixture of multiple distribution"""

    def __init__(
            self,
            distributions: Union[List[HyperparameterDistribution], Tuple[HyperparameterDistribution, ...]],
            distribution_amplitudes: Union[List[float], Tuple[float, ...]]
    ):
        """
        Create a mixture of multiple distributions.

        Distribution amplitude are normalized to make sure that the sum equals one.
        This normalization ensure to keep a random variable at the end (0 < probability < 1).

        :param distributions: list of multiple instantiated distribution.
        :param distribution_amplitudes: list of float representing the amplitude in the probability distribution function for each distribution.
        """
        # Normalize distribution amplitude
        distribution_amplitudes = np.array(distribution_amplitudes)
        distribution_amplitudes = distribution_amplitudes / np.sum(distribution_amplitudes)
        self.distributions = distributions
        self.distribution_amplitudes = distribution_amplitudes

        ScipyDistributionWrapper.__init__(
            self,
            scipy_distribution=DistributionMixtureScipyDistribution(),
            distributions=distributions,
            distribution_amplitudes=distribution_amplitudes,
            null_default_value=None
        )

    @staticmethod
    def build_gaussian_mixture(
            distribution_amplitudes: Union[List[float], Tuple[float, ...]],
            means: Union[List[float], Tuple[float, ...]],
            stds: Union[List[float], Tuple[float, ...]],
            distributions_mins: Union[List[float], Tuple[float, ...]],
            distributions_max: Union[List[float], Tuple[float, ...]],
            use_logs: bool = False,
            use_quantized_distributions: bool = False
    ):
        """
        Create a gaussian mixture.

         Will create a mixture distribution which consist of multiple gaussians of different amplitudes, means, standard deviations, mins and max.

        :param distribution_amplitudes: list of different amplitudes to infer to the mixture. The amplitudes are normalized to sum to 1.
        :param means: list of means to infer mean to each gaussian.
        :param stds: list of standard deviations to infer standard deviation to each gaussian.
        :param distributions_mins: list of minimum value that will clip each gaussian. If it is -Inf or None, it will not be clipped.
        :param distributions_max: list of maximal value that will clip each gaussian. If it is +Inf or None, it will not be clipped.
        :param distributions_max: bool weither to use a quantized version or not.
        :param use_logs:
        :param use_quantized_distributions:

        :return DistributionMixture instance
        """

        distribution_class = Normal

        if use_logs:
            distribution_class = LogNormal

        distributions = []

        for mean, std, single_min, single_max in zip(means, stds, distributions_mins, distributions_max):

            if single_min is None or np.isneginf(single_min):
                single_min = None

            if single_max is None or np.isposinf(single_max):
                single_max = None

            distribution_instance = distribution_class(mean, std, hard_clip_min=single_min, hard_clip_max=single_max)

            if use_quantized_distributions:
                distribution_instance = Quantized(distribution_instance)

            distributions.append(distribution_instance)

        return DistributionMixture(distributions, distribution_amplitudes)

