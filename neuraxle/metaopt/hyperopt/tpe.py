"""
Tree parzen estimator
====================================
Code for tree parzen estimator auto ml.
"""
from collections import Counter
from operator import itemgetter
from typing import Any, List, Tuple, Union

import numpy as np
from neuraxle.hyperparams.distributions import (
    Choice, DiscreteHyperparameterDistribution, DistributionMixture,
    HPSampledValue, HyperparameterDistribution, LogSpaceDistributionMixin,
    PriorityChoice, Quantized)
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace)
from neuraxle.metaopt.data.reporting import RoundReport, TrialReport
from neuraxle.metaopt.optimizer import (BaseHyperparameterOptimizer,
                                        GridExplorationSampler)

_QUANTIZED_DISTRIBUTION = (Quantized,)


class TreeParzenEstimator(BaseHyperparameterOptimizer):
    """
    This is a Tree Parzen Estimator (TPE) algorithm as found in Hyperopt,
    that is better than the Random Search algorithm and supports supporting
    intelligent exploration v.s. exploitation of the search space over time,
    using Neuraxle hyperparameters.

    Here, the algorithm is modified compared to the original one, as it uses a
    :class:`neuraxle.metaopt.automl.GridExplorationSampler` instead of a random search
    to pick the first exploration samples to furthermore explore the space at the beginning.
    """

    def __init__(
            self,
            number_of_initial_random_step: int = 15,
            quantile_threshold: float = 0.3,
            number_good_trials_max_cap: int = 25,
            number_possible_hyperparams_candidates: int = 100,
            use_linear_forgetting_weights: bool = False,
            number_recent_trials_at_full_weights: int = 25
    ):
        """
        Initialize the TPE with some configuration.

        :param number_of_initial_random_step: Number of random steps to take before starting the optimization.
        :param quantile_threshold: threshold between 0 and 1 representing the proportion of good trials to keep.
        :param number_good_trials_max_cap: maximum number of good trials to keep, that will cap the `quantile_threshold`.
        :param number_possible_hyperparams_candidates: number of possible hyperparams candidates to explore in the `good / bad` trials posterior.
        :param use_linear_forgetting_weights: if True, the weights will be linearly decreasing past the `number_recent_trial_at_full_weights`.
        :param number_recent_trials_at_full_weights: number of recent trials to use at full weights before linear forgetting when `use_linear_forgetting_weights` is set to True.
        """
        super().__init__()

        self.number_of_initial_random_step: int = number_of_initial_random_step
        self.number_possible_hyperparams_candidates: int = number_possible_hyperparams_candidates

        self.mixture_factory: _DividedMixturesFactory = _DividedMixturesFactory(
            quantile_threshold,
            number_good_trials_max_cap,
            use_linear_forgetting_weights,
            number_recent_trials_at_full_weights,
        )

        self.initial_auto_ml_algo: BaseHyperparameterOptimizer = GridExplorationSampler(
            expected_n_trials=number_of_initial_random_step
        )

    def find_next_best_hyperparams(self, _round: RoundReport, hp_space: HyperparameterSpace) -> HyperparameterSamples:
        """
        Find the next best hyperparams using previous trials.

        :param round_scope: round scope
        :return: next best hyperparams
        """

        # Perform a first pseudo-randomized search:
        if len(_round) < self.number_of_initial_random_step:
            return self.initial_auto_ml_algo.find_next_best_hyperparams(_round, hp_space)

        # Create gaussian mixture of good and gaussian mixture of bads. Lists here are on a per-hp basis:
        hyperparams_keys, divided_good_and_bad_distrs = self.mixture_factory.create_from(_round, hp_space)

        # Sample the next hyperparams finally:
        return self._sample_next_hyperparams_from_gaussians_div(
            hyperparams_keys, divided_good_and_bad_distrs)

    def _sample_next_hyperparams_from_gaussians_div(
        self,
        hp_keys: List[str],
        divided_good_and_bad_distrs: List['_DividedTPEPosteriors']
    ) -> HyperparameterSamples:
        best_hyperparams: List[Tuple[str, HPSampledValue]] = []

        for (hp_key, good_bad_posterior_div) in zip(hp_keys, divided_good_and_bad_distrs):

            best_next_hyperparam_value: HPSampledValue = self._sample_one_hp_from_gaussians_div(
                hp_key, good_bad_posterior_div)

            best_hyperparams.append((hp_key, best_next_hyperparam_value))

        return HyperparameterSamples(best_hyperparams)

    def _sample_one_hp_from_gaussians_div(
        self,
        hp_key: List[str],
        good_bad_posterior_div: '_DividedTPEPosteriors'
    ) -> HPSampledValue:

        best_next_hyperparam_value: HPSampledValue = None
        best_ratio: float = None
        for _ in range(self.number_possible_hyperparams_candidates):

            possible_new_hyperparm, proba_ratio = good_bad_posterior_div.rvs_good_with_pdf_division_proba()

            if best_next_hyperparam_value is None or proba_ratio > best_ratio:
                best_next_hyperparam_value = possible_new_hyperparm
                best_ratio = proba_ratio

        return best_next_hyperparam_value


class _DividedMixturesFactory:

    def __init__(
        self,
        quantile_threshold: float,
        number_good_trials_max_cap: int,
        use_linear_forgetting_weights: bool,
        number_recent_trials_at_full_weights: int,
    ):
        self.quantile_threshold: float = quantile_threshold
        self.number_good_trials_max_cap: int = number_good_trials_max_cap
        self.use_linear_forgetting_weights = use_linear_forgetting_weights
        self.number_recent_trials_at_full_weights = number_recent_trials_at_full_weights

    def create_from(
        self,
        round: RoundReport,
        hp_space: HyperparameterSpace,
    ) -> Tuple[List[str], List['_DividedTPEPosteriors']]:
        # TODO: pass a RoundReport instead. This will require the ability to save hyperparameter spaces in round reports, and update them.

        # Split trials into good and bad using quantile threshold.
        good_trials: List[TrialReport] = []
        bad_trials: List[TrialReport] = []
        good_trials, bad_trials = self._split_good_and_bad_trials(round)

        flat_hp_space_tuples: List[(str, HyperparameterDistribution)] = list(
            hp_space.to_flat_dict().items())
        hyperparams_keys: List[str] = list(map(itemgetter(0), flat_hp_space_tuples))

        good_posteriors: List[HyperparameterDistribution] = self._create_posterior(
            flat_hp_space_tuples, good_trials)
        bad_posteriors: List[HyperparameterDistribution] = self._create_posterior(
            flat_hp_space_tuples, bad_trials)

        divided_good_and_bad_distrs: _DividedMixturesFactory = [
            _DividedTPEPosteriors(g, b)
            for (g, b)
            in zip(good_posteriors, bad_posteriors)
        ]

        return hyperparams_keys, divided_good_and_bad_distrs

    def _split_good_and_bad_trials(self, round_report: RoundReport) -> Tuple[List[TrialReport], List[TrialReport]]:

        # Split trials into good and bad using quantile threshold.
        trials_scores: List[float] = round_report.list_successful_avg_validation_scores()

        trial_sorted_indexes: List[int] = np.argsort(trials_scores)
        if round_report.is_higher_score_better():
            trial_sorted_indexes = list(reversed(trial_sorted_indexes))

        # In hyperopt they use this to split, where default_gamma_cap = 25. They clip the max of item they use in the good item.
        # default_gamma_cap is link to the number of recent_trial_at_full_weight also.
        # n_below = min(int(np.ceil(gamma * np.sqrt(len(l_vals)))), gamma_cap)
        n_good = int(min(
            self.number_good_trials_max_cap,
            np.ceil(self.quantile_threshold * len(trials_scores)),
        ))

        good_trials_indexes = trial_sorted_indexes[:n_good]
        bad_trials_indexes = trial_sorted_indexes[n_good:]

        good_trials: List[TrialReport] = [round_report.successful_trials[i] for i in good_trials_indexes]
        bad_trials: List[TrialReport] = [round_report.successful_trials[i] for i in bad_trials_indexes]

        return good_trials, bad_trials

    def _create_posterior(
        self,
        flat_hp_space_tuples: List[Tuple[str, HyperparameterDistribution]],
        trials: List[TrialReport],
    ) -> List[HyperparameterDistribution]:

        # Loop through all hyperparams
        posterior_distributions: List[HyperparameterDistribution] = []
        for (hp_key, hp_distribution) in flat_hp_space_tuples:
            # Typing:
            hp_key: str = hp_key
            hp_distribution: HyperparameterDistribution = hp_distribution

            # Get these trials' hyperparam values for this hyperparam
            trial_hyperparams: List[HPSampledValue] = [trial.get_hyperparams()[hp_key] for trial in trials]

            if hp_distribution.is_discrete():
                posterior_distribution: Union[Choice, PriorityChoice] = self._reweights_categorical(
                    hp_distribution, trial_hyperparams)
            else:
                posterior_distribution: DistributionMixture = self._create_gaussian_mixture(
                    hp_distribution, trial_hyperparams)

            posterior_distributions.append(posterior_distribution)

        return posterior_distributions

    def _reweights_categorical(
        self,
        discrete_distribution: DiscreteHyperparameterDistribution,
        trial_hyperparameters: List[HPSampledValue]
    ) -> Union[Choice, PriorityChoice]:
        # For discrete categorical distribution
        # We need to reweights probability depending on trial counts.
        probas: List[float] = discrete_distribution.probabilities()
        values: List[Any] = discrete_distribution.values()

        number_probas = len(probas)

        # Since the reweighted is proportional to N*p_i + C_i,
        # where N is the number of probas, p_i is the original probas and c_i is the count.
        reweighted_probas = np.array(number_probas) * probas

        # Count number of occurence for each values.
        count_trials = Counter(trial_hyperparameters)
        values_keys = list(count_trials.keys())
        counts = list(count_trials.values())

        for value_key, count in zip(values_keys, counts):
            # find index in the orignal probas and values.
            index_value = values.index(value_key)

            # Calculate reweigthed proba.
            reweighted_probas[index_value] += count

        # Normalize reweighted probas
        reweighted_probas = np.array(reweighted_probas)
        reweighted_probas = reweighted_probas / np.sum(reweighted_probas)

        if isinstance(discrete_distribution, PriorityChoice):
            return PriorityChoice(values, probas=reweighted_probas)

        return Choice(values, probas=reweighted_probas)

    def _create_gaussian_mixture(
            self,
            continuous_distribution: HyperparameterDistribution,
            trial_hyperparameters: List[HPSampledValue]
    ) -> DistributionMixture:

        use_logs = False
        if isinstance(continuous_distribution, LogSpaceDistributionMixin):
            use_logs = True

        use_quantized_distributions = False
        if isinstance(continuous_distribution, Quantized):
            use_quantized_distributions = True
            if isinstance(continuous_distribution.hd, LogSpaceDistributionMixin):
                use_logs = True

        # Find means, std, amplitudes, min and max.
        distribution_amplitudes, means, stds, distributions_mins, distributions_max = self._adaptive_parzen_normal(
            continuous_distribution,
            trial_hyperparameters)

        # Create appropriate gaussian mixture that wrapped all hyperparams.
        gmm: DistributionMixture = DistributionMixture.build_gaussian_mixture(
            distribution_amplitudes=distribution_amplitudes,
            means=means,
            stds=stds,
            distributions_mins=distributions_mins,
            distributions_max=distributions_max,
            use_logs=use_logs,
            use_quantized_distributions=use_quantized_distributions
        )

        return gmm

    def _adaptive_parzen_normal(self, hyperparam_distribution: HyperparameterDistribution, distribution_trials: List[HPSampledValue]):

        hp_std = hyperparam_distribution.std()

        means = np.array(distribution_trials)
        distributions_mins = hyperparam_distribution.min() * np.ones_like(means)
        distributions_max = hyperparam_distribution.max() * np.ones_like(means)

        means, stds = self._compute_distributions_means_stds(hyperparam_distribution, distribution_trials)

        # Magic stds formula from hyperopt:
        min_std = hp_std / min(100.0, (1.0 + len(means)))
        max_std = hp_std / 1.0
        stds = np.clip(stds, min_std, max_std)

        distribution_amplitudes: np.ndarray = self._generate_linear_forget_weights(len(means))

        return distribution_amplitudes, means, stds, distributions_mins, distributions_max

    def _compute_distributions_means_stds(self, hyperparam_distribution: HyperparameterDistribution, distribution_trials: List[HPSampledValue]):

        prior_mean = hyperparam_distribution.mean()
        prior_sigma = hyperparam_distribution.std()

        # Initialize with prior mean and sigma and replace when appropriate.
        means = np.array(sorted(distribution_trials)) if len(distribution_trials) > 0 else np.array([prior_mean])
        stds = np.array([prior_sigma])

        if len(means) > 1:
            # Compute the STDs specially with a far-left and far-right tail.
            stds = np.zeros_like(means)
            # Element-wise std computation is from the maximal difference in closest neighbor.
            stds[1:-1] = np.maximum(means[1:-1] - means[:-2],
                                    means[2:] - means[1:-1])
            left_std = means[1] - means[0]
            right_std = means[-1] - means[-2]
            stds[0] = left_std
            stds[-1] = right_std

        return means, stds

    def _generate_linear_forget_weights(self, number_samples: int) -> np.ndarray:
        # From the TPE article: TPE substitutes an equally-weighted mixture of that prior with Gaussians centered at each observations.

        if self.use_linear_forgetting_weights and number_samples >= self.number_recent_trials_at_full_weights:
            weights_ramp = np.linspace(
                start=1.0 / number_samples,
                stop=1.0,
                num=number_samples - self.number_recent_trials_at_full_weights
            )
            weights_flat = np.ones(self.number_recent_trials_at_full_weights)

            distribution_amplitudes = np.concatenate((weights_ramp, weights_flat), axis=0)
        else:
            distribution_amplitudes = np.ones(number_samples)

        return distribution_amplitudes


class _DividedTPEPosteriors:
    """
    Sample possible new hyperparams in the `good_trials`.

    Verify if we use the ratio directly or we use the loglikelihood of b_post under both distribution like hyperopt.
    In hyperopt, they use the following to calculate the log likelihood of b_post under both distributions:

    .. code-block:: python
        below_llik = fn_lpdf(*([b_post] + b_post.pos_args), **b_kwargs)
        above_llik = fn_lpdf(*([b_post] + a_post.pos_args), **a_kwargs)
        improvement = below_llik - above_llik
        new_node = scope.broadcast_best(b_post, improvement)
        new_node = scope.broadcast_best(b_post, below_llik, above_llik)

    Verify ratio good pdf versus bad pdf for all possible new hyperparams.

    The gist of it is that this is as dividing good probabilities by bad probabilities.

    We used what is described in the article which is the ratio (gamma + g(x) / l(x) ( 1- gamma))^-1 that we have to maximize.

    Since there is `^-1`, we have to maximize `l(x) / g(x)`.

    Only the best ratio is kept and is the new best hyperparams.

    They seem to take log of pdf and not pdf directly so it's probable to have `-` instead of `/` as an implementation. Regardless, both operators are transitive in nature for `max` comparisons in their regular or log space.

    # TODO: Maybe they use the likelyhood to sum over all possible parameters to find the max so it become a join distribution of all hyperparameters, would make sense.

    # TODO: verify is for quantized we do not want to do cdf(value higher) - cdf(value lower) to have pdf.
    """

    def __init__(self, good_trials: HyperparameterDistribution, bad_trials: HyperparameterDistribution):
        self.good_trials = good_trials
        self.bad_trials = bad_trials

    def rvs_good_with_pdf_division_proba(self) -> Tuple[HPSampledValue, float]:
        """
        Sample an hyperparameter from the good distribution and return the probability ratio.

        :return: sampled good hyperparameter and probability ratio
        """
        good_hyperparam = self.rvs_good()
        proba_ratio = self.proba_ratio(good_hyperparam)
        return good_hyperparam, proba_ratio

    def rvs_good(self) -> HPSampledValue:
        """
        Sample an hyperparameter from the good distribution.

        :return: sampled good hyperparameter
        """
        return self.good_trials.rvs()

    def proba_ratio(self, possible_new_hyperparm: HPSampledValue) -> float:
        """
        Return the probability ratio of the sampled good hyperparam.

        :return: probability ratio
        """
        proba_ratio = self.good_trials.pdf(possible_new_hyperparm) / self.bad_trials.pdf(possible_new_hyperparm)
        return proba_ratio
