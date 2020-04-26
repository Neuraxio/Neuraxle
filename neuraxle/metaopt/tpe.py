"""
Tree parzen estimator
====================================
Code for tree parzen estimator auto ml.
"""
from collections import Counter

import numpy as np

from neuraxle.hyperparams.distributions import DistributionMixture, Choice, Quantized, LogNormal, LogUniform, RandInt, \
    PriorityChoice
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.metaopt.auto_ml import BaseHyperparameterSelectionStrategy, RandomSearchHyperparameterSelectionStrategy, \
    TRIAL_STATUS

_INDEPENDANT_DISCRET_DISTRIBUTION = (Choice, PriorityChoice, RandInt)
_LOG_DISTRIBUTION = (LogNormal, LogUniform)
_QUANTIZED_DISTRIBUTION = (Quantized,)


class TreeParzenEstimatorHyperparameterSelectionStrategy(BaseHyperparameterSelectionStrategy):

    def __init__(self, number_of_initial_random_step: int = 40, quantile_threshold: float = 0.3,
                 number_good_trials_max_cap: int = 25,
                 number_possible_hyperparams_candidates=100,
                 prior_weight: float = 0.,
                 use_linear_forgetting_weights: bool = False,
                 number_recent_trial_at_full_weights=25):
        super().__init__()
        self.initial_auto_ml_algo = RandomSearchHyperparameterSelectionStrategy()
        self.number_of_initial_random_step = number_of_initial_random_step
        self.quantile_threshold = quantile_threshold
        self.number_good_trials_max_cap = number_good_trials_max_cap
        self.number_possible_hyperparams_candidates = number_possible_hyperparams_candidates
        self.prior_weight = prior_weight
        self.use_linear_forgetting_weights = use_linear_forgetting_weights
        self.number_recent_trial_at_full_weights = number_recent_trial_at_full_weights

    def find_next_best_hyperparams(self, auto_ml_container: 'AutoMLContainer') -> HyperparameterSamples:
        """
        Find the next best hyperparams using previous trials.

        :param auto_ml_container: trials data container
        :type auto_ml_container: Trials
        :return: next best hyperparams
        :rtype: HyperparameterSamples
        """
        # Flatten hyperparameter space
        flat_hyperparameter_space = auto_ml_container.hyperparameter_space.to_flat()

        if auto_ml_container.trial_number < self.number_of_initial_random_step:
            # Perform random search
            return self.initial_auto_ml_algo.find_next_best_hyperparams(auto_ml_container)

        # Keep only success trials
        success_trials = auto_ml_container.trials.filter(TRIAL_STATUS.SUCCESS)

        # Split trials into good and bad using quantile threshold.
        good_trials, bad_trials = self._split_trials(success_trials)

        # Create gaussian mixture of good and gaussian mixture of bads.
        good_posteriors = self._create_posterior(flat_hyperparameter_space, good_trials)
        bad_posteriors = self._create_posterior(flat_hyperparameter_space, bad_trials)

        best_hyperparams = []
        for (hyperparam_key, good_posterior) in good_posteriors.items():
            best_new_hyperparam_value = None
            best_ratio = None
            for _ in range(self.number_possible_hyperparams_candidates):
                # Sample possible new hyperparams in the good_trials.
                possible_new_hyperparm = good_posterior.rvs()

                # Verify if we use the ratio directly or we use the loglikelihood of b_post under both distribution like hyperopt.
                # In hyperopt they use :
                # # calculate the log likelihood of b_post under both distributions
                # below_llik = fn_lpdf(*([b_post] + b_post.pos_args), **b_kwargs)
                # above_llik = fn_lpdf(*([b_post] + a_post.pos_args), **a_kwargs)
                #
                # # improvement = below_llik - above_llik
                # # new_node = scope.broadcast_best(b_post, improvement)
                # new_node = scope.broadcast_best(b_post, below_llik, above_llik)

                # Verify ratio good pdf versus bad pdf for all possible new hyperparms.
                # Used what is describe in the article which is the ratio (gamma + g(x) / l(x) ( 1- gamma))^-1 that we have to maximize.
                # Since there is ^-1, we have to maximize l(x) / g(x)
                # Only the best ratio is kept and is the new best hyperparams.
                # Seems to take log of pdf and not pdf directly probable to have `-` instead of `/`.
                # TODO: Maybe they use the likelyhood to sum over all possible parameters to find the max so it become a join distribution of all hyperparameters, would make sense.
                # TODO: verify is for quantized we do not want to do cdf(value higher) - cdf(value lower) to have pdf.
                ratio = good_posterior.pdf(possible_new_hyperparm) / bad_posteriors[hyperparam_key].pdf(
                    possible_new_hyperparm)

                if best_new_hyperparam_value is None:
                    best_new_hyperparam_value = possible_new_hyperparm
                    best_ratio = ratio
                else:
                    if ratio > best_ratio:
                        best_new_hyperparam_value = possible_new_hyperparm
                        best_ratio[hyperparam_key] = ratio

            best_hyperparams.append((hyperparam_key, best_new_hyperparam_value))
        return HyperparameterSamples(best_hyperparams)

    def _split_trials(self, success_trials):
        # Split trials into good and bad using quantile threshold.
        # TODO: maybe better place in the Trials class.
        # TODO: manage higher_score_is_better.
        trials_scores = np.array([trial.score for trial in success_trials])
        trial_sorted_indexes = np.argsort(trials_scores)

        # In hyperopt they use this to split, where default_gamma_cap = 25. They clip the max of item they use in the good item.
        # default_gamma_cap is link to the number of recent_trial_at_full_weight also.
        # n_below = min(int(np.ceil(gamma * np.sqrt(len(l_vals)))), gamma_cap)
        n_good = int(min(np.ceil(self.quantile_threshold * len(trials_scores)), self.number_good_trials_max_cap))

        good_trials_indexes = trial_sorted_indexes[:n_good]
        bad_trials_indexes = trial_sorted_indexes[n_good:]

        good_trials = []
        bad_trials = []
        for trial_index, trial in enumerate(success_trials):
            if trial_index in good_trials_indexes:
                good_trials.append(trial)
            if bad_trials_indexes in bad_trials_indexes:
                bad_trials.append(trial)
        return good_trials, bad_trials

    def _create_posterior(self, flat_hyperparameter_space, trials):
        # Create a list of all hyperparams and their trials.

        # Loop through all hyperparams
        posterior_distributions = {}
        for (hyperparam_key, hyperparam_distribution) in flat_hyperparameter_space.items():

            # Get distribution_trials
            distribution_trials = [trial.hyperparams.to_flat_as_dict_primitive()[hyperparam_key] for trial in trials]

            if isinstance(hyperparam_distribution, _INDEPENDANT_DISCRET_DISTRIBUTION):
                # If hyperparams is a discret distribution
                posterior_distribution = self._reweights_categorical(hyperparam_distribution, distribution_trials)

            else:
                # If hyperparams is a continuous distribution
                posterior_distribution = self._create_gaussian_mixture(hyperparam_distribution, distribution_trials)

            posterior_distributions[hyperparam_key] = posterior_distribution
        return posterior_distributions

    def _reweights_categorical(self, hyperparam_distribution, distribution_trials):

        # For discret categorical distribution
        # We need to reweights probability depending on trial counts.
        # TODO: add a probas method to access all probas.
        probas = hyperparam_distribution.probas()

        # TODO: add a values method to acess all values.
        values = hyperparam_distribution.values()

        number_probas = len(probas)

        # Since the reweighted is proportional to N*p_i + C_i,
        # where N is the number of probas, p_i is the original probas and c_i is the count.
        reweighted_probas = number_probas * probas

        # Count number of occurence for each values.
        count_trials = Counter(distribution_trials)
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

        if isinstance(hyperparam_distribution, PriorityChoice):
            return PriorityChoice(values, probas=reweighted_probas)

        return Choice(values, probas=reweighted_probas)

    def _create_gaussian_mixture(self, hyperparam_distribution, distribution_trials):

        # TODO: see how to manage distribution mixture here.

        use_logs = False
        if isinstance(hyperparam_distribution, _LOG_DISTRIBUTION):
            use_logs = True

        use_quantized_distributions = False
        if isinstance(hyperparam_distribution, Quantized):
            use_quantized_distributions = True
            if isinstance(hyperparam_distribution.hd, _LOG_DISTRIBUTION):
                use_logs = True

        # Find means, std, amplitudes, min and max.
        distribution_amplitudes, means, stds, distributions_mins, distributions_max = self._adaptive_parzen_normal(
            hyperparam_distribution,
            distribution_trials)

        # Create appropriate gaussian mixture that wrapped all hyperparams.
        gmm = DistributionMixture.build_gaussian_mixture(distribution_amplitudes, means=means, stds=stds,
                                                         distributions_mins=distributions_mins,
                                                         distributions_max=distributions_max, use_logs=use_logs,
                                                         use_quantized_distributions=use_quantized_distributions)

        return gmm

    def _adaptive_parzen_normal(self, hyperparam_distribution, distribution_trials):
        """This code is enterily inspire from Hyperopt (https://github.com/hyperopt) code."""

        # TODO: check if someone use the DistributionMixture how to manage it in here.
        # TODO: Distribution Mixture : Treat has a standard distribution or prior distribution is all small gaussian for each distribution in the distribution mixture.

        use_prior = (self.prior_weight - 0.) > 1e-10

        prior_mean = hyperparam_distribution.mean()
        prior_sigma = hyperparam_distribution.std()

        means = distribution_trials
        distributions_mins = hyperparam_distribution.min() * len(means)
        distributions_max = hyperparam_distribution.max() * len(means)

        # Index to sort in increasing order the means.
        # Easier in order to insert prior.
        sort_indexes = np.argsort(means)

        if len(means) == 0:
            if use_prior:
                prior_pos = 0
                sorted_means = np.array([prior_mean])
                sorted_stds = np.array([prior_sigma])
        elif len(means) == 1:
            if use_prior and prior_mean < means[0]:
                prior_pos = 0
                sorted_means = np.array([prior_mean, means[0]])
                sorted_stds = np.array([prior_sigma, prior_sigma * 0.5])
            elif use_prior and prior_mean >= means[0]:
                prior_pos = 1
                sorted_means = np.array([means[0], prior_mean])
                sorted_stds = np.array([prior_sigma * 0.5, prior_sigma])
            else:
                sorted_means = means
                sorted_stds = prior_sigma
        else:
            if use_prior:
                # Insert the prior at the right place.
                prior_pos = np.searchsorted(means[sort_indexes], prior_mean)
                sorted_means = np.zeros(len(means) + 1)
                sorted_means[:prior_pos] = means[sort_indexes[:prior_pos]]
                sorted_means[prior_pos] = prior_mean
                sorted_means[prior_pos + 1:] = means[sort_indexes[prior_pos:]]
            else:
                sorted_means = means[sort_indexes]

            sorted_stds = np.zeros_like(sorted_means)
            sorted_stds[1:-1] = np.maximum(sorted_means[1:-1] - sorted_means[0:-2],
                                           sorted_means[2:] - sorted_means[1:-1])
            left_std = sorted_means[1] - sorted_means[0]
            right_std = sorted_means[-1] - sorted_means[-2]
            sorted_stds[0] = left_std
            sorted_stds[-1] = right_std

        # Magic formula from hyperopt.
        # -- magic formula:
        # maxsigma = old_div(prior_sigma, 1.0)
        # minsigma = old_div(prior_sigma, min(100.0, (1.0 + len(srtd_mus))))
        #
        # sigma = np.clip(sigma, minsigma, maxsigma)
        # sigma[prior_pos] = prior_sigma
        min_std = prior_sigma / min(100.0, (1.0 + len(sorted_means)))
        max_std = prior_sigma / 1.0
        sorted_stds = np.clip(sorted_stds, min_std, max_std)

        if self.use_linear_forgetting_weights:
            distribution_amplitudes = _linear_forgetting_Weights(len(means), self.number_recent_trial_at_full_weights)
        else:
            # From tpe article : TPE substitutes an equally-weighted mixture of that prior with Gaussians centered at each observations.
            distribution_amplitudes = np.ones(len(means))

        if use_prior:
            sorted_stds[prior_pos] = prior_sigma
            sorted_distribution_amplitudes = np.zeros_like(sorted_means)
            sorted_distribution_amplitudes[:prior_pos] = distribution_amplitudes[sort_indexes[:prior_pos]]
            sorted_distribution_amplitudes[prior_pos] = sort_indexes
            sorted_distribution_amplitudes[prior_pos + 1:] = distribution_amplitudes[sort_indexes[prior_pos:]]
        else:
            sorted_distribution_amplitudes = distribution_amplitudes

        # Normalize distribution amplitudes.
        distribution_amplitudes = np.array(distribution_amplitudes)
        distribution_amplitudes /= np.sum(distribution_amplitudes)

        return sorted_distribution_amplitudes, sorted_means, sorted_stds, distributions_mins, distributions_max


def _linear_forgetting_Weights(number_samples, number_recent_trial_at_full_weights):
    """This code has been taken from Hyperopt (https://github.com/hyperopt) code."""
    if number_samples == 0:
        return np.asarray([])

    if number_samples < number_recent_trial_at_full_weights:
        return np.ones(number_samples)

    weights_ramp = np.linspace(1.0 / number_samples, 1.0, number_samples - number_recent_trial_at_full_weights)
    weights_flat = np.ones(number_recent_trial_at_full_weights)
    weights = np.concatenate((weights_ramp, weights_flat), axis=0)

    return weights
