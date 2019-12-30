"""
Tree parzen estimator
====================================
Code for tree parzen estimator auto ml.
"""
import numpy as np
from typing import Optional
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.hyperparams.distributions import DistributionMixture
from .auto_ml import BaseHyperparameterOptimizer, RandomSearchHyperparameterOptimizer, TRIAL_STATUS


class TreeParzenEstimatorHyperparameterOptimizer(BaseHyperparameterOptimizer):

    def __init__(self, number_of_initial_random_step: int = 40, quantile_threshold: float = 30,
                 number_possible_hyperparams_sample=100,
                 use_prior_normal_distribution: bool = True,
                 use_linear_forgetting_weights: bool = False,
                 number_recent_trial_at_full_weights=25):
        super().__init__()
        self.initial_auto_ml_algo = RandomSearchHyperparameterOptimizer()
        self.number_of_initial_random_step = number_of_initial_random_step
        self.quantile_threshold = quantile_threshold
        self.number_possible_hyperparams_sample = number_possible_hyperparams_sample
        self.use_prior_normal_distribution = use_prior_normal_distribution
        self.use_linear_forgetting_weights = use_linear_forgetting_weights
        self.number_recent_trial_at_full_weights = number_recent_trial_at_full_weights

    def _split_trials(self, success_trials):
        # Split trials into good and bad using quantile threshold.
        # TODO: maybe better place in the Trials class.
        trials_scores = np.array([trial.score for trial in success_trials])

        # TODO: do we want to clip the number of good trials like in hyperopt.
        percentile_thresholds = np.percentile(trials_scores, self.quantile_threshold)

        # In hyperopt they use this to split, where default_gamma_cap = 25. They clip the max of item they use in the good item.
        # n_below = min(int(np.ceil(gamma * np.sqrt(len(l_vals)))), gamma_cap)

        good_trials = []
        bad_trials = []
        for trial in success_trials:
            if trial.score < percentile_thresholds:
                good_trials.append(trial)
            else:
                bad_trials.append(trial)
        return good_trials, bad_trials

    def _create_posterior(self, flat_hyperparameter_space, trials):

        # TODO: modify this to a class so we can rvs, pdf or cdf all togheter.

        # Create a list of all hyperparams and their trials.

        # Loop through all hyperparams
        posterior_distributions = {}
        for (hyperparam_key, hyperparam_distribution) in flat_hyperparameter_space.items():

            # Get distribution_trials
            distribution_trials = [trial.hyperparams.to_flat_as_dict_primitive()[hyperparam_key] for trial in trials]

            # TODO : create a discret and uniform class in order to be able to discretize them.
            if isinstance(hyperparam_distribution, discret_distribution):
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

    def _create_gaussian_mixture(self, hyperparam_distribution, distribution_trials)

        # TODO: add Condition if log distribution or not.
        use_logs = False

        # Find means, std, amplitudes, min and max.
        distribution_amplitudes, means, stds, distributions_mins, distributions_max = self._adaptive_parzen_normal(
            distribution_trials)

        # Create appropriate gaussian mixture that wrapped all hyperparams.
        gmm = DistributionMixture.build_gaussian_mixture(distribution_amplitudes, means=means, stds=stds,
                                                         distributions_mins=distributions_mins,
                                                         distributions_max=distributions_max, use_logs=use_logs)

        return gmm

    def _adaptive_parzen_normal(self, distribution_trials):

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

        # TODO: transfer to a hyperparms warappers.

        best_hyperparams = []
        for (hyperparam_key, good_posterior) in good_posteriors.items():
            best_new_hyperparam_value = None
            best_ratio = None
            for _ in range(self.number_possible_hyperparams_sample):
                # Sample possible new hyperparams in the good_trials.
                possible_new_hyperparm = good_posterior.rvs()

                # Verify ratio good pdf versus bad pdf for all possible new hyperparms.
                # Only the best ratio is kept and is the new best hyperparams.
                # TODO: verify if we use the ratio directly or we use the loglikelihood of b_post under both distribution like hyperopt.
                # In hyperopt they use :
                # # calculate the log likelihood of b_post under both distributions
                # below_llik = fn_lpdf(*([b_post] + b_post.pos_args), **b_kwargs)
                # above_llik = fn_lpdf(*([b_post] + a_post.pos_args), **a_kwargs)
                #
                # # improvement = below_llik - above_llik
                # # new_node = scope.broadcast_best(b_post, improvement)
                # new_node = scope.broadcast_best(b_post, below_llik, above_llik)
                ratio = bad_posteriors[hyperparam_key].pdf(possible_new_hyperparm) / good_posterior.pdf(
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
