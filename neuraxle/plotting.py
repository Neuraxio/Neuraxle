"""
Notebook matplotlib plotting functions
================================================

Utility function for plotting in notebooks.

..
    Note: some of the code in the present code block is derived from another project licensed under The MIT License (MIT),
    Copyright (c) 2017 Vooban Inc. For the full information, see:
    https://github.com/guillaume-chevalier/Hyperopt-Keras-CNN-CIFAR-100/blob/Vooban/LICENSE

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
import matplotlib.pyplot as plt
import os
from neuraxle.metaopt.auto_ml import HyperparamsRepository

from neuraxle.hyperparams.distributions import *
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.observable import _Observer, T
from neuraxle.metaopt.trial import Trial, TRIAL_STATUS, Trials

DISCRETE_NUM_BINS = 40
CONTINUOUS_NUM_BINS = 1000
NUM_TRIALS = 100000
X_DOMAIN = np.array(range(-100, 600)) / 100


def plot_histogram(title: str, distribution: HyperparameterDistribution, num_bins=50):
    samples = np.array([
        distribution.rvs() for _ in range(NUM_TRIALS)
    ], dtype=np.float).flatten()

    pdf_values = []
    for value in X_DOMAIN:
        try:
            pdf_values.append(distribution.pdf(value))
        except ValueError:
            pdf_values.append(0.)

    pdf_values = np.array(pdf_values)

    plt.figure(figsize=(17, 5))
    ax = plt.gca()
    hist = plt.hist(samples, bins=num_bins, label="hist")
    renormalization_const = hist[0].max() / pdf_values.max()
    renormalized_pdf_values = renormalization_const * pdf_values

    plt.plot(X_DOMAIN, renormalized_pdf_values, label="renorm pdf")
    plt.title("Histogram (pdf) for a {} distribution".format(title))
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels)
    plt.xlim([X_DOMAIN.min(), X_DOMAIN.max()])
    plt.show()


def plot_pdf_cdf(title: str, distribution: HyperparameterDistribution):
    pdf_values = []
    cdf_values = []
    for value in X_DOMAIN:
        try:
            pdf_values.append(distribution.pdf(value))
        except ValueError:
            pdf_values.append(0.)

        try:
            cdf_values.append(distribution.cdf(value))
        except ValueError:
            cdf_values.append(0.)

    pdf_values = np.array(pdf_values)
    cdf_values = np.array(cdf_values)
    plt.figure(figsize=(17, 5))
    ax = plt.gca()
    plt.plot(X_DOMAIN, pdf_values, label="pdf")
    plt.plot(X_DOMAIN, cdf_values, label="cdf")
    plt.title("Pdf and cdf for a {} distribution".format(title))
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels)
    plt.xlim([X_DOMAIN.min(), X_DOMAIN.max()])
    plt.show()


def plot_distribution_space(hyperparameter_space: HyperparameterSpace, num_bins=50):
    for title, distribution in hyperparameter_space.items():
        print(title + ":")
        plot_histogram(title, distribution, num_bins=num_bins)
        plot_pdf_cdf(title, distribution)


class TrialMetricsPlottingObserver(_Observer[Tuple[HyperparamsRepository, Trial]]):
    """
    An observer that receives trial updates and plots metric results.
    It can plot individual trials on each update, or upon completion.
    It can also plot all trials in the same plot upon completion.

    Usage Example:

    .. code-block:: python

        hyperparams_repository: HyperparamsJSONRepository = HyperparamsJSONRepository(cache_folder='trials')
        hyperparams_repository.subscribe(TrialMetricsPlottingObserver(
            plotting_folder_name: str = 'metric_results',
            plot_individual_trials_on_complete=False,
            plot_trial_on_next=True,
            plot_all_trials_on_complete=False,
            save_plots=True
        ))

        auto_ml = AutoML(
            pipeline,
            n_trials=n_iter,
            validation_split_function=validation_splitter(0.2),
            hyperparams_optimizer=RandomSearchHyperparameterSelectionStrategy(),
            scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
            callbacks=[
                MetricCallback('mse', metric_function=mean_squared_error, higher_score_is_better=False)
            ],
            refit_trial=True,
            cache_folder_when_no_handle=str(tmpdir)
        )

        auto_ml = auto_ml.fit(data_inputs, expected_outputs)

    .. seealso::
        :class:`~neuraxle.metaopt._Observer`,
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`~neuraxle.metaopt.trial.Trials`,
        :class:`~neuraxle.metaopt.auto_ml.AutoML`,
        :class:`HyperparamsRepository`,
        :class:`HyperparamsJSONRepository`
    """
    def __init__(
            self,
            plotting_folder_name: str = 'metric_results',
            save_plots: bool = True,
            plot_trial_on_next: bool = True,
            plot_all_trials_on_complete: bool = True,
            plot_individual_trials_on_complete: bool = True
    ):
        self.plot_individual_trials_on_complete = plot_individual_trials_on_complete
        self.plot_trial_on_next: bool = plot_trial_on_next
        self.plot_all_trials_on_complete: bool = plot_all_trials_on_complete
        self.plotting_folder_name: str = plotting_folder_name
        self.save: bool = save_plots

    def on_next(self, value: Tuple[HyperparamsRepository, Trial]):
        """
        Plot updated trial metric results.

        :param value: hyperparams_repository, trial
        :return:
        """
        repo, trial = value
        if not self.plot_trial_on_next:
            return

        self._plot_all_trial_main_and_validation_metric_results(repo, trial)

    def _plot_all_trial_main_and_validation_metric_results(self, repo, trial):
        trial_hash = repo._get_trial_hash(trial)
        for split_number, split in enumerate(trial.validation_splits):
            for metric_name in split.get_metric_names():
                train_results = split.get_metric_train_results(metric_name=metric_name)
                validation_results = split.get_metric_validation_results(metric_name=metric_name)
                plt.plot(train_results)
                plt.plot(validation_results)
                plt.ylabel(metric_name)
                plt.xlabel('epoch')
                plt.legend(['train', 'validation'], loc='upper left')
                plt.title(metric_name)

                plotting_folder = os.path.join(repo.cache_folder, trial_hash, str(split_number))
                if not os.path.exists(plotting_folder):
                    os.makedirs(plotting_folder)
                plotting_file = os.path.join(plotting_folder, '{}.png'.format(metric_name))

                self._show_or_save_plot(plotting_file)

    def on_complete(self, value: Tuple[HyperparamsRepository, Trial]):
        """
        Plot trial metric results upon completion.

        :param value: hyperparams_repository, trial
        :return:
        """
        repo, trial = value
        trials: Trials = repo.load_all_trials(TRIAL_STATUS.SUCCESS)
        if not self.plot_all_trials_on_complete:
            return
        if len(trials) == 0:
            return

        if self.plot_individual_trials_on_complete:
            for trial in trials:
                self._plot_all_trial_main_and_validation_metric_results(repo, trial)

        if self.plot_all_trials_on_complete:
            self._plot_all_trials_on_complete(repo, trials)

    def _plot_all_trials_on_complete(self, repo, trials):
        n_splits = trials.get_number_of_split()
        metric_names = trials.get_metric_names()
        for metric_name in metric_names:
            for split_number in range(n_splits):
                self._plot_all_trials_training_results_for_metric(
                    trials=trials,
                    metric_name=metric_name,
                    cache_folder=repo.cache_folder,
                    split_number=split_number
                )
                self._plot_all_trials_validation_results_for_metric(
                    trials=trials,
                    metric_name=metric_name,
                    cache_folder=repo.cache_folder,
                    split_number=split_number
                )

    def _plot_all_trials_validation_results_for_metric(self, trials, metric_name, cache_folder, split_number):
        for trial in trials:
            validation_results = trial[split_number].get_metric_validation_results(metric_name=metric_name)
            plt.plot(validation_results)

        plt.ylabel(metric_name)
        plt.xlabel('epoch')
        plt.title('Validation {}'.format(metric_name))
        metric_file_name = '{}_{}_validation.png'.format(metric_name, split_number)
        plotting_folder = os.path.join(cache_folder, self.plotting_folder_name, str(split_number))
        if not os.path.exists(plotting_folder):
            os.makedirs(plotting_folder)
        plotting_file = os.path.join(plotting_folder, metric_file_name)

        self._show_or_save_plot(plotting_file)

    def _plot_all_trials_training_results_for_metric(self, trials, metric_name, cache_folder, split_number):
        for trial in trials:
            validation_results = trial[split_number].get_metric_train_results(metric_name=metric_name)
            plt.plot(validation_results, linewidth=0.5)

        plt.ylabel(metric_name)
        plt.xlabel('epoch')
        plt.title('Train {}'.format(metric_name))
        metric_file_name = '{}_{}_train.png'.format(metric_name, split_number)
        plotting_folder = os.path.join(cache_folder, self.plotting_folder_name, str(split_number))
        if not os.path.exists(plotting_folder):
            os.makedirs(plotting_folder)
        plotting_file = os.path.join(plotting_folder, metric_file_name)

        self._show_or_save_plot(plotting_file)

    def _show_or_save_plot(self, plotting_file):
        if self.save:
            plt.savefig(plotting_file)
        else:
            plt.show()
        plt.close()
