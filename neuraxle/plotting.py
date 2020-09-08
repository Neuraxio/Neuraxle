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
from neuraxle.metaopt.trial import Trial, TRIAL_STATUS

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
    def __init__(
            self,
            metric_name: str,
            plotting_folder_name: str = 'metric_results',
            save_plots: bool = True,
            plot_on_next: bool = True,
            plot_on_complete: bool = True
    ):
        self.plot_on_next: bool = plot_on_next
        self.plot_on_complete: bool = plot_on_complete
        self.plotting_folder_name: str = plotting_folder_name
        self.save: bool = save_plots
        self.metric_name: str = metric_name

    def on_next(self, value: Tuple[HyperparamsRepository, Trial]):
        repo, trial = value
        trial_hash = repo._get_trial_hash(trial)

        for split in trial.validation_splits:
            for metric_name in split.get_metric_names():
                train_results = split.get_metric_train_results(metric_name=metric_name)
                validation_results = split.get_metric_validation_results(metric_name=metric_name)
                plt.plot(train_results)
                plt.plot(validation_results)
                plt.ylabel(metric_name)
                plt.xlabel('epoch')
                plt.legend(['train', 'validation'], loc='upper left')
                plotting_file = os.path.join(trial.cache_folder, trial_hash, '{}.png'.format(metric_name))
                self._show_or_save_plot(plotting_file)

    def on_complete(self, value: Tuple[HyperparamsRepository, Trial]):
        repo, trial = value
        trials = repo.load_all_trials(TRIAL_STATUS.SUCCESS)
        if len(trials) == 0:
            return

        n_splits = trials.get_number_of_split()
        metric_names = trials.get_metric_names()

        for metric_name in metric_names:
            for split_number in range(n_splits):
                for trial in trials:
                    validation_results = trial[split_number].get_metric_validation_results(metric_name=metric_name)
                    plt.plot(validation_results)

                plt.ylabel(metric_name)
                plt.xlabel('epoch')

                metric_file_name = '{}_{}.png'.format(metric_name, split_number)
                plotting_file = os.path.join(trial.cache_folder, self.plotting_folder_name, metric_file_name)
                self._show_or_save_plot(plotting_file)

    def _show_or_save_plot(self, plotting_file):
        if self.save:
            plt.savefig(plotting_file)
        else:
            plt.show()
        plt.close()
