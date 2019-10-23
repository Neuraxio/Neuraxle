# Note: some of the code in the present code block is derived from another project licensed under The MIT License (MIT),
# Copyright (c) 2017 Vooban Inc. For the full information, see:
#     https://github.com/guillaume-chevalier/Hyperopt-Keras-CNN-CIFAR-100/blob/Vooban/LICENSE

from neuraxle.hyperparams.distributions import *
from neuraxle.hyperparams.space import HyperparameterSpace
import matplotlib.pyplot as plt
import numpy as np

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

if __name__ == '__main__':
    discrete_hyperparameter_space = HyperparameterSpace({
        "randint": RandInt(1, 4)
    })
    plot_distribution_space(discrete_hyperparameter_space, num_bins=DISCRETE_NUM_BINS)