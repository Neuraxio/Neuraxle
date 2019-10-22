import os
import time

import numpy as np

from neuraxle.checkpoints import DefaultCheckpoint
from neuraxle.hyperparams.distributions import RandInt
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.random import RandomSearch
from neuraxle.pipeline import ResumablePipeline, DEFAULT_CACHE_FOLDER, Pipeline
from neuraxle.steps.misc import Sleep
from neuraxle.steps.numpy import MultiplyByN


def main(sleep_time):
    EXPECTED_OUTPUTS = np.array(range(10, 20))

    DATA_INPUTS = np.array(range(10))

    HYPERPARAMETER_SPACE = HyperparameterSpace({
        'multiplication_2__multiply_by': RandInt(1, 3),
        'multiplication_3__multiply_by': RandInt(1, 3),
        'multiplication_4__multiply_by': RandInt(1, 3),
        'multiplication_5__multiply_by': RandInt(1, 3),
        'multiplication_6__multiply_by': RandInt(1, 3),
        'multiplication_7__multiply_by': RandInt(1, 3),
        'multiplication_8__multiply_by': RandInt(1, 3),
        'multiplication_9__multiply_by': RandInt(1, 3),
        'multiplication_10__multiply_by': RandInt(1, 3),
        'multiplication_11__multiply_by': RandInt(1, 3)
    })

    def mean_squared_error(actual_outputs, expected_outputs):
        mses = [(a - b) ** 2 for a, b in zip(actual_outputs, expected_outputs)]
        return np.mean(mses)

    pipeline = Pipeline([
        ('multiplication_2', MultiplyByN(1)),
        ('multiplication_3', MultiplyByN(1)),
        ('sleep_1', Sleep(sleep_time)),
        ('multiplication_4', MultiplyByN(1)),
        ('multiplication_5', MultiplyByN(1)),
        ('sleep_2', Sleep(sleep_time)),
        ('multiplication_6', MultiplyByN(1)),
        ('multiplication_7', MultiplyByN(1)),
        ('sleep_3', Sleep(sleep_time)),
        ('multiplication_8', MultiplyByN(1)),
        ('multiplication_9', MultiplyByN(1)),
        ('sleep_4', Sleep(sleep_time)),
        ('multiplication_10',MultiplyByN(1)),
        ('multiplication_11', MultiplyByN(1)),
    ]).set_hyperparams_space(HYPERPARAMETER_SPACE)

    print('Classic Pipeline')

    time_a = time.time()

    best_model = RandomSearch(
        pipeline,
        n_iter=200,
        higher_score_is_better=True
    ).fit(DATA_INPUTS, EXPECTED_OUTPUTS) \
        .get_best_model()

    outputs = best_model.transform(DATA_INPUTS)

    time_b = time.time()

    actual_score = mean_squared_error(outputs, EXPECTED_OUTPUTS)
    print('{0} seconds'.format(time_b - time_a))
    print('output: {0}'.format(outputs))
    print('smallest mse: {0}'.format(actual_score))
    print('best hyperparams: {0}'.format(pipeline.get_hyperparams()))

    assert isinstance(actual_score, float)
    assert isinstance(outputs, np.ndarray)

    pipeline = ResumablePipeline([
        ('multiplication_2', MultiplyByN(1)),
        ('multiplication_3', MultiplyByN(1)),
        ('sleep_1', Sleep(sleep_time)),
        ('checkpoint_1', DefaultCheckpoint()),
        ('multiplication_4', MultiplyByN(1)),
        ('multiplication_5', MultiplyByN(1)),
        ('sleep_2', Sleep(sleep_time)),
        ('checkpoint_2', DefaultCheckpoint()),
        ('multiplication_6', MultiplyByN(1)),
        ('multiplication_7', MultiplyByN(1)),
        ('sleep_3', Sleep(sleep_time)),
        ('checkpoint_3', DefaultCheckpoint()),
        ('multiplication_8', MultiplyByN(1)),
        ('multiplication_9', MultiplyByN(1)),
        ('sleep_4', Sleep(sleep_time)),
        ('checkpoint_4', DefaultCheckpoint()),
        ('multiplication_10', MultiplyByN(1)),
        ('multiplication_11', MultiplyByN(1)),
    ], cache_folder=DEFAULT_CACHE_FOLDER).set_hyperparams_space(HYPERPARAMETER_SPACE)

    print('Resumable Pipeline')

    time_a = time.time()

    pipeline.set_hyperparams_space(HYPERPARAMETER_SPACE)

    best_model = RandomSearch(
        pipeline,
        n_iter=200,
        higher_score_is_better=True
    ).fit(DATA_INPUTS, EXPECTED_OUTPUTS) \
     .get_best_model()

    outputs = best_model.transform(DATA_INPUTS)

    time_b = time.time()

    actual_score = mean_squared_error(outputs, EXPECTED_OUTPUTS)
    print('{0} seconds'.format(time_b - time_a))
    print('output: {0}'.format(outputs))
    print('smallest mse: {0}'.format(actual_score))
    print('best hyperparams: {0}'.format(pipeline.get_hyperparams()))

    assert isinstance(actual_score, float)
    assert isinstance(outputs, np.ndarray)

    pipeline.flush_all_cache()

if __name__ == "__main__":
    main(0.1)