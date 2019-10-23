import time

import numpy as np
from sklearn.metrics import mean_squared_error

from neuraxle.checkpoints import DefaultCheckpoint
from neuraxle.hyperparams.distributions import RandInt
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.random import RandomSearch
from neuraxle.pipeline import ResumablePipeline, DEFAULT_CACHE_FOLDER, Pipeline
from neuraxle.steps.loop import ForEachDataInput
from neuraxle.steps.misc import Sleep
from neuraxle.steps.numpy import MultiplyByN


def main(sleep_time, n_iter=200):
    DATA_INPUTS = np.array(range(10))
    EXPECTED_OUTPUTS = np.array(range(10, 20))

    HYPERPARAMETER_SPACE = HyperparameterSpace({
        'multiplication_1__multiply_by': RandInt(1, 2),
        'multiplication_2__multiply_by': RandInt(1, 2),
        'multiplication_3__multiply_by': RandInt(1, 2),
    })

    print('Classic Pipeline:')

    pipeline = Pipeline([
        ('multiplication_1', MultiplyByN(1)),
        ('sleep_1', ForEachDataInput(Sleep(sleep_time))),
        ('multiplication_2', MultiplyByN(1)),
        ('sleep_2', ForEachDataInput(Sleep(sleep_time))),
        ('multiplication_3', MultiplyByN(1)),
    ]).set_hyperparams_space(HYPERPARAMETER_SPACE)


    time_a = time.time()
    best_model = RandomSearch(
        pipeline,
        n_iter=n_iter,
        higher_score_is_better=True
    ).fit(DATA_INPUTS, EXPECTED_OUTPUTS)
    outputs = best_model.transform(DATA_INPUTS)
    time_b = time.time()

    actual_score = mean_squared_error(EXPECTED_OUTPUTS, outputs)
    print('{0} seconds'.format(time_b - time_a))
    print('output: {0}'.format(outputs))
    print('smallest mse: {0}'.format(actual_score))
    print('best hyperparams: {0}'.format(pipeline.get_hyperparams()))

    assert isinstance(actual_score, float)
    assert isinstance(outputs, np.ndarray)

    print('Resumable Pipeline:')

    pipeline = ResumablePipeline([
        ('multiplication_1', MultiplyByN(1)),
        ('sleep_1', ForEachDataInput(Sleep(sleep_time))),
        DefaultCheckpoint(),
        ('multiplication_2', MultiplyByN(1)),
        ('sleep_2', ForEachDataInput(Sleep(sleep_time))),
        DefaultCheckpoint(),
        ('multiplication_3', MultiplyByN(1))
    ], cache_folder=DEFAULT_CACHE_FOLDER).set_hyperparams_space(HYPERPARAMETER_SPACE)


    time_a = time.time()
    best_model = RandomSearch(
        pipeline,
        n_iter=n_iter,
        higher_score_is_better=True
    ).fit(DATA_INPUTS, EXPECTED_OUTPUTS)
    outputs = best_model.transform(DATA_INPUTS)
    time_b = time.time()
    pipeline.flush_all_cache()

    actual_score = mean_squared_error(EXPECTED_OUTPUTS, outputs)
    print('{0} seconds'.format(time_b - time_a))
    print('output: {0}'.format(outputs))
    print('smallest mse: {0}'.format(actual_score))
    print('best hyperparams: {0}'.format(pipeline.get_hyperparams()))

    assert isinstance(actual_score, float)
    assert isinstance(outputs, np.ndarray)


if __name__ == "__main__":
    main(sleep_time=0.01, n_iter=30)
