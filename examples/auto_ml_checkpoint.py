import shutil
import time

import numpy as np

from neuraxle.base import BaseStep, NonFittableMixin
from neuraxle.checkpoints import DefaultCheckpoint
from neuraxle.hyperparams.distributions import RandInt
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.random import RandomSearch
from neuraxle.pipeline import ResumablePipeline, DEFAULT_CACHE_FOLDER, Pipeline


class Multiplication(NonFittableMixin, BaseStep):
    def __init__(self, sleep_time=0.010, hyperparams=None, hyperparams_space=None):
        BaseStep.__init__(self, hyperparams=hyperparams, hyperparams_space=hyperparams_space)
        self.sleep_time = sleep_time

    def transform(self, data_inputs):
        time.sleep(self.sleep_time)
        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        return data_inputs * self.hyperparams['hp_mul']


def main():
    hyperparams_space = HyperparameterSpace(
        {
            'multiplication_2__hp_mul': RandInt(1, 3),
            'multiplication_3__hp_mul': RandInt(1, 3),
            'multiplication_4__hp_mul': RandInt(1, 3),
            'multiplication_5__hp_mul': RandInt(1, 3),
            'multiplication_6__hp_mul': RandInt(1, 3),
            'multiplication_7__hp_mul': RandInt(1, 3),
            'multiplication_8__hp_mul': RandInt(1, 3),
            'multiplication_9__hp_mul': RandInt(1, 3),
            'multiplication_10__hp_mul': RandInt(1, 3),
            'multiplication_11__hp_mul': RandInt(1, 3)
        }
    )

    pipeline = Pipeline([
        ('multiplication_2', Multiplication(sleep_time=0.0)),
        ('multiplication_3', Multiplication(sleep_time=0.0)),
        ('multiplication_4', Multiplication(sleep_time=0.0)),
        ('multiplication_5', Multiplication(sleep_time=0.0)),
        ('multiplication_6', Multiplication(sleep_time=0.0)),
        ('multiplication_7', Multiplication(sleep_time=0.0)),
        ('multiplication_8', Multiplication(sleep_time=0.0)),
        ('multiplication_9', Multiplication(sleep_time=0.0)),
        ('multiplication_10', Multiplication(sleep_time=0.0)),
        ('multiplication_11', Multiplication(sleep_time=0.0)),
    ])

    resumable_pipeline = ResumablePipeline([
        ('multiplication_2', Multiplication(sleep_time=0.0)),
        ('multiplication_3', Multiplication(sleep_time=0.0)),
        ('checkpoint_1', DefaultCheckpoint()),
        ('multiplication_4', Multiplication(sleep_time=0.0)),
        ('multiplication_5', Multiplication(sleep_time=0.0)),
        ('checkpoint_2', DefaultCheckpoint()),
        ('multiplication_6', Multiplication(sleep_time=0.0)),
        ('multiplication_7', Multiplication(sleep_time=0.0)),
        ('checkpoint_3', DefaultCheckpoint()),
        ('multiplication_8', Multiplication(sleep_time=0.0)),
        ('multiplication_9', Multiplication(sleep_time=0.0)),
        ('checkpoint_4', DefaultCheckpoint()),
        ('multiplication_10', Multiplication(sleep_time=0.0)),
        ('multiplication_11', Multiplication(sleep_time=0.0)),
    ])

    data_inputs = np.array(range(10))
    expected_outputs = np.array(range(10, 20))

    print('Classic Pipeline')
    run_random_search(pipeline, hyperparams_space, data_inputs, expected_outputs)

    print('\n')

    print('Resumable Pipeline')
    run_random_search(resumable_pipeline, hyperparams_space, data_inputs, expected_outputs)

    shutil.rmtree(DEFAULT_CACHE_FOLDER)


def run_random_search(p, hyperparams_space, data_inputs, expected_outputs):
    time_a = time.time()
    p.set_hyperparams_space(hyperparams_space)

    random_search = RandomSearch(
       p,
       n_iter=200,
       higher_score_is_better=True,
       print=True
    ).fit(data_inputs, expected_outputs)

    outputs = random_search.transform(data_inputs)

    time_b = time.time()

    actual_score = mean_squared_error(outputs, expected_outputs)
    print('{0} seconds'.format(time_b - time_a))
    print('output: {0}'.format(outputs))
    print('smallest mse: {0}'.format(actual_score))
    print('best hyperparams: {0}'.format(p.get_hyperparams()))


def mean_squared_error(actual_outputs, expected_outputs):
    mses = [(a - b) ** 2 for a, b in zip(actual_outputs, expected_outputs)]
    return sum(mses) / float(len(mses))


if __name__ == '__main__':
    main()