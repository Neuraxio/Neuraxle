import shutil
import time

import numpy as np

from neuraxle.base import BaseStep, NonFittableMixin
from neuraxle.checkpoints import DefaultCheckpoint
from neuraxle.hyperparams.distributions import RandInt
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace
from neuraxle.pipeline import ResumablePipeline, DEFAULT_CACHE_FOLDER, Pipeline


class Multiplication(NonFittableMixin, BaseStep):
    def __init__(self, sleep_time=0.050, hyperparams=None, hyperparams_space=None):
        BaseStep.__init__(self, hyperparams=hyperparams, hyperparams_space=hyperparams_space)
        self.sleep_time = sleep_time

    def transform(self, data_inputs):
        time.sleep(self.sleep_time)
        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        return data_inputs * self.hyperparams['hp_mul']


class AutoMLSequentialWrapper:
    def __init__(
            self,
            pipeline: Pipeline,
            hyperparameters_space: HyperparameterSpace,
            objective_function,
            n_iters=100
    ):
        self.objective_function = objective_function
        self.hyperparameters_space = hyperparameters_space
        self.pipeline = pipeline
        self.n_iters = n_iters

    def fit(self, data_inputs, expected_outputs) -> Pipeline:
        best_score = None
        best_hp = None
        for _ in range(self.n_iters):
            next_hp: HyperparameterSamples = self.hyperparameters_space.rvs()
            self.pipeline.set_hyperparams(next_hp)

            self.pipeline, actual_outputs = self.pipeline.fit_transform(data_inputs, expected_outputs)

            score = self.objective_function(actual_outputs, expected_outputs)

            if best_score is None or score < best_score:
                best_score = score
                best_hp = next_hp
                print('score: {0}'.format(score))
                print('best hp: {0}'.format(best_hp))
                print('outputs: {0}'.format(actual_outputs))
                print('\n')

        fitted_pipeline = self.pipeline.set_hyperparams(best_hp)
        return fitted_pipeline


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
        ('multiplication_2', Multiplication(sleep_time=0.05)),
        ('multiplication_3', Multiplication(sleep_time=0.05)),
        ('multiplication_4', Multiplication(sleep_time=0.05)),
        ('multiplication_5', Multiplication(sleep_time=0.05)),
        ('multiplication_6', Multiplication(sleep_time=0.05)),
        ('multiplication_7', Multiplication(sleep_time=0.05)),
        ('multiplication_8', Multiplication(sleep_time=0.05)),
        ('multiplication_9', Multiplication(sleep_time=0.05)),
        ('multiplication_10', Multiplication(sleep_time=0.05)),
        ('multiplication_11', Multiplication(sleep_time=0.05)),
    ])

    resumable_pipeline = ResumablePipeline([
        ('multiplication_2', Multiplication(sleep_time=0.05)),
        ('multiplication_3', Multiplication(sleep_time=0.05)),
        ('checkpoint_1', DefaultCheckpoint()),
        ('multiplication_4', Multiplication(sleep_time=0.05)),
        ('multiplication_5', Multiplication(sleep_time=0.05)),
        ('checkpoint_2', DefaultCheckpoint()),
        ('multiplication_6', Multiplication(sleep_time=0.05)),
        ('multiplication_7', Multiplication(sleep_time=0.05)),
        ('checkpoint_3', DefaultCheckpoint()),
        ('multiplication_8', Multiplication(sleep_time=0.05)),
        ('multiplication_9', Multiplication(sleep_time=0.05)),
        ('checkpoint_4', DefaultCheckpoint()),
        ('multiplication_10', Multiplication(sleep_time=0.05)),
        ('multiplication_11', Multiplication(sleep_time=0.05)),
    ])

    data_inputs = np.array(range(10))
    expected_outputs = np.array(range(10, 20))

    print('Classic Pipeline')
    run_pipeline(pipeline, hyperparams_space, data_inputs, expected_outputs)

    print('\n')

    print('Resumable Pipeline')
    run_pipeline(resumable_pipeline, hyperparams_space, data_inputs, expected_outputs)

    shutil.rmtree(DEFAULT_CACHE_FOLDER)


def run_pipeline(p, hyperparams_space, data_inputs, expected_outputs):
    time_a = time.time()

    p = AutoMLSequentialWrapper(
        pipeline=p,
        n_iters=200,
        hyperparameters_space=hyperparams_space,
        objective_function=mean_squared_error
    ).fit(data_inputs, expected_outputs)

    p, outputs = p.fit_transform(data_inputs, expected_outputs)

    time_b = time.time()

    actual_score = mean_squared_error(outputs, expected_outputs)
    print('{0} seconds'.format(time_b - time_a))
    print('output: {0}'.format(outputs))
    print('smallest error: {0}'.format(actual_score))
    print('best hyperparams: {0}'.format(p.get_hyperparams()))


def mean_squared_error(actual_outputs, expected_outputs):
    mses = [(a - b) ** 2 for a, b in zip(actual_outputs, expected_outputs)]
    return sum(mses) / float(len(mses))


if __name__ == '__main__':
    main()