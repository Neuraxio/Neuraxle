"""
Usage of Checkpoints in Automatic Machine Learning (AutoML)
=============================================================

This demonstrates how you can use checkpoints in a pipeline to save computing time when doing a hyperparameter search.

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

import time

import numpy as np
from sklearn.metrics import mean_squared_error

from neuraxle.checkpoints import DefaultCheckpoint
from neuraxle.hyperparams.distributions import RandInt
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import RandomSearch
from neuraxle.metaopt.random import ValidationSplitWrapper
from neuraxle.pipeline import ResumablePipeline, DEFAULT_CACHE_FOLDER, Pipeline
from neuraxle.steps.flow import ExpandDim
from neuraxle.steps.loop import ForEachDataInput
from neuraxle.steps.misc import Sleep
from neuraxle.steps.numpy import MultiplyByN


def main(tmpdir, sleep_time: float = 0, n_iter: int = 10):
    DATA_INPUTS = np.array(range(100))
    EXPECTED_OUTPUTS = np.array(range(100, 200))

    HYPERPARAMETER_SPACE = HyperparameterSpace({
        'multiplication_1__multiply_by': RandInt(1, 2),
        'multiplication_2__multiply_by': RandInt(1, 2),
        'multiplication_3__multiply_by': RandInt(1, 2),
    })

    print('Classic Pipeline:')

    pipeline = Pipeline([
        ('multiplication_1', MultiplyByN()),
        ('sleep_1', ForEachDataInput(Sleep(sleep_time))),
        ('multiplication_2', MultiplyByN()),
        ('sleep_2', ForEachDataInput(Sleep(sleep_time))),
        ('multiplication_3', MultiplyByN()),
    ]).set_hyperparams_space(HYPERPARAMETER_SPACE)

    time_a = time.time()
    random_search = RandomSearch(
        ValidationSplitWrapper(pipeline, test_size=0.1),
        n_iter=n_iter,
        higher_score_is_better=True,
        cache_folder_when_no_handle=tmpdir
    ).fit(DATA_INPUTS, EXPECTED_OUTPUTS)
    outputs = random_search.get_best_model().transform(DATA_INPUTS)
    time_b = time.time()

    actual_score = mean_squared_error(EXPECTED_OUTPUTS, outputs)
    print('{0} seconds'.format(time_b - time_a))
    print('output: {0}'.format(outputs))
    print('smallest mse: {0}'.format(actual_score))
    print('best hyperparams: {0}'.format(pipeline.get_hyperparams()))

    assert isinstance(actual_score, float)

    print('Resumable Pipeline:')

    pipeline = ResumablePipeline([
        ('multiplication_1', MultiplyByN()),
        ('ForEach(sleep_1)', ForEachDataInput(Sleep(sleep_time))),
        ('checkpoint1', ExpandDim(DefaultCheckpoint())),
        ('multiplication_2', MultiplyByN()),
        ('sleep_2', ForEachDataInput(Sleep(sleep_time))),
        ('checkpoint2', ExpandDim(DefaultCheckpoint())),
        ('multiplication_3', MultiplyByN())
    ], cache_folder=tmpdir).set_hyperparams_space(HYPERPARAMETER_SPACE)

    time_a = time.time()
    random_search = RandomSearch(
        ValidationSplitWrapper(pipeline, test_size=0.1),
        n_iter=n_iter,
        higher_score_is_better=True,
        cache_folder_when_no_handle=tmpdir
    ).fit(DATA_INPUTS, EXPECTED_OUTPUTS)
    outputs = random_search.get_best_model().transform(DATA_INPUTS)
    time_b = time.time()
    pipeline.flush_all_cache()

    actual_score = mean_squared_error(EXPECTED_OUTPUTS, outputs)
    print('{0} seconds'.format(time_b - time_a))
    print('output: {0}'.format(outputs))
    print('smallest mse: {0}'.format(actual_score))
    print('best hyperparams: {0}'.format(pipeline.get_hyperparams()))

    assert isinstance(actual_score, float)


if __name__ == "__main__":
    main(DEFAULT_CACHE_FOLDER, sleep_time=0.001, n_iter=50)
