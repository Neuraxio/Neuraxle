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
import os
import time

import numpy as np
from sklearn.metrics import mean_squared_error
from neuraxle.base import ExecutionContext

from neuraxle.hyperparams.distributions import RandInt
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import AutoML, RandomSearchHyperparameterSelectionStrategy, ValidationSplitter
from neuraxle.metaopt.callbacks import MetricCallback, ScoringCallback
from neuraxle.pipeline import Pipeline
from neuraxle.steps.flow import ExpandDim
from neuraxle.steps.loop import ForEach
from neuraxle.steps.misc import Sleep
from neuraxle.steps.numpy import MultiplyByN


def main(tmpdir, sleep_time: float = 0.001, n_iter: int = 10):
    DATA_INPUTS = np.array(range(100))
    EXPECTED_OUTPUTS = np.array(range(100, 200))

    HYPERPARAMETER_SPACE = HyperparameterSpace({
        'multiplication_1__multiply_by': RandInt(1, 2),
        'multiplication_2__multiply_by': RandInt(1, 2),
    })

    print('Classic Pipeline:')
    classic_pipeline_folder = os.path.join(str(tmpdir), 'classic')

    pipeline = Pipeline([
        ('multiplication_1', MultiplyByN()),
        ('sleep_1', ForEach(Sleep(sleep_time))),
        ('multiplication_2', MultiplyByN()),
    ], cache_folder=classic_pipeline_folder).set_hyperparams_space(HYPERPARAMETER_SPACE)

    time_a = time.time()
    auto_ml = AutoML(
        pipeline,
        refit_trial=True,
        n_trials=n_iter,
        cache_folder_when_no_handle=classic_pipeline_folder,
        validation_splitter=ValidationSplitter(0.2),
        hyperparams_optimizer=RandomSearchHyperparameterSelectionStrategy(),
        scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
        callbacks=[
            MetricCallback('mse', metric_function=mean_squared_error, higher_score_is_better=False)
        ],
    )
    auto_ml = auto_ml.fit(DATA_INPUTS, EXPECTED_OUTPUTS)
    outputs = auto_ml.get_best_model().predict(DATA_INPUTS)
    time_b = time.time()

    actual_score = mean_squared_error(EXPECTED_OUTPUTS, outputs)
    print('{0} seconds'.format(time_b - time_a))
    print('smallest mse: {0}'.format(actual_score))
    print('best hyperparams: {0}'.format(pipeline.get_hyperparams()))

    assert isinstance(actual_score, float)

    print('Resumable Pipeline:')
    resumable_pipeline_folder = os.path.join(str(tmpdir), 'resumable')

    pipeline = ResumablePipeline([
        ('multiplication_1', MultiplyByN()),
        ('sleep_1', ForEach(Sleep(sleep_time))),
        ('checkpoint1', ExpandDim(DefaultCheckpoint())),
        ('multiplication_2', MultiplyByN()),
    ], cache_folder=resumable_pipeline_folder).set_hyperparams_space(HYPERPARAMETER_SPACE)

    time_a = time.time()
    auto_ml = AutoML(
        pipeline,
        refit_trial=True,
        n_trials=n_iter,
        cache_folder_when_no_handle=resumable_pipeline_folder,
        validation_splitter=ValidationSplitter(0.2),
        hyperparams_optimizer=RandomSearchHyperparameterSelectionStrategy(),
        scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
        callbacks=[
            MetricCallback('mse', metric_function=mean_squared_error, higher_score_is_better=False)
        ]
    )
    auto_ml = auto_ml.fit(DATA_INPUTS, EXPECTED_OUTPUTS)
    outputs2 = auto_ml.get_best_model().predict(DATA_INPUTS)
    time_b = time.time()
    pipeline.flush_all_cache()

    actual_score = mean_squared_error(EXPECTED_OUTPUTS, outputs2)
    print('{0} seconds'.format(time_b - time_a))
    print('smallest mse: {0}'.format(actual_score))
    print('best hyperparams: {0}'.format(pipeline.get_hyperparams()))

    assert isinstance(actual_score, float)
    assert (outputs == outputs2).all()


if __name__ == "__main__":
    main(ExecutionContext.get_new_cache_folder(), sleep_time=0.005, n_iter=10)
