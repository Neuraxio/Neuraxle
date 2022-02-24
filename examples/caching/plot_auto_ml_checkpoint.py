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

from neuraxle.base import ExecutionContext as CX
from neuraxle.data_container import DataContainer as DACT
from neuraxle.data_container import TrainDACT, PredsDACT
from neuraxle.hyperparams.distributions import RandInt
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import AutoML, ValidationSplitter
from neuraxle.metaopt.callbacks import MetricCallback
from neuraxle.pipeline import Pipeline
from neuraxle.steps.loop import ForEach
from neuraxle.steps.misc import Sleep
from neuraxle.steps.numpy import MultiplyByN
from neuraxle.steps.caching import JoblibTransformDIValueCachingWrapper


def main(tmpdir, sleep_time: float = 0.001, n_iter: int = 10):
    DATA_CONTAINER: TrainDACT = DACT(data_inputs=np.array(range(100)), expected_outputs=np.array(range(100, 200)))

    HYPERPARAMETER_SPACE = HyperparameterSpace({
        'multiplication_1__multiply_by': RandInt(1, 2),
        'multiplication_2__multiply_by': RandInt(1, 2),
    })

    print('Classic Pipeline:')
    classic_pipeline_folder = os.path.join(str(tmpdir), 'classic')
    classic_pipeline_context = CX(root=classic_pipeline_folder)

    pipeline = Pipeline([
        ('multiplication_1', MultiplyByN()),
        ('sleep_1', ForEach(Sleep(sleep_time))),
        ('multiplication_2', MultiplyByN()),
    ]).set_hyperparams_space(HYPERPARAMETER_SPACE)

    time_a = time.time()
    auto_ml = AutoML(
        pipeline,
        refit_best_trial=True,
        n_trials=n_iter,
        validation_splitter=ValidationSplitter(0.2),
        callbacks=[MetricCallback('mse', metric_function=mean_squared_error, higher_score_is_better=False)],
    )
    auto_ml = auto_ml.handle_fit(DATA_CONTAINER, classic_pipeline_context)
    outputs: PredsDACT = auto_ml.handle_predict(DATA_CONTAINER.without_eo(), classic_pipeline_context)
    time_b = time.time()

    actual_score = mean_squared_error(DATA_CONTAINER.eo, outputs.di)
    print('{0} seconds'.format(time_b - time_a))
    print('smallest mse: {0}'.format(actual_score))
    print('best hyperparams: {0}'.format(pipeline.get_hyperparams()))

    assert isinstance(actual_score, float)

    print('Resumable Pipeline:')
    caching_pipeline_folder = os.path.join(str(tmpdir), 'cache')
    caching_pipeline_context = CX(root=caching_pipeline_folder)

    pipeline: Pipeline = Pipeline([
        ('multiplication_1', MultiplyByN()),
        ("value_checkpoints", JoblibTransformDIValueCachingWrapper(Pipeline([
            ('sleep_1', ForEach(Sleep(sleep_time))),
            ('multiplication_2', MultiplyByN()),
        ]))),
    ]).set_hyperparams_space(HYPERPARAMETER_SPACE)

    time_a = time.time()
    auto_ml = AutoML(
        pipeline,
        refit_best_trial=True,
        n_trials=n_iter,
        validation_splitter=ValidationSplitter(0.2),
        callbacks=[MetricCallback('mse', metric_function=mean_squared_error, higher_score_is_better=False)]
    )
    auto_ml = auto_ml.handle_fit(DATA_CONTAINER, caching_pipeline_context)
    outputs2: PredsDACT = auto_ml.handle_predict(DATA_CONTAINER.without_eo(), caching_pipeline_context)
    time_b = time.time()

    actual_score = mean_squared_error(DATA_CONTAINER.eo, outputs2.di)
    print('{0} seconds'.format(time_b - time_a))
    print('smallest mse: {0}'.format(actual_score))
    print('best hyperparams: {0}'.format(pipeline.get_hyperparams()))

    assert isinstance(actual_score, float)
    assert (outputs == outputs2).all()


if __name__ == "__main__":
    main(CX.get_new_cache_folder(), sleep_time=0.005, n_iter=10)
