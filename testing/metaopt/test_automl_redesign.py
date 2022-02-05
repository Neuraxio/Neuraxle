import os

import numpy as np
from neuraxle.base import ExecutionContext as CX
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.distributions import (Boolean, Choice, LogUniform,
                                                RandInt)
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import AutoML, RandomSearchSampler
from neuraxle.metaopt.callbacks import (EarlyStoppingCallback, MetricCallback,
                                        ScoringCallback)
from neuraxle.metaopt.data.vanilla import VanillaHyperparamsRepository
from neuraxle.metaopt.validation import ValidationSplitter
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import NumpyRavel
from neuraxle.steps.output_handlers import OutputTransformerWrapper
from neuraxle.steps.sklearn import SKLearnWrapper
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


def _create_data_source():
    data_inputs = np.random.random((25, 50)).astype(np.float32)
    expected_outputs = (np.random.random((25,)) > 0.5).astype(np.int32)
    return data_inputs, expected_outputs


def _create_pipeline():
    return Pipeline([
        StandardScaler(),
        OutputTransformerWrapper(NumpyRavel()),
        SKLearnWrapper(
            LogisticRegression(),
            HyperparameterSpace({
                'C': LogUniform(0.01, 10.0),
                'fit_intercept': Boolean(),
                'penalty': Choice(['none', 'l2']),
                'max_iter': RandInt(20, 200)
            })
        )
    ])


def test_automl_api_entry_point(tmpdir):
    data_inputs, expected_outputs = _create_data_source()
    dact = DACT(data_inputs=data_inputs, expected_outputs=expected_outputs)
    pipeline = _create_pipeline()
    # TODO: # HyperbandControllerLoop(), ClusteringParallelFor() ?

    a: AutoML = AutoML(
        pipeline=pipeline,
        validation_splitter=ValidationSplitter(0.20),
        hyperparams_optimizer=RandomSearchSampler(),
        hyperparams_repository=VanillaHyperparamsRepository(cache_folder=os.path.join(tmpdir, "hp")),
        scoring_callback=ScoringCallback(mean_squared_error),
        callbacks=[
            MetricCallback('accuracy', metric_function=accuracy_score, higher_score_is_better=False),
            EarlyStoppingCallback(max_epochs_without_improvement=3)
        ],
        continue_loop_on_error=True,
        n_trials=17,
        epochs=11,
        refit_best_trial=True,
    )

    a, _out = a.handle_fit_transform(
        dact,
        CX(root=os.path.join(tmpdir, "automl"))
    )

    assert _out is not None
