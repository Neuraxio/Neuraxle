import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from neuraxle.base import ExecutionContext
from neuraxle.data_container import DataContainer
from neuraxle.metaopt.auto_ml import AutoML, DefaultLoop, HyperparamsJSONRepository, ValidationSplitter, Trainer, RandomSearchHyperparameterSelectionStrategy
from neuraxle.metaopt.callbacks import MetricCallback
from neuraxle.pipeline import Pipeline
from neuraxle.hyperparams.distributions import Choice, RandInt, Boolean, LogUniform
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.steps.numpy import NumpyRavel
from neuraxle.steps.output_handlers import OutputTransformerWrapper
from neuraxle.steps.sklearn import SKLearnWrapper


def _create_data_source():
    data_inputs = np.random.random((25, 50)).astype(np.float32)
    expected_outputs = np.random.random((25,)).astype(np.float32)
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
    dact = DataContainer(data_inputs=data_inputs, expected_outputs=expected_outputs)
    pipeline = _create_pipeline()
    # TODO: # HyperbandControllerLoop(), ClusteringParallelFor() ?

    a: AutoML = AutoML(
        pipeline=pipeline,
        controller_loop=DefaultLoop(
            trainer=Trainer(
                validation_splitter=ValidationSplitter(0.20),
                callbacks=[
                    MetricCallback('mse', metric_function=mean_squared_error, higher_score_is_better=False),
                    MetricCallback('accuracy', metric_function=accuracy_score, higher_score_is_better=False),
                    # EarlyStoppingCallback(max_epochs_without_improvement=3)
                ],
                main_metric_name="mse"
            ),
            # or `Trainer(...).with_val_set(sdsdfg)`  # TODO: add this `with_val_set` method that would change splitter to PresetValidationSetSplitter(self, val) and override.
            next_best_prediction_algo=RandomSearchHyperparameterSelectionStrategy(),
            n_trials=17,
            n_epochs=11,
            continue_loop_on_error=True
        ),
        hp_repo=HyperparamsJSONRepository(cache_folder=os.path.join(tmpdir, "hp")),
        start_new_run=True,  # otherwise, pick last run.
        refit_best_trial=True,
    )

    a, _out = a.handle_fit_transform(
        dact,
        ExecutionContext(root=os.path.join(tmpdir, "automl"))
    )

    assert _out is not None
