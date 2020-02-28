import numpy as np
from sklearn.metrics import mean_squared_error

from neuraxle.metaopt.auto_ml import RandomSearchHyperparameterOptimizer
from neuraxle.metaopt.new_automl import AutoML, InMemoryHyperparamsRepository
from neuraxle.metaopt.random import ValidationSplitWrapper, KFoldCrossValidationWrapper
from neuraxle.pipeline import Pipeline


def test_automl_should_start_trial_with_validation_split_wrapper():
    auto_ml = AutoML(
        pipeline=Pipeline([]),
        validation_technique=ValidationSplitWrapper(test_size=0.2, scoring_function=mean_squared_error),
        hyperparams_repository=InMemoryHyperparamsRepository(RandomSearchHyperparameterOptimizer()),
        scoring_function=mean_squared_error,
        n_trial=1,
        metrics={'mse': mean_squared_error},
        epochs=2,
        callbacks=[]
    )

    auto_ml = auto_ml.fit(data_inputs=np.array([0, 1]), expected_outputs=np.array([0, 1]))

    assert auto_ml


def test_automl_should_start_trial_with_merge_kfold():
    auto_ml = AutoML(
        pipeline=Pipeline([]),
        validation_technique=KFoldCrossValidationWrapper(k_fold=2, scoring_function=mean_squared_error),
        hyperparams_repository=InMemoryHyperparamsRepository(RandomSearchHyperparameterOptimizer()),
        scoring_function=mean_squared_error,
        n_trial=1,
        metrics={'mse': mean_squared_error},
        epochs=2,
        callbacks=[]
    )

    auto_ml = auto_ml.fit(data_inputs=np.array([0, 1]), expected_outputs=np.array([0, 1]))

    assert auto_ml
