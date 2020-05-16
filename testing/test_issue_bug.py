import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from neuraxle.hyperparams.distributions import RandInt
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import AutoML, InMemoryHyperparamsRepository, \
    ValidationSplitter
from neuraxle.metaopt.callbacks import MetricCallback, ScoringCallback
from neuraxle.pipeline import Pipeline
from neuraxle.steps.sklearn import SKLearnWrapper


def test_issue():
    data_inputs = np.random.randint(0, 100, (100, 3))
    expected_outputs = np.random.randint(0, 3, 100)

    p = Pipeline([
        SKLearnWrapper(StandardScaler()),
        SKLearnWrapper(LinearSVC(),
                       HyperparameterSpace({'C': RandInt(0, 10000)})),
    ])

    auto_ml = AutoML(
        p,
        validation_splitter=ValidationSplitter(0.20),
        refit_trial=True,
        n_trials=10,
        epochs=10,
        cache_folder_when_no_handle='cache',
        scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
        callbacks=[MetricCallback('mse', metric_function=mean_squared_error,
                                  higher_score_is_better=False)],
        hyperparams_repository=InMemoryHyperparamsRepository(
            cache_folder='cache')
    )

    random_search = auto_ml.fit(data_inputs, expected_outputs)
