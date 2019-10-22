import numpy as np

from examples.auto_ml_checkpoint import run_random_search_normal_pipeline, run_random_search_resumable_pipeline
from neuraxle.hyperparams.space import HyperparameterSamples


def test_run_random_search_normal_pipeline():
    actual_score, outputs, hyperparams = run_random_search_normal_pipeline(0)
    assert isinstance(actual_score, float)
    assert isinstance(hyperparams, HyperparameterSamples)
    assert isinstance(outputs, np.ndarray)

def test_run_random_search_resumable_pipeline():
    actual_score, outputs, hyperparams = run_random_search_resumable_pipeline(0)
    assert isinstance(actual_score, float)
    assert isinstance(hyperparams, HyperparameterSamples)
    assert isinstance(outputs, np.ndarray)
