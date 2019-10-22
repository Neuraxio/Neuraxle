import numpy as np

from examples.auto_ml_checkpoint import run_random_search_normal_pipeline, run_random_search_resumable_pipeline, main
from neuraxle.hyperparams.space import HyperparameterSamples


def test_automl_checkpoints():
    main(0.0)
