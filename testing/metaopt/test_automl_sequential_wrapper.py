from neuraxle.base import Identity

from neuraxle.metaopt.auto_ml import HyperparamsJSONRepository, AutoMLSequentialWrapper


def test_automl_sequential_wrapper(tmpdir):
    auto_ml_sequential_wrapper = AutoMLSequentialWrapper(
        auto_ml_strategy=Identity(),
        hyperparams_repository=HyperparamsJSONRepository(tmpdir),
        n_iters=10
    )
    pass
