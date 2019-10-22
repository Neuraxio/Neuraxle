import os
import shutil

from neuraxle.base import DEFAULT_CACHE_FOLDER


def test_non_fittable_mixin():
    os.makedirs(DEFAULT_CACHE_FOLDER)

    from examples.non_fittable_mixin import main
    main()

    shutil.rmtree(DEFAULT_CACHE_FOLDER)

def test_inverse_transform():
    os.makedirs(DEFAULT_CACHE_FOLDER)

    from examples.inverse_transform import main
    main()

    shutil.rmtree(DEFAULT_CACHE_FOLDER)

def test_nested_pipelines():
    os.makedirs(DEFAULT_CACHE_FOLDER)

    from examples.nested_pipelines import main
    main()

    shutil.rmtree(DEFAULT_CACHE_FOLDER)

def test_hyperparams():
    os.makedirs(DEFAULT_CACHE_FOLDER)

    from examples.hyperparams import main
    main()

    shutil.rmtree(DEFAULT_CACHE_FOLDER)

def test_rest_api_serving():
    os.makedirs(DEFAULT_CACHE_FOLDER)

    from examples.easy_rest_api_serving import main
    main()

    shutil.rmtree(DEFAULT_CACHE_FOLDER)

def test_boston_housing_regression_with_model_stacking():
    os.makedirs(DEFAULT_CACHE_FOLDER)

    from examples.boston_housing_regression_with_model_stacking import main
    main()

    shutil.rmtree(DEFAULT_CACHE_FOLDER)


def test_boston_housing_meta_optimization():
    os.makedirs(DEFAULT_CACHE_FOLDER)

    from examples.boston_housing_meta_optimization import main
    main()

    shutil.rmtree(DEFAULT_CACHE_FOLDER)


def test_automl_checkpoint():
    os.makedirs(DEFAULT_CACHE_FOLDER)

    from examples.auto_ml_checkpoint import main
    main(0)

    shutil.rmtree(DEFAULT_CACHE_FOLDER)
