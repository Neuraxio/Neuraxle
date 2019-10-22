import os
import shutil

from examples.non_fittable_mixin import main
from neuraxle.base import DEFAULT_CACHE_FOLDER

def test_auto_ml_checkpoint():
    from examples.auto_ml_checkpoint import main
    example_test(lambda: main(0))

def test_boston_housing_meta_optimization():
    from examples.boston_housing_meta_optimization import main
    example_test(lambda: main())

def test_boston_housing_regression_with_model_stacking():
    from examples.boston_housing_regression_with_model_stacking import main
    example_test(lambda: main())

def test_easy_rest_api_serving():
    from examples.easy_rest_api_serving import main
    example_test(lambda: main())

def test_hyperparams():
    from examples.hyperparams import main
    example_test(lambda: main())

def test_inverse_transform():
    from examples.inverse_transform import main
    example_test(lambda: main())

def test_nested_pipelines():
    from examples.nested_pipelines import main
    example_test(lambda: main())

def test_non_fittable_mixin():
    example_test(lambda: main())

def example_test(example_method):
    if not os.path.exists(DEFAULT_CACHE_FOLDER):
        os.makedirs(DEFAULT_CACHE_FOLDER)

    try:
        example_method()
    except Exception as err:
        shutil.rmtree(DEFAULT_CACHE_FOLDER)
        raise err

    shutil.rmtree(DEFAULT_CACHE_FOLDER)
