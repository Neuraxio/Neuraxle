from py._path.local import LocalPath


def test_auto_ml_checkpoint(tmpdir: LocalPath):
    from examples.caching.auto_ml_checkpoint import main
    main(tmpdir)


def test_boston_housing_meta_optimization(tmpdir: LocalPath):
    from examples.sklearn.boston_housing_meta_optimization import main
    main(tmpdir)


def test_boston_housing_regression_with_model_stacking():
    from examples.sklearn.boston_housing_regression_with_model_stacking import main
    main()


def test_easy_rest_api_serving():
    from examples.deployment.easy_rest_api_serving import main
    main()


def test_hyperparams():
    from examples.hyperparams import main
    main()


def test_inverse_transform():
    from examples.getting_started.inverse_transform import main
    main()


def test_nested_pipelines():
    from examples.getting_started.nested_pipelines import main
    main()


def test_non_fittable_mixin():
    from examples.getting_started.non_fittable_mixin import main
    main()


def test_label_encoder_across_multiple_columns():
    from examples.getting_started.label_encoder_across_multiple_columns import main
    main()
