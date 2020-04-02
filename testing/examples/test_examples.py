from py._path.local import LocalPath


def test_auto_ml_checkpoint(tmpdir: LocalPath):
    from examples.caching.plot_auto_ml_checkpoint import main
    main(tmpdir)


def test_boston_housing_meta_optimization(tmpdir: LocalPath):
    from examples.sklearn.plot_boston_housing_meta_optimization import main
    main(tmpdir)


def test_boston_housing_regression_with_model_stacking():
    from examples.sklearn.plot_boston_housing_regression_with_model_stacking import main
    main()


def test_easy_rest_api_serving():
    from examples.deployment.plot_easy_rest_api_serving import main
    main()


def test_hyperparams():
    from examples.hyperparams.plot_hyperparams import main
    main()


def test_inverse_transform():
    from examples.getting_started.plot_inverse_transform import main
    main()


def test_nested_pipelines():
    from examples.getting_started.plot_nested_pipelines import main
    main()


def test_non_fittable_mixin():
    from examples.getting_started.plot_non_fittable_mixin import main
    main()


def test_label_encoder_across_multiple_columns():
    from examples.getting_started.plot_label_encoder_across_multiple_columns import main
    main()
