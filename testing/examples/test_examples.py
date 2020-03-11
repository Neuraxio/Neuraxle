from py._path.local import LocalPath


def test_auto_ml_checkpoint(tmpdir: LocalPath):
    from examples.auto_ml_checkpoint import main
    main(tmpdir)


def test_boston_housing_meta_optimization(tmpdir: LocalPath):
    from examples.boston_housing_meta_optimization import main
    main(tmpdir)


def test_boston_housing_regression_with_model_stacking():
    from examples.boston_housing_regression_with_model_stacking import main
    main()


def test_easy_rest_api_serving():
    from examples.easy_rest_api_serving import main
    main()


def test_hyperparams():
    from examples.hyperparams import main
    main()


def test_inverse_transform():
    from examples.inverse_transform import main
    main()


def test_nested_pipelines():
    from examples.nested_pipelines import main
    main()


def test_non_fittable_mixin():
    from examples.non_fittable_mixin import main
    main()


def test_label_encoder_across_multiple_columns():
    from examples.label_encoder_across_multiple_columns import main
    main()
