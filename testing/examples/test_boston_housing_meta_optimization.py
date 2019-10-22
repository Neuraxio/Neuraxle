from examples.boston_housing_meta_optimization import main


def test_boston_housing_meta_optimization():
    y_train_predicted, y_test_predicted, score_transform, score_test = main()

    assert y_train_predicted is not None
    assert y_test_predicted is not None
    assert score_transform is not None
    assert score_test is not None
