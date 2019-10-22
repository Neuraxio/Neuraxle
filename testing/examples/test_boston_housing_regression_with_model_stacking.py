from examples.boston_housing_regression_with_model_stacking import main


def test_boston_housing_regression_with_model_stacking():
    y_train_predicted, y_test_predicted, score_train, score_test = main()

    assert y_train_predicted.shape == (379,)
    assert y_test_predicted.shape == (127,)
    assert isinstance(score_train, float)
    assert isinstance(score_test, float)
