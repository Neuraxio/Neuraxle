import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.distributions import FixedHyperparameter
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import InMemoryHyperparamsRepository, AutoML, RandomSearchHyperparameterSelectionStrategy, \
    kfold_cross_validation_split, create_split_data_container_function, validation_splitter, HyperparamsJSONRepository
from neuraxle.metaopt.callbacks import MetricCallback, ScoringCallback
from neuraxle.pipeline import Pipeline
from neuraxle.steps.misc import FitTransformCallbackStep
from neuraxle.steps.numpy import MultiplyByN, NumpyReshape


def test_automl_early_stopping_callback(tmpdir):
    # TODO: fix this unit test
    # Given
    hp_repository = InMemoryHyperparamsRepository(cache_folder=str(tmpdir))
    n_epochs = 60
    auto_ml = AutoML(
        pipeline=Pipeline([
            FitTransformCallbackStep().set_name('callback'),
            MultiplyByN(2).set_hyperparams_space(HyperparameterSpace({
                'multiply_by': FixedHyperparameter(2)
            })),
            NumpyReshape(new_shape=(-1, 1)),
            linear_model.LinearRegression()
        ]),
        hyperparams_optimizer=RandomSearchHyperparameterSelectionStrategy(),
        validation_split_function=validation_splitter(0.20),
        scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
        callbacks=[
            MetricCallback('mse', metric_function=mean_squared_error, higher_score_is_better=False),
        ],
        n_trials=1,
        refit_trial=True,
        epochs=n_epochs,
        hyperparams_repository=hp_repository
    )

    # When
    data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_outputs = data_inputs * 2
    auto_ml = auto_ml.fit(data_inputs=data_inputs, expected_outputs=expected_outputs)

    # Then
    p = auto_ml.get_best_model()


def test_automl_with_kfold(tmpdir):
    # Given
    hp_repository = HyperparamsJSONRepository(cache_folder=str(tmpdir))
    auto_ml = AutoML(
        pipeline=Pipeline([
            MultiplyByN(2).set_hyperparams_space(HyperparameterSpace({
                'multiply_by': FixedHyperparameter(2)
            })),
            NumpyReshape(new_shape=(-1, 1)),
            linear_model.LinearRegression()
        ]),
        validation_split_function=validation_splitter(0.20),
        hyperparams_optimizer=RandomSearchHyperparameterSelectionStrategy(),
        scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
        callbacks=[
            MetricCallback('mse', metric_function=mean_squared_error,
                           higher_score_is_better=False),
        ],
        n_trials=1,
        epochs=10,
        refit_trial=True,
        hyperparams_repository=hp_repository
    )

    data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_outputs = data_inputs * 4

    # When
    auto_ml.fit(data_inputs=data_inputs, expected_outputs=expected_outputs)

    # Then
    p = auto_ml.get_best_model()
    outputs = p.transform(data_inputs)
    mse = mean_squared_error(expected_outputs, outputs)

    assert mse < 1000


def test_kfold_cross_validation_should_split_data_properly():
    # Given
    data_inputs = np.random.random((4, 2, 2048, 6)).astype(np.float32)
    expected_outputs = np.random.random((4, 2, 2048, 1)).astype(np.float32)
    splitter = create_split_data_container_function(kfold_cross_validation_split(k_fold=4))

    # When
    train_data_container, validation_data_container = splitter(
        DataContainer(data_inputs=data_inputs, expected_outputs=expected_outputs)
    )

    train_di = train_data_container.data_inputs
    train_eo = train_data_container.expected_outputs

    validation_di = validation_data_container.data_inputs
    validation_eo = validation_data_container.expected_outputs

    # Then
    assert len(train_di[0]) == 3
    assert np.array_equal(np.array(train_di[0]), data_inputs[1:])
    assert len(train_eo[0]) == 3
    assert np.array_equal(np.array(train_eo[0]), expected_outputs[1:])

    assert len(train_di[1]) == 3
    assert np.array_equal(np.array(train_di[1]),
                          np.concatenate((np.expand_dims(data_inputs[0], axis=0), data_inputs[2:]), axis=0))
    assert len(train_eo[1]) == 3
    assert np.array_equal(np.array(train_eo[1]),
                          np.concatenate((np.expand_dims(expected_outputs[0], axis=0), expected_outputs[2:]), axis=0))

    assert len(train_di[2]) == 3
    assert np.array_equal(np.array(train_di[2]),
                          np.concatenate((data_inputs[0:2], np.expand_dims(data_inputs[3], axis=0)), axis=0))
    assert len(train_eo[2]) == 3
    assert np.array_equal(np.array(train_eo[2]),
                          np.concatenate((expected_outputs[0:2], np.expand_dims(expected_outputs[3], axis=0)), axis=0))

    assert len(train_di[3]) == 3
    assert np.array_equal(np.array(train_di[3]), data_inputs[0:3])
    assert len(train_eo[3]) == 3
    assert np.array_equal(np.array(train_eo[3]), expected_outputs[0:3])

    assert len(validation_di[0]) == 1
    assert np.array_equal(validation_di[0][0], data_inputs[0])
    assert len(validation_eo[0]) == 1
    assert np.array_equal(validation_eo[0][0], expected_outputs[0])

    assert len(validation_di[1]) == 1
    assert np.array_equal(validation_di[1][0], data_inputs[1])
    assert len(validation_eo[1]) == 1
    assert np.array_equal(validation_eo[1][0], expected_outputs[1])

    assert len(validation_di[2]) == 1
    assert np.array_equal(validation_di[2][0], data_inputs[2])
    assert len(validation_eo[2]) == 1
    assert np.array_equal(validation_eo[2][0], expected_outputs[2])

    assert len(validation_di[3]) == 1
    assert np.array_equal(validation_di[3][0], data_inputs[3])
    assert len(validation_eo[3]) == 1
    assert np.array_equal(validation_eo[3][0], expected_outputs[3])


def test_kfold_cross_validation_should_split_data_properly_bug():
    data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_outputs = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    data_container = DataContainer(
        data_inputs=data_inputs,
        expected_outputs=expected_outputs
    )
    splitter = create_split_data_container_function(kfold_cross_validation_split(k_fold=2))

    # When
    train_data_container, validation_data_container = splitter(data_container)

    train_di = train_data_container.data_inputs
    train_eo = train_data_container.expected_outputs

    validation_di = validation_data_container.data_inputs
    validation_eo = validation_data_container.expected_outputs

    # Then
    assert len(train_di[0]) == 6
    assert np.array_equal(np.array(train_di[0]), data_inputs[5:])
    assert len(train_eo[0]) == 6
    assert np.array_equal(np.array(train_eo[0]), expected_outputs[5:])

    assert len(train_di[1]) == 5
    assert np.array_equal(
        np.array(train_di[1]),
        data_inputs[:5]
    )
    assert len(train_eo[1]) == 5
    assert np.array_equal(
        np.array(train_eo[1]),
        expected_outputs[:5]
    )

    assert len(validation_di[0]) == 5
    assert np.array_equal(np.array(validation_di[0]), data_inputs[:5])
    assert len(validation_eo[0]) == 5
    assert np.array_equal(np.array(validation_eo[0]), expected_outputs[:5])

    assert len(validation_di[1]) == 6
    assert np.array_equal(np.array(validation_di[1]), data_inputs[5:])
    assert len(validation_eo[1]) == 6
    assert np.array_equal(validation_eo[1], expected_outputs[5:])
