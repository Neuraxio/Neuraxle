import numpy as np
import pytest
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVC

from neuraxle.base import ExecutionContext
from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.distributions import FixedHyperparameter, RandInt
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import InMemoryHyperparamsRepository, AutoML, RandomSearchHyperparameterSelectionStrategy, \
    HyperparamsJSONRepository, \
    ValidationSplitter, KFoldCrossValidationSplitter, Trainer
from neuraxle.metaopt.callbacks import MetricCallback, ScoringCallback, EarlyStoppingCallback, SaveBestModelCallback
from neuraxle.metaopt.trial import Trial, Trials
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import MultiplyByN, NumpyReshape
from neuraxle.steps.sklearn import SKLearnWrapper


def test_automl_early_stopping_callback(tmpdir):
    # Given
    hp_repository = InMemoryHyperparamsRepository(cache_folder=str(tmpdir))
    n_epochs = 10
    max_epochs_without_improvement=3
    auto_ml = AutoML(
        pipeline=Pipeline([
            MultiplyByN(2).set_hyperparams_space(HyperparameterSpace({
                'multiply_by': FixedHyperparameter(2)
            })),
            NumpyReshape(new_shape=(-1, 1)),
        ]),
        hyperparams_optimizer=RandomSearchHyperparameterSelectionStrategy(),
        validation_splitter=ValidationSplitter(0.20),
        scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
        callbacks=[
            MetricCallback('mse', metric_function=mean_squared_error, higher_score_is_better=False),
            EarlyStoppingCallback(max_epochs_without_improvement)
        ],
        n_trials=1,
        refit_trial=True,
        epochs=n_epochs,
        hyperparams_repository=hp_repository,
        continue_loop_on_error=False
    )

    # When
    data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_outputs = data_inputs * 2
    auto_ml.fit(data_inputs=data_inputs, expected_outputs=expected_outputs)

    # Then
    trial = hp_repository.trials[0]
    assert len(trial.validation_splits) == 1
    validation_scores = trial.validation_splits[0].get_validation_scores()
    nepochs_executed = len(validation_scores)
    assert nepochs_executed == max_epochs_without_improvement +1

@pytest.mark.skip
def test_automl_savebestmodel_callback(tmpdir):
    # Given
    hp_repository = HyperparamsJSONRepository(cache_folder=str('caching'))
    validation_splitter = ValidationSplitter(0.20)
    auto_ml = AutoML(
        pipeline=Pipeline([
            MultiplyByN(2).set_hyperparams_space(HyperparameterSpace({
                'multiply_by': FixedHyperparameter(2)
            })),
            NumpyReshape(new_shape=(-1, 1)),
            linear_model.LinearRegression()
        ]),
        validation_splitter=validation_splitter,
        hyperparams_optimizer=RandomSearchHyperparameterSelectionStrategy(),
        scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
        callbacks=[
            SaveBestModelCallback()
        ],
        n_trials=1,
        epochs=10,
        refit_trial=False,
        print_func=print,
        hyperparams_repository=hp_repository,
        continue_loop_on_error=False
    )

    data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_outputs = data_inputs * 4

    # When
    auto_ml.fit(data_inputs=data_inputs, expected_outputs=expected_outputs)


    #Then
    trials: Trials = hp_repository.load_all_trials()
    best_trial = trials.get_best_trial()
    best_trial_score = best_trial.get_validation_score()
    best_trial.cache_folder = hp_repository.cache_folder
    best_model = best_trial.get_model('best')
    _, _, valid_inputs, valid_outputs = ValidationSplitter(0.20).split(data_inputs, expected_outputs)
    predicted_output = best_model.predict(valid_inputs)
    score = mean_squared_error(valid_outputs, predicted_output)

    assert best_trial_score == score

def test_automl_with_kfold(tmpdir):
    # Given
    hp_repository = HyperparamsJSONRepository(cache_folder=str('caching'))
    auto_ml = AutoML(
        pipeline=Pipeline([
            MultiplyByN(2).set_hyperparams_space(HyperparameterSpace({
                'multiply_by': FixedHyperparameter(2)
            })),
            NumpyReshape(new_shape=(-1, 1)),
            linear_model.LinearRegression()
        ]),
        validation_splitter=ValidationSplitter(0.20),
        hyperparams_optimizer=RandomSearchHyperparameterSelectionStrategy(),
        scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
        callbacks=[
            MetricCallback('mse', metric_function=mean_squared_error,
                           higher_score_is_better=False),
        ],
        n_trials=1,
        epochs=10,
        refit_trial=True,
        hyperparams_repository=hp_repository,
        continue_loop_on_error=False
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


def test_validation_splitter_should_split_data_properly():
    # Given
    data_inputs = np.random.random((4, 2, 2048, 6)).astype(np.float32)
    expected_outputs = np.random.random((4, 2, 2048, 1)).astype(np.float32)
    splitter = ValidationSplitter(test_size=0.2)

    # When
    validation_splits = splitter.split_data_container(
        data_container=DataContainer(data_inputs=data_inputs, expected_outputs=expected_outputs),
        context=ExecutionContext()
    )
    train_di, train_eo, validation_di, validation_eo = extract_validation_split_data(validation_splits)

    train_di = train_di[0]
    train_eo = train_eo[0]

    validation_di = validation_di[0]
    validation_eo = validation_eo[0]

    # Then
    assert len(train_di) == 3
    assert np.array_equal(np.array(train_di), data_inputs[0:3])
    assert len(train_eo) == 3
    assert np.array_equal(np.array(train_eo), expected_outputs[0:3])

    assert len(validation_di) == 1
    assert np.array_equal(validation_di[0], data_inputs[-1])
    assert len(validation_eo) == 1
    assert np.array_equal(validation_eo[0], expected_outputs[-1])


def test_kfold_cross_validation_should_split_data_properly():
    # Given
    data_inputs = np.random.random((4, 2, 2048, 6)).astype(np.float32)
    expected_outputs = np.random.random((4, 2, 2048, 1)).astype(np.float32)
    splitter = KFoldCrossValidationSplitter(k_fold=4)

    # When
    validation_splits = splitter.split_data_container(
        data_container=DataContainer(data_inputs=data_inputs, expected_outputs=expected_outputs),
        context=ExecutionContext()
    )
    train_di, train_eo, validation_di, validation_eo = extract_validation_split_data(validation_splits)

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
    splitter = KFoldCrossValidationSplitter(k_fold=2)

    # When
    validation_splits = splitter.split_data_container(data_container, ExecutionContext())

    train_di, train_eo, validation_di, validation_eo = extract_validation_split_data(validation_splits)

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


def extract_validation_split_data(validation_splits):
    train_di = []
    train_eo = []
    validation_di = []
    validation_eo = []
    for train_dc, validation_dc in validation_splits:
        train_di.append(train_dc.data_inputs)
        train_eo.append(train_dc.expected_outputs)

        validation_di.append(validation_dc.data_inputs)
        validation_eo.append(validation_dc.expected_outputs)
    return train_di, train_eo, validation_di, validation_eo


def test_automl_should_shallow_copy_data_before_each_epoch():
    # see issue #332 https://github.com/Neuraxio/Neuraxle/issues/332
    data_inputs = np.random.randint(0, 100, (100, 3))
    expected_outputs = np.random.randint(0, 3, 100)

    from sklearn.preprocessing import StandardScaler
    p = Pipeline([
        SKLearnWrapper(StandardScaler()),
        SKLearnWrapper(LinearSVC(), HyperparameterSpace({'C': RandInt(0, 10000)})),
    ])

    auto_ml = AutoML(
        p,
        validation_splitter=ValidationSplitter(0.20),
        refit_trial=True,
        n_trials=10,
        epochs=10,
        cache_folder_when_no_handle='cache',
        scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
        callbacks=[MetricCallback('mse', metric_function=mean_squared_error, higher_score_is_better=False)],
        hyperparams_repository=InMemoryHyperparamsRepository(
            cache_folder='cache'),
        continue_loop_on_error=False
    )

    random_search = auto_ml.fit(data_inputs, expected_outputs)

    best_model = random_search.get_best_model()

    assert isinstance(best_model, Pipeline)


def test_trainer_train():
    data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_outputs = data_inputs * 4
    p = Pipeline([
        MultiplyByN(2).set_hyperparams_space(HyperparameterSpace({
            'multiply_by': FixedHyperparameter(2)
        })),
        NumpyReshape(new_shape=(-1, 1)),
        linear_model.LinearRegression()
    ])

    trainer: Trainer = Trainer(
        epochs=10,
        scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
        validation_splitter=ValidationSplitter(test_size=0.20)
    )

    repo_trial: Trial = trainer.train(pipeline=p, data_inputs=data_inputs, expected_outputs=expected_outputs)

    trained_pipeline = repo_trial.get_trained_pipeline(split_number=0)

    outputs = trained_pipeline.transform(data_inputs)
    mse = mean_squared_error(expected_outputs, outputs)

    assert mse < 1

