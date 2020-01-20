import math

import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from neuraxle.api import DeepLearningPipeline
from neuraxle.pipeline import Pipeline

TIMESTEPS = 10

VALIDATION_SIZE = 0.1
BATCH_SIZE = 32
N_EPOCHS = 10

DATA_INPUTS_PAST_SHAPE = (BATCH_SIZE, TIMESTEPS)


def test_deep_learning_pipeline():
    # Given
    i = 0
    data_inputs = []
    for batch_index in range(BATCH_SIZE):
        batch = []
        for _ in range(TIMESTEPS):
            batch.append(i)
            i += 1
        data_inputs.append(batch)
    data_inputs = np.array(data_inputs)

    random_noise = np.random.random(DATA_INPUTS_PAST_SHAPE)
    expected_outputs = 3 * data_inputs + 4 * random_noise
    expected_outputs = expected_outputs.astype(np.float32)
    data_inputs = data_inputs.astype(np.float32)

    p = DeepLearningPipeline(
        Pipeline([
            linear_model.LinearRegression()
        ]),
        validation_size=VALIDATION_SIZE,
        batch_size=BATCH_SIZE,
        batch_metrics={'mse': to_numpy_metric_wrapper(mean_squared_error)},
        shuffle_in_each_epoch_at_train=True,
        n_epochs=N_EPOCHS,
        epochs_metrics={'mse': to_numpy_metric_wrapper(mean_squared_error)},
        scoring_function=to_numpy_metric_wrapper(mean_squared_error),
    )

    # When
    p, outputs = p.fit_transform(data_inputs, expected_outputs)

    # Then
    batch_mse_train = p.get_batch_metric_train('mse')
    epoch_mse_train = p.get_epoch_metric_train('mse')

    batch_mse_validation = p.get_batch_metric_validation('mse')
    epoch_mse_validation = p.get_epoch_metric_validation('mse')

    assert len(epoch_mse_train) == N_EPOCHS
    assert len(epoch_mse_validation) == N_EPOCHS

    expected_len_batch_mse_train = math.ceil((len(data_inputs) / BATCH_SIZE) * (1 - VALIDATION_SIZE)) * N_EPOCHS
    expected_len_batch_mse_validation = math.ceil((len(data_inputs) / BATCH_SIZE) * VALIDATION_SIZE) * N_EPOCHS

    assert len(batch_mse_train) == expected_len_batch_mse_train
    assert len(batch_mse_validation) == expected_len_batch_mse_validation

    last_batch_mse_validation = batch_mse_validation[-1]
    last_batch_mse_train = batch_mse_train[-1]

    last_epoch_mse_train = epoch_mse_train[-1]
    last_epoch_mse_validation = epoch_mse_validation[-1]

    assert last_batch_mse_train < last_batch_mse_validation
    assert last_epoch_mse_train < last_epoch_mse_validation


def to_numpy_metric_wrapper(metric_fun):
    def metric(data_inputs, expected_outputs):
        return metric_fun(np.array(data_inputs), np.array(expected_outputs))

    return metric
