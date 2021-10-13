import numpy as np

from neuraxle.base import ExecutionContext
from neuraxle.data_container import DataContainer
from neuraxle.pipeline import Pipeline
from neuraxle.steps.flow import TestOnlyWrapper, TrainOnlyWrapper
from neuraxle.steps.misc import TapeCallbackFunction, CallbackWrapper
from neuraxle.steps.numpy import MultiplyByN

from testing.mocks.step_mocks import SomeStepWithHyperparams


def test_basestep_print_str_representation_works_correctly():
    output = str(SomeStepWithHyperparams())
    assert output == "SomeStepWithHyperparams(name='MockStep')"


def test_basestep_repr_representation_works_correctly():
    output = repr(SomeStepWithHyperparams())
    assert output == """SomeStepWithHyperparams(name='MockStep', hyperparameters=HyperparameterSamples([('learning_rate', 0.1),
                       ('l2_weight_reg', 0.001),
                       ('hidden_size', 32),
                       ('num_layers', 3),
                       ('num_lstm_layers', 1),
                       ('use_xavier_init', True),
                       ('use_max_pool_else_avg_pool', True),
                       ('dropout_drop_proba', 0.5),
                       ('momentum', 0.1)]))"""


def test_handle_predict_should_predict_in_test_mode():
    tape_fit = TapeCallbackFunction()
    tape_transform = TapeCallbackFunction()
    p = Pipeline([
        TestOnlyWrapper(CallbackWrapper(MultiplyByN(2), tape_transform, tape_fit)),
        TrainOnlyWrapper(CallbackWrapper(MultiplyByN(4), tape_transform, tape_fit))
    ])

    data_container = p.handle_predict(
        data_container=DataContainer(data_inputs=np.array([1, 1]), expected_outputs=np.array([1, 1])),
        context=ExecutionContext()
    )

    assert np.array_equal(data_container.data_inputs, np.array([2, 2]))


def test_handle_predict_should_handle_transform_with_initial_is_train_mode_after_predict():
    tape_fit = TapeCallbackFunction()
    tape_transform = TapeCallbackFunction()
    p = Pipeline([
        TestOnlyWrapper(CallbackWrapper(MultiplyByN(2), tape_transform, tape_fit)),
        TrainOnlyWrapper(CallbackWrapper(MultiplyByN(4), tape_transform, tape_fit))
    ])
    data_container = DataContainer(data_inputs=np.array([1, 1]), expected_outputs=np.array([1, 1]))

    p.handle_predict(
        data_container=data_container.copy(),
        context=ExecutionContext()
    )
    data_container = p.handle_transform(data_container, ExecutionContext())

    assert np.array_equal(data_container.data_inputs, np.array([4, 4]))


def test_predict_should_predict_in_test_mode():
    tape_fit = TapeCallbackFunction()
    tape_transform = TapeCallbackFunction()
    p = Pipeline([
        TestOnlyWrapper(CallbackWrapper(MultiplyByN(2), tape_transform, tape_fit)),
        TrainOnlyWrapper(CallbackWrapper(MultiplyByN(4), tape_transform, tape_fit))
    ])

    outputs = p.predict(np.array([1, 1]))

    assert np.array_equal(outputs, np.array([2, 2]))


def test_predict_should_transform_with_initial_is_train_mode_after_predict():
    tape_fit = TapeCallbackFunction()
    tape_transform = TapeCallbackFunction()
    p = Pipeline([
        TestOnlyWrapper(CallbackWrapper(MultiplyByN(2), tape_transform, tape_fit)),
        TrainOnlyWrapper(CallbackWrapper(MultiplyByN(4), tape_transform, tape_fit))
    ])

    p.predict(np.array([1, 1]))
    outputs = p.transform(np.array([1, 1]))

    assert np.array_equal(outputs, np.array([4, 4]))
