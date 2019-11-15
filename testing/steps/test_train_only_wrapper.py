import numpy as np

from neuraxle.steps.flow import TrainOnlyWrapper
from neuraxle.steps.misc import TransformCallbackStep, TapeCallbackFunction


def test_train_only_wrapper_given_test_only_true_should_transform_in_test_mode():
    tape = TapeCallbackFunction()
    data_inputs = np.array([1])
    step = TrainOnlyWrapper(TransformCallbackStep(tape))
    step.set_train(False)

    data_inputs = step.transform(data_inputs)

    assert len(tape.data) > 0


def test_train_only_wrapper_given_test_only_false_should_transform_in_train_mode():
    tape = TapeCallbackFunction()
    data_inputs = np.array([1])
    step = TrainOnlyWrapper(TransformCallbackStep(tape))

    data_inputs = step.transform(data_inputs)

    assert len(tape.data) > 0


def test_train_only_wrapper_given_test_only_true_should_not_transform_in_train_mode():
    tape = TapeCallbackFunction()
    data_inputs = np.array([1])
    step = TrainOnlyWrapper(TransformCallbackStep(tape))
    step.set_train(False)

    data_inputs = step.transform(data_inputs)

    assert len(tape.data) > 0


def test_train_only_wrapper_given_test_only_false_should_not_transform_in_test_mode():
    tape = TapeCallbackFunction()
    data_inputs = np.array([1])
    step = TrainOnlyWrapper(TransformCallbackStep(tape))

    data_inputs = step.transform(data_inputs)

    assert len(tape.data) > 0


def test_train_only_wrapper_given_test_only_true_should_fit_transform_in_test_mode():
    tape = TapeCallbackFunction()
    data_inputs = np.array([1])
    step = TrainOnlyWrapper(TransformCallbackStep(tape))
    step.set_train(False)

    data_inputs = step.transform(data_inputs)

    assert len(tape.data) > 0
    pass


def test_train_only_wrapper_given_test_only_false_should_fit_transform_in_train_mode():
    tape = TapeCallbackFunction()
    data_inputs = np.array([1])
    step = TrainOnlyWrapper(TransformCallbackStep(tape))

    data_inputs = step.transform(data_inputs)

    assert len(tape.data) > 0


def test_train_only_wrapper_given_test_only_true_should_not_fit_transform_in_train_mode():
    tape = TapeCallbackFunction()
    data_inputs = np.array([1])
    step = TrainOnlyWrapper(TransformCallbackStep(tape))
    step.set_train(False)

    data_inputs = step.transform(data_inputs)

    assert len(tape.data) > 0
    pass


def test_train_only_wrapper_given_test_only_false_should_not_fit_transform_in_test_mode():
    tape = TapeCallbackFunction()
    data_inputs = np.array([1])
    step = TrainOnlyWrapper(TransformCallbackStep(tape))

    data_inputs = step.transform(data_inputs)

    assert len(tape.data) > 0


def test_train_only_wrapper_given_test_only_true_should_fit_in_test_mode():
    tape = TapeCallbackFunction()
    data_inputs = np.array([1])
    step = TrainOnlyWrapper(TransformCallbackStep(tape))
    step.set_train(False)

    data_inputs = step.transform(data_inputs)

    assert len(tape.data) > 0
    pass


def test_train_only_wrapper_given_test_only_false_should_fit_in_train_mode():
    tape = TapeCallbackFunction()
    data_inputs = np.array([1])
    step = TrainOnlyWrapper(TransformCallbackStep(tape))

    data_inputs = step.transform(data_inputs)

    assert len(tape.data) > 0


def test_train_only_wrapper_given_test_only_true_should_not_fit_in_train_mode():
    tape = TapeCallbackFunction()
    data_inputs = np.array([1])
    step = TrainOnlyWrapper(TransformCallbackStep(tape))
    step.set_train(False)

    data_inputs = step.transform(data_inputs)

    assert len(tape.data) > 0
    pass


def test_train_only_wrapper_given_test_only_false_should_not_fit_in_test_mode():
    tape = TapeCallbackFunction()
    data_inputs = np.array([1])
    step = TrainOnlyWrapper(TransformCallbackStep(tape))

    data_inputs = step.transform(data_inputs)

    assert len(tape.data) > 0
