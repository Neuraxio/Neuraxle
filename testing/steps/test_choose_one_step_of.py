import numpy as np
import pytest

from neuraxle.hyperparams.distributions import Boolean
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.pipeline import Pipeline
from neuraxle.steps.flow import ChooseOneOrManyStepsOf
from neuraxle.steps.misc import TransformCallbackStep, TapeCallbackFunction, FitTransformCallbackStep

DATA_INPUTS = np.array(range(10))
EXPECTED_OUTPUTS = np.array(range(10))


class ChooseStepsTestCase:
    def __init__(
            self,
            pipeline,
            callbacks,
            expected_callbacks_data,
            hyperparams_space,
            hyperparams,
            expected_processed_outputs
    ):
        self.pipeline = pipeline
        self.callbacks = callbacks
        self.expected_callbacks_data = expected_callbacks_data
        self.hyperparams = hyperparams
        self.hyperparams_space = hyperparams_space
        self.expected_processed_outputs = expected_processed_outputs


def create_test_case_single_step_choosen():
    a_callback = TapeCallbackFunction()
    b_callback = TapeCallbackFunction()

    return ChooseStepsTestCase(
        pipeline=Pipeline([
            ChooseOneOrManyStepsOf([
                ('a', TransformCallbackStep(a_callback, transform_function=lambda di: di * 2)),
                ('b', TransformCallbackStep(b_callback, transform_function=lambda di: di * 2))
            ]),
        ]),
        callbacks=[a_callback, b_callback],
        expected_callbacks_data=[
            DATA_INPUTS,
            []
        ],
        hyperparams={
            'ChooseOneOrManyStepsOf__a__enabled': True,
            'ChooseOneOrManyStepsOf__b__enabled': False
        },
        hyperparams_space={
            'ChooseOneOrManyStepsOf__a__enabled': Boolean(),
            'ChooseOneOrManyStepsOf__b__enabled': Boolean()
        },
        expected_processed_outputs=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    )


def create_test_case_multiple_steps_choosen():
    a_callback = TapeCallbackFunction()
    b_callback = TapeCallbackFunction()

    return ChooseStepsTestCase(
        pipeline=Pipeline([
            ChooseOneOrManyStepsOf([
                ('a', TransformCallbackStep(a_callback, transform_function=lambda di: di * 2)),
                ('b', TransformCallbackStep(b_callback, transform_function=lambda di: di * 2))
            ]),
        ]),
        callbacks=[a_callback, b_callback],
        expected_callbacks_data=[DATA_INPUTS, DATA_INPUTS],
        hyperparams={
            'ChooseOneOrManyStepsOf__a__enabled': True,
            'ChooseOneOrManyStepsOf__b__enabled': True
        },
        hyperparams_space={
            'ChooseOneOrManyStepsOf__a__enabled': Boolean(),
            'ChooseOneOrManyStepsOf__b__enabled': Boolean()
        },
        expected_processed_outputs=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    )


def create_test_case_invalid_step_choosen():
    a_callback = TapeCallbackFunction()
    b_callback = TapeCallbackFunction()

    return ChooseStepsTestCase(
        pipeline=Pipeline([
            ChooseOneOrManyStepsOf([
                ('a', TransformCallbackStep(a_callback, transform_function=lambda di: di * 2)),
                ('b', TransformCallbackStep(b_callback, transform_function=lambda di: di * 2))
            ]),
        ]),
        callbacks=[a_callback, b_callback],
        expected_callbacks_data=[DATA_INPUTS, DATA_INPUTS],
        hyperparams={
            'ChooseOneOrManyStepsOf__c__enabled': True,
            'ChooseOneOrManyStepsOf__b__enabled': False
        },
        hyperparams_space={
            'ChooseOneOrManyStepsOf__a__enabled': Boolean(),
            'ChooseOneOrManyStepsOf__b__enabled': Boolean()
        },
        expected_processed_outputs=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    )


def create_test_case_invalid_step_not_choosen():
    a_callback = TapeCallbackFunction()
    b_callback = TapeCallbackFunction()

    return ChooseStepsTestCase(
        pipeline=Pipeline([
            ChooseOneOrManyStepsOf([
                ('a', TransformCallbackStep(a_callback, transform_function=lambda di: di * 2)),
                ('b', TransformCallbackStep(b_callback, transform_function=lambda di: di * 2))
            ]),
        ]),
        callbacks=[a_callback, b_callback],
        expected_callbacks_data=[DATA_INPUTS, DATA_INPUTS],
        hyperparams={
            'ChooseOneOrManyStepsOf__c__enabled': False,
            'ChooseOneOrManyStepsOf__b__enabled': False
        },
        hyperparams_space={
            'ChooseOneOrManyStepsOf__a__enabled': Boolean(),
            'ChooseOneOrManyStepsOf__b__enabled': Boolean()
        },
        expected_processed_outputs=np.array(range(10))
    )


@pytest.mark.parametrize('test_case', [
    create_test_case_single_step_choosen(),
    create_test_case_multiple_steps_choosen()
])
def test_choose_one_or_many_step_of_transform_should_choose_step(
        test_case: ChooseStepsTestCase):
    p = test_case.pipeline
    p.set_hyperparams_space(HyperparameterSpace(test_case.hyperparams_space))
    p.set_hyperparams(test_case.hyperparams)

    outputs = p.transform(DATA_INPUTS)

    assert np.array_equal(outputs, test_case.expected_processed_outputs)
    assert_callback_data_is_as_expected(test_case)


def assert_callback_data_is_as_expected(test_case):
    for callback, expected_callback_data in zip(test_case.callbacks, test_case.expected_callbacks_data):
        if len(callback.data) > 0:
            assert np.array_equal(
                np.array(callback.data[0]),
                expected_callback_data
            )
        else:
            assert np.array_equal(
                np.array([]),
                np.array(expected_callback_data)
            )


def create_test_case_fit_transform_single_step_choosen():
    a_callback = TapeCallbackFunction()
    b_callback = TapeCallbackFunction()
    c_callback = TapeCallbackFunction()
    d_callback = TapeCallbackFunction()

    return ChooseStepsTestCase(
        pipeline=Pipeline([
            ChooseOneOrManyStepsOf([
                ('a', FitTransformCallbackStep(a_callback, c_callback, transform_function=lambda di: di * 2)),
                ('b', FitTransformCallbackStep(b_callback, d_callback, transform_function=lambda di: di * 2))
            ]),
        ]),
        callbacks=[a_callback, c_callback, b_callback, d_callback],
        expected_callbacks_data=[
            DATA_INPUTS,
            (DATA_INPUTS, EXPECTED_OUTPUTS),
            [],
            []
        ],
        hyperparams={
            'ChooseOneOrManyStepsOf__a__enabled': True,
            'ChooseOneOrManyStepsOf__b__enabled': False
        },
        hyperparams_space={
            'ChooseOneOrManyStepsOf__a__enabled': Boolean(),
            'ChooseOneOrManyStepsOf__b__enabled': Boolean()
        },
        expected_processed_outputs=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    )


def create_test_case_fit_transform_multiple_steps_choosen():
    a_callback = TapeCallbackFunction()
    b_callback = TapeCallbackFunction()
    c_callback = TapeCallbackFunction()
    d_callback = TapeCallbackFunction()

    return ChooseStepsTestCase(
        pipeline=Pipeline([
            ChooseOneOrManyStepsOf([
                ('a', FitTransformCallbackStep(a_callback, c_callback, transform_function=lambda di: di * 2)),
                ('b', FitTransformCallbackStep(b_callback, d_callback, transform_function=lambda di: di * 2))
            ]),
        ]),
        callbacks=[a_callback, c_callback, b_callback, d_callback],
        expected_callbacks_data=[
            DATA_INPUTS,
            (DATA_INPUTS, EXPECTED_OUTPUTS),
            DATA_INPUTS,
            (DATA_INPUTS, EXPECTED_OUTPUTS)
        ],
        hyperparams={
            'ChooseOneOrManyStepsOf__a__enabled': True,
            'ChooseOneOrManyStepsOf__b__enabled': True
        },
        hyperparams_space={
            'ChooseOneOrManyStepsOf__a__enabled': Boolean(),
            'ChooseOneOrManyStepsOf__b__enabled': Boolean()
        },
        expected_processed_outputs=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    )

def create_test_case_fit_single_step_choosen():
    a_callback = TapeCallbackFunction()
    b_callback = TapeCallbackFunction()
    c_callback = TapeCallbackFunction()
    d_callback = TapeCallbackFunction()

    return ChooseStepsTestCase(
        pipeline=Pipeline([
            ChooseOneOrManyStepsOf([
                ('a', FitTransformCallbackStep(a_callback, c_callback, transform_function=lambda di: di * 2)),
                ('b', FitTransformCallbackStep(b_callback, d_callback, transform_function=lambda di: di * 2))
            ]),
        ]),
        callbacks=[a_callback, c_callback, b_callback, d_callback],
        expected_callbacks_data=[
            [],
            (DATA_INPUTS, EXPECTED_OUTPUTS),
            [],
            []
        ],
        hyperparams={
            'ChooseOneOrManyStepsOf__a__enabled': True,
            'ChooseOneOrManyStepsOf__b__enabled': False
        },
        hyperparams_space={
            'ChooseOneOrManyStepsOf__a__enabled': Boolean(),
            'ChooseOneOrManyStepsOf__b__enabled': Boolean()
        },
        expected_processed_outputs=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    )

def create_test_case_fit_multiple_steps_choosen():
    a_callback = TapeCallbackFunction()
    b_callback = TapeCallbackFunction()
    c_callback = TapeCallbackFunction()
    d_callback = TapeCallbackFunction()

    return ChooseStepsTestCase(
        pipeline=Pipeline([
            ChooseOneOrManyStepsOf([
                ('a', FitTransformCallbackStep(a_callback, c_callback, transform_function=lambda di: di * 2)),
                ('b', FitTransformCallbackStep(b_callback, d_callback, transform_function=lambda di: di * 2))
            ]),
        ]),
        callbacks=[a_callback, c_callback, b_callback, d_callback],
        expected_callbacks_data=[
            [],
            (DATA_INPUTS, EXPECTED_OUTPUTS),
            [],
            (DATA_INPUTS, EXPECTED_OUTPUTS)
        ],
        hyperparams={
            'ChooseOneOrManyStepsOf__a__enabled': True,
            'ChooseOneOrManyStepsOf__b__enabled': True
        },
        hyperparams_space={
            'ChooseOneOrManyStepsOf__a__enabled': Boolean(),
            'ChooseOneOrManyStepsOf__b__enabled': Boolean()
        },
        expected_processed_outputs=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    )


@pytest.mark.parametrize('test_case', [
    create_test_case_fit_transform_single_step_choosen(),
    create_test_case_fit_transform_multiple_steps_choosen()
])
def test_choose_one_or_many_step_of_fit_transform_should_choose_step(
        test_case: ChooseStepsTestCase):
    p = test_case.pipeline
    p.set_hyperparams_space(test_case.hyperparams_space)
    p.set_hyperparams(test_case.hyperparams)

    p, outputs = p.fit_transform(DATA_INPUTS, EXPECTED_OUTPUTS)

    assert np.array_equal(outputs, test_case.expected_processed_outputs)
    assert_callback_data_is_as_expected(test_case)


@pytest.mark.parametrize('test_case', [
    create_test_case_fit_single_step_choosen(),
    create_test_case_fit_multiple_steps_choosen()
])
def test_choose_one_or_many_step_of_fit_should_choose_step(
        test_case: ChooseStepsTestCase):
    p = test_case.pipeline
    p.set_hyperparams_space(test_case.hyperparams_space)
    p.set_hyperparams(test_case.hyperparams)

    p = p.fit(DATA_INPUTS, EXPECTED_OUTPUTS)

    assert_callback_data_is_as_expected(test_case)


@pytest.mark.parametrize('test_case', [
    create_test_case_invalid_step_choosen(),
    create_test_case_invalid_step_not_choosen()
])
def test_choose_one_or_many_step_of_should_throw_exception_on_invalid_chosen_step(
        test_case: ChooseStepsTestCase):
    with pytest.raises(ValueError):
        p = test_case.pipeline
        p.set_hyperparams_space(test_case.hyperparams_space)
        p.set_hyperparams(test_case.hyperparams)
