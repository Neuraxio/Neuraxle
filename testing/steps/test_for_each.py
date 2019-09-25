from neuraxle.pipeline import Pipeline
from neuraxle.steps.util import TransformCallbackStep, ForEachDataInputs, TapeCallbackFunction, FitCallbackStep, \
    FitTransformCallbackStep


def test_fit_for_each_should_fit_all_steps_for_each_data_inputs_expected_outputs():
    tape = TapeCallbackFunction()
    p = Pipeline([
        ForEachDataInputs([
            FitCallbackStep(tape.callback, ["1"]),
            FitCallbackStep(tape.callback, ["2"]),
        ])
    ])

    p = p.fit([[0, 1], [1, 2]])

    assert tape.get_name_tape() == ["1", "1", "2", "2"]


def test_fit_transform_should_fit_transform_all_steps_for_each_data_inputs_expected_outputs():
    tape = TapeCallbackFunction()
    p = Pipeline([
        ForEachDataInputs([
            FitTransformCallbackStep(tape.callback, ["1"]),
            FitTransformCallbackStep(tape.callback, ["2"]),
        ])
    ])

    p, outputs = p.fit_transform([[0, 1], [1, 2]])

    assert tape.get_name_tape() == ["1", "1", "2", "2"]


def test_transform_should_transform_all_steps_for_each_data_inputs_expected_outputs():
    tape = TapeCallbackFunction()
    p = Pipeline([
        ForEachDataInputs([
            TransformCallbackStep(tape.callback, ["1"]),
            TransformCallbackStep(tape.callback, ["2"]),
        ])
    ])

    outputs = p.transform([[0, 1], [1, 2]])

    assert tape.get_name_tape() == ["1", "1", "2", "2"]
