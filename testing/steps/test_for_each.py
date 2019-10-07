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
    data_inputs = [[0, 1], [1, 2]]
    expected_outputs = [[2, 3], [4, 5]]

    p = p.fit(data_inputs, expected_outputs)

    assert isinstance(p, Pipeline)
    assert tape.get_name_tape() == ["1", "1", "2", "2"]
    assert tape.data == [([0, 1], [2, 3]), ([1, 2], [4, 5]), ([0, 1], [2, 3]), ([1, 2], [4, 5])]


def test_fit_transform_should_fit_transform_all_steps_for_each_data_inputs_expected_outputs():
    tape = TapeCallbackFunction()
    p = Pipeline([
        ForEachDataInputs([
            FitTransformCallbackStep(tape.callback, ["1"]),
            FitTransformCallbackStep(tape.callback, ["2"]),
        ])
    ])

    data_inputs = [[0, 1], [1, 2]]
    expected_outputs = [[2, 3], [4, 5]]

    p, outputs = p.fit_transform(data_inputs, expected_outputs)

    assert tape.get_name_tape() == ["1", "1", "2", "2"]
    assert tape.data == [([0, 1], [2, 3]), ([1, 2], [4, 5]), ([0, 1], [2, 3]), ([1, 2], [4, 5])]


def test_transform_should_transform_all_steps_for_each_data_inputs_expected_outputs():
    tape = TapeCallbackFunction()
    p = Pipeline([
        ForEachDataInputs([
            TransformCallbackStep(tape.callback, ["1"]),
            TransformCallbackStep(tape.callback, ["2"]),
        ])
    ])
    data_inputs = [[0, 1], [1, 2]]

    outputs = p.transform(data_inputs)

    assert tape.get_name_tape() == ["1", "1", "2", "2"]
