from neuraxle.pipeline import Pipeline
from neuraxle.steps.loop import ForEachDataInput
from neuraxle.steps.misc import TransformCallbackStep, TapeCallbackFunction, FitCallbackStep, \
    FitTransformCallbackStep


def test_fit_for_each_should_fit_all_steps_for_each_data_inputs_expected_outputs():
    tape = TapeCallbackFunction()
    p = Pipeline([
        ForEachDataInput(Pipeline([
            FitCallbackStep(tape.callback, ["1"]),
            FitCallbackStep(tape.callback, ["2"]),
        ]))
    ])
    data_inputs = [[0, 1], [1, 2]]
    expected_outputs = [[2, 3], [4, 5]]

    p = p.fit(data_inputs, expected_outputs)

    assert isinstance(p, Pipeline)
    assert tape.get_name_tape() == ["1", "2", "1", "2"]
    assert tape.data == [([0, 1], [2, 3]), ([0, 1], [2, 3]), ([1, 2], [4, 5]), ([1, 2], [4, 5])]


def test_fit_transform_should_fit_transform_all_steps_for_each_data_inputs_expected_outputs():
    tape = TapeCallbackFunction()
    tape_fit = TapeCallbackFunction()
    p = Pipeline([
        ForEachDataInput(Pipeline([
            FitTransformCallbackStep(tape.callback, tape_fit, ["1"]),
            FitTransformCallbackStep(tape.callback, tape_fit, ["2"]),
        ]))
    ])
    data_inputs = [[0, 1], [1, 2]]
    expected_outputs = [[2, 3], [4, 5]]

    p, outputs = p.fit_transform(data_inputs, expected_outputs)

    assert tape.get_name_tape() == ["1", "2", "1", "2"]
    assert tape_fit.get_name_tape() == ["1", "2", "1", "2"]
    assert tape_fit.data == [([0, 1], [2, 3]), ([0, 1], [2, 3]), ([1, 2], [4, 5]), ([1, 2], [4, 5])]


def test_transform_should_transform_all_steps_for_each_data_inputs_expected_outputs():
    tape = TapeCallbackFunction()
    p = Pipeline([
        ForEachDataInput(Pipeline([
            TransformCallbackStep(tape.callback, ["1"]),
            TransformCallbackStep(tape.callback, ["2"]),
        ]))
    ])
    data_inputs = [[0, 1], [1, 2]]

    outputs = p.transform(data_inputs)

    assert tape.get_name_tape() == ["1", "2", "1", "2"]
