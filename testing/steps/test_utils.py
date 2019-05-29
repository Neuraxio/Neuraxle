import numpy as np

from neuraxle.pipeline import Pipeline
from neuraxle.steps.util import TapeCallbackFunction, TransformCallbackStep
from neuraxle.union import Identity, AddFeatures


def test_tape_callback():
    expected_tape = ["1", "2", "3", "a", "b", "4"]
    tape = TapeCallbackFunction()

    p = Pipeline([
        Identity(),
        TransformCallbackStep(tape.callback, ["1"]),
        TransformCallbackStep(tape.callback, ["2"]),
        TransformCallbackStep(tape.callback, ["3"]),
        AddFeatures([
            TransformCallbackStep(tape.callback, ["a"]),
            TransformCallbackStep(tape.callback, ["b"]),
        ]),
        TransformCallbackStep(tape.callback, ["4"]),
        Identity()
    ])
    p.fit_transform(np.ones((1, 1)))

    assert expected_tape == tape.get_name_tape()
