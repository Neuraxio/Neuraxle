"""
Tests for Util Steps
========================================

..
    Copyright 2019, Neuraxio Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""

import copy

import numpy as np

from neuraxle.pipeline import Pipeline
from neuraxle.steps.util import TapeCallbackFunction, TransformCallbackStep, StepClonerForEachDataInput
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

    assert tape.get_name_tape() == expected_tape


def test_step_cloner():
    tape = TapeCallbackFunction()
    data = [[1], [2], [3]]

    sc = StepClonerForEachDataInput(TransformCallbackStep(tape, ["-"]), copy_op=copy.copy)
    sc.fit_transform(data)

    print(tape)
    print(tape.get_name_tape())
    print(tape.get_data())
    assert tape.get_data() == data
    assert tape.get_name_tape() == ["-"] * 3
