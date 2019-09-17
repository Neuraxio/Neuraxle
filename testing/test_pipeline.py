"""
Tests for Pipelines
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

import numpy as np
import pytest

from neuraxle.base import BaseStep, RangeHasher
from neuraxle.hyperparams.distributions import RandInt, LogUniform
from neuraxle.hyperparams.space import nested_dict_to_flat, HyperparameterSpace
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import NumpyTranspose
from neuraxle.steps.sklearn import SKLearnWrapper
from neuraxle.steps.util import TransformCallbackStep, TapeCallbackFunction
from neuraxle.union import Identity, AddFeatures, ModelStacking

AN_INPUT = "I am an input"
AN_EXPECTED_OUTPUT = "I am an expected output"


class SomeStep(BaseStep):
    def __init__(self, hyperparams_space: HyperparameterSpace = None, hasher=RangeHasher()):
        super().__init__(hyperparams=None, hyperparams_space=hyperparams_space, hasher=hasher)

    def fit_one(self, data_input, expected_output=None) -> 'SomeStep':
        return self

    def transform_one(self, data_input):
        return AN_EXPECTED_OUTPUT


steps_lists = [
    [("just_one_step", SomeStep())],
    [
        ("some_step_1", SomeStep()),
        ("some_step_2", SomeStep()),
        ("some_step_3", SomeStep())
    ]
]


@pytest.mark.parametrize("steps_list", steps_lists)
def test_pipeline_fit_transform(steps_list):
    data_input_ = [AN_INPUT]
    expected_output_ = [AN_EXPECTED_OUTPUT]
    p = Pipeline(steps_list)

    p, result = p.fit_transform(data_input_, expected_output_)

    assert tuple(result) == tuple(expected_output_)


@pytest.mark.parametrize("steps_list", steps_lists)
def test_pipeline_fit_then_transform(steps_list):
    data_input_ = [AN_INPUT]
    expected_output_ = [AN_EXPECTED_OUTPUT]
    p = Pipeline(steps_list)

    p = p.fit(data_input_, expected_output_)
    result = p.transform(data_input_)

    assert tuple(result) == tuple(expected_output_)


def test_pipeline_slicing_before():
    p = Pipeline([
        ("a", SomeStep()),
        ("b", SomeStep()),
        ("c", SomeStep())
    ])

    r = p["b":]

    assert "a" not in r
    assert "b" in r
    assert "c" in r


def test_pipeline_slicing_after():
    p = Pipeline([
        ("a", SomeStep()),
        ("b", SomeStep()),
        ("c", SomeStep())
    ])

    r = p[:"c"]

    assert "a" in r
    assert "b" in r
    assert "c" not in r


def test_pipeline_slicing_both():
    p = Pipeline([
        ("a", SomeStep()),
        ("b", SomeStep()),
        ("c", SomeStep())
    ])

    r = p["b":"c"]

    assert "a" not in r
    assert "b" in r
    assert "c" not in r


def test_pipeline_set_one_hyperparam_level_one_flat():
    p = Pipeline([
        ("a", SomeStep()),
        ("b", SomeStep()),
        ("c", SomeStep())
    ])

    p.set_hyperparams({
        "a__learning_rate": 7
    })

    assert p["a"].hyperparams["learning_rate"] == 7
    assert p["b"].hyperparams == dict()
    assert p["c"].hyperparams == dict()


def test_pipeline_set_one_hyperparam_level_one_dict():
    p = Pipeline([
        ("a", SomeStep()),
        ("b", SomeStep()),
        ("c", SomeStep())
    ])

    p.set_hyperparams({
        "b": {
            "learning_rate": 7
        }
    })

    assert p["a"].hyperparams == dict()
    assert p["b"].hyperparams["learning_rate"] == 7
    assert p["c"].hyperparams == dict()


def test_pipeline_set_one_hyperparam_level_two_flat():
    p = Pipeline([
        ("a", SomeStep()),
        ("b", Pipeline([
            ("a", SomeStep()),
            ("b", SomeStep()),
            ("c", SomeStep())
        ])),
        ("c", SomeStep())
    ])

    p.set_hyperparams({
        "b__a__learning_rate": 7
    })
    print(p.get_hyperparams())

    assert p["b"]["a"].hyperparams["learning_rate"] == 7
    assert p["b"]["c"].hyperparams == dict()
    assert p["b"].hyperparams == dict()
    assert p["c"].hyperparams == dict()


def test_pipeline_set_one_hyperparam_level_two_dict():
    p = Pipeline([
        ("a", SomeStep()),
        ("b", Pipeline([
            ("a", SomeStep()),
            ("b", SomeStep()),
            ("c", SomeStep())
        ])),
        ("c", SomeStep())
    ])

    p.set_hyperparams({
        "b": {
            "a": {
                "learning_rate": 7
            },
            "learning_rate": 9
        }
    })
    print(p.get_hyperparams())

    assert p["b"]["a"].hyperparams["learning_rate"] == 7
    assert p["b"]["c"].hyperparams == dict()
    assert p["b"].hyperparams["learning_rate"] == 9
    assert p["c"].hyperparams == dict()


def test_pipeline_tosklearn():
    import sklearn.pipeline
    the_step = SomeStep()
    step_to_check = the_step.tosklearn()

    p = Pipeline([
        ("a", SomeStep()),
        ("b", SKLearnWrapper(sklearn.pipeline.Pipeline([
            ("a", sklearn.pipeline.Pipeline([
                ('z', step_to_check)
            ])),
            ("b", SomeStep().tosklearn()),
            ("c", SomeStep().tosklearn())
        ]), return_all_sklearn_default_params_on_get=True)),
        ("c", SomeStep())
    ])

    # assert False
    p.set_hyperparams({
        "b": {
            "a__z__learning_rate": 7,
            "b__learning_rate": 9
        }
    })
    assert the_step.get_hyperparams()["learning_rate"] == 7

    p = p.tosklearn()
    p = sklearn.pipeline.Pipeline([('sk', p)])

    p.set_params(**{"sk__b__a__z__learning_rate": 11})
    assert p.named_steps["sk"].p["b"].wrapped_sklearn_predictor.named_steps["a"]["z"]["learning_rate"] == 11

    p.set_params(**nested_dict_to_flat({
        "sk__b": {
            "a__z__learning_rate": 12,
            "b__learning_rate": 9
        }
    }))
    # p.set_params(**{"sk__b__a__z__learning_rate": 12})
    assert p.named_steps["sk"].p["b"].wrapped_sklearn_predictor.named_steps["a"]["z"]["learning_rate"] == 12
    print(the_step.get_hyperparams())
    # assert the_step.get_hyperparams()["learning_rate"] == 12  # TODO: debug why wouldn't this work


def test_pipeline_simple_mutate_inverse_transform():
    expected_tape = ["1", "2", "3", "4", "4", "3", "2", "1"]
    tape = TapeCallbackFunction()

    p = Pipeline([
        Identity(),
        TransformCallbackStep(tape.callback, ["1"]),
        TransformCallbackStep(tape.callback, ["2"]),
        TransformCallbackStep(tape.callback, ["3"]),
        TransformCallbackStep(tape.callback, ["4"]),
        Identity()
    ])

    p, _ = p.fit_transform(np.ones((1, 1)))

    print("[mutating]")
    p = p.mutate(new_method="inverse_transform", method_to_assign_to="transform")

    p.transform(np.ones((1, 1)))

    assert expected_tape == tape.get_name_tape()


def test_pipeline_nested_mutate_inverse_transform():
    expected_tape = ["1", "2", "3", "4", "5", "6", "7", "7", "6", "5", "4", "3", "2", "1"]
    tape = TapeCallbackFunction()

    p = Pipeline([
        Identity(),
        TransformCallbackStep(tape.callback, ["1"]),
        TransformCallbackStep(tape.callback, ["2"]),
        Pipeline([
            Identity(),
            TransformCallbackStep(tape.callback, ["3"]),
            TransformCallbackStep(tape.callback, ["4"]),
            TransformCallbackStep(tape.callback, ["5"]),
            Identity()
        ]),
        TransformCallbackStep(tape.callback, ["6"]),
        TransformCallbackStep(tape.callback, ["7"]),
        Identity()
    ])

    p, _ = p.fit_transform(np.ones((1, 1)))  # will add range(1, 8) to tape.

    print("[mutating]")
    p = p.mutate(new_method="inverse_transform", method_to_assign_to="transform")

    p.transform(np.ones((1, 1)))  # will add reversed(range(1, 8)) to tape.

    print(expected_tape)
    print(tape.get_name_tape())
    assert expected_tape == tape.get_name_tape()


def test_pipeline_nested_mutate_inverse_transform_without_identities():
    """
    This test was required for a strange bug at the border of the pipelines
    that happened when the identities were not used.
    """
    expected_tape = ["1", "2", "3", "4", "5", "6", "7", "7", "6", "5", "4", "3", "2", "1"]
    tape = TapeCallbackFunction()

    p = Pipeline([
        TransformCallbackStep(tape.callback, ["1"]),
        TransformCallbackStep(tape.callback, ["2"]),
        Pipeline([
            TransformCallbackStep(tape.callback, ["3"]),
            TransformCallbackStep(tape.callback, ["4"]),
            TransformCallbackStep(tape.callback, ["5"]),
        ]),
        TransformCallbackStep(tape.callback, ["6"]),
        TransformCallbackStep(tape.callback, ["7"]),
    ])

    p, _ = p.fit_transform(np.ones((1, 1)))  # will add range(1, 8) to tape.

    print("[mutating, inversing, and calling each inverse_transform]")
    reversed(p).transform(np.ones((1, 1)))  # will add reversed(range(1, 8)) to tape, calling inverse_transforms.

    print(expected_tape)
    print(tape.get_name_tape())
    assert expected_tape == tape.get_name_tape()


def test_hyperparam_space():
    p = Pipeline([
        AddFeatures([
            SomeStep(hyperparams_space=HyperparameterSpace({"n_components": RandInt(1, 5)})),
            SomeStep(hyperparams_space=HyperparameterSpace({"n_components": RandInt(1, 5)}))
        ]),
        ModelStacking([
            SomeStep(hyperparams_space=HyperparameterSpace({"n_estimators": RandInt(1, 1000)})),
            SomeStep(hyperparams_space=HyperparameterSpace({"n_estimators": RandInt(1, 1000)})),
            SomeStep(hyperparams_space=HyperparameterSpace({"max_depth": RandInt(1, 100)})),
            SomeStep(hyperparams_space=HyperparameterSpace({"max_depth": RandInt(1, 100)}))
        ],
            joiner=NumpyTranspose(),
            judge=SomeStep(hyperparams_space=HyperparameterSpace({"alpha": LogUniform(0.1, 10.0)}))
        )
    ])

    rvsed = p.get_hyperparams_space()
    p.set_hyperparams(rvsed)

    hyperparams = p.get_hyperparams()

    assert 'AddFeatures__SomeStep1__n_components' in hyperparams.keys()
    assert 'AddFeatures__SomeStep__n_components' in hyperparams.keys()
    assert 'AddFeatures__SomeStep1__n_components' in hyperparams.keys()
    assert 'ModelStacking__SomeStep__n_estimators' in hyperparams.keys()
    assert 'ModelStacking__SomeStep1__n_estimators' in hyperparams.keys()
    assert 'ModelStacking__SomeStep2__max_depth' in hyperparams.keys()
    assert 'ModelStacking__SomeStep3__max_depth' in hyperparams.keys()
