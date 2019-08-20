"""
Neuraxle's Pipeline Classes
====================================
This is the core of Neuraxle's pipelines. You can chain steps to call them one after an other.

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

from copy import copy
from typing import Any

from neuraxle.base import BaseStep, BasePipelineRunner, TruncableSteps, BaseBarrier, NamedTupleList
from neuraxle.runners import CheckpointPipelineRunner


class DataObject:
    def __init__(self, i, x):
        self.i = i
        self.x = x

        def __hash__(self):
            return hash((self.i, self.x))

class Pipeline(TruncableSteps):

    def __init__(self,
                 steps: NamedTupleList,
                 pipeline_runner: BasePipelineRunner = CheckpointPipelineRunner()):
        BaseStep.__init__(self)
        TruncableSteps.__init__(self, steps)
        self.pipeline_runner: BasePipelineRunner = pipeline_runner

    def fit(self, data_inputs, expected_outputs=None) -> 'Pipeline':
        # TODO: overwrite steps?
        self.pipeline_runner.set_steps(self.steps_as_tuple).fit(data_inputs, expected_outputs)
        return self

    def transform(self, data_inputs):
        return self.pipeline_runner.transform(data_inputs)

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('Pipeline', Any):
        self.pipeline_runner.set_steps(self.steps_as_tuple)

        self.pipeline_runner, out = self.pipeline_runner.fit_transform(data_inputs, expected_outputs)
        return self, out

    def inverse_transform(self, processed_outputs):
        if self.transform != self.inverse_transform:
            raise BrokenPipeError("Don't call inverse_transform on a pipeline before having mutated it inversely or "
                                  "before having called the `.reverse()` or `reversed(.)` on it.")

        reversed_steps_as_tuple = list(reversed(self.steps_as_tuple))
        return self.pipeline_runner.set_steps(reversed_steps_as_tuple).transform(processed_outputs)
