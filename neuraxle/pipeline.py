"""
Neuraxle's Pipeline Classes
====================================
This is the core of Neuraxle's pipelines. You can chain steps to call them one after an other.

..
    Copyright 2019, The Neuraxle Authors

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


class BlockPipelineRunner(BasePipelineRunner):

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BlockPipelineRunner', Any):
        new_steps_as_tuple: NamedTupleList = []
        for step_name, step in self.steps_as_tuple:
            step, data_inputs = step.fit_transform(data_inputs, expected_outputs)
            new_steps_as_tuple.append((step_name, step))
        self.steps_as_tuple = new_steps_as_tuple
        processed_outputs = data_inputs
        return self, processed_outputs

    def fit(self, data_inputs, expected_outputs=None) -> 'BlockPipelineRunner':
        # TODO: don't do last transform.
        new_steps_as_tuple: NamedTupleList = []
        for step_name, step in self.steps_as_tuple:
            step, data_inputs = step.fit_transform(data_inputs, expected_outputs)
            new_steps_as_tuple.append((step_name, step))
        self.steps_as_tuple = new_steps_as_tuple
        return self

    def transform(self, data_inputs):
        for step_name, step in self.steps_as_tuple:
            data_inputs = step.transform(data_inputs)
        processed_outputs = data_inputs
        return processed_outputs


class DataObject:
    def __init__(self, i, x):
        self.i = i
        self.x = x

        def __hash__(self):
            return hash((self.i, self.x))


class ResumablePipelineRunner(BasePipelineRunner):
    # TODO: this is code to do or to delete.

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('ResumablePipelineRunner', Any):
        for step_name, step in self.steps_as_tuple[:-1]:
            # TODO: overwrite step's self return value on fitting.
            data_inputs = step.fit_transform(data_inputs, expected_outputs)
        processed_outputs = self.steps_as_tuple[-1][-1].fit_transform(data_inputs, expected_outputs)
        return processed_outputs

    def fit(self, data_inputs, expected_outputs=None) -> 'ResumablePipelineRunner':
        for step_name, step in self.steps_as_tuple[:-1]:
            # TODO: overwrite step's self return value on fitting.
            data_inputs = step.fit_transform(data_inputs, expected_outputs)
        processed_outputs = self.steps_as_tuple[-1][-1].fit(data_inputs, expected_outputs)
        return processed_outputs

    def transform(self, data_inputs):

        dos = [DataObject(i, di) for i, di in enumerate(data_inputs)]

        # TODO: review this and do it in fit also.
        steps = self.resume_transform_entry_point(dos, self.steps_as_tuple)

        for step_name, step in steps:
            data_inputs = step.transform(data_inputs)
        processed_outputs = data_inputs
        return processed_outputs

    def resume_transform_entry_point(self, dos, steps_as_tuple):
        steps = copy(steps_as_tuple)
        for step_name, step in reversed(steps_as_tuple):

            # TODO: maybe change to: if hasattr(step, 'should_resume'):
            if isinstance(step, BaseBarrier) or isinstance(step, Pipeline):

                if step.should_resume(dos):
                    if isinstance(step, Pipeline):  # TODO: maybe change to: if hasattr(step, 'trim_for_resume'):
                        step = step.trim_for_resume(dos)
                        steps[step_name] = step
                    steps = steps_as_tuple[step_name:]  # TODO: fix get
                    return steps


class Pipeline(TruncableSteps):

    def __init__(
            self,
            steps: NamedTupleList,
            pipeline_runner: BasePipelineRunner = BlockPipelineRunner(),
    ):
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
