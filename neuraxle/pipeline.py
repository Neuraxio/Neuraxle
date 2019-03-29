# Copyright 2019, The NeurAxle Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict

from neuraxle.base import BaseStep, PipelineRunner, TruncableSteps
from neuraxle.typing import NamedTupleList


class BlockPipelineRunner(PipelineRunner):

    def fit_transform(self, data_inputs, expected_outputs=None):
        for step_name, step in self.steps_as_tuple[:-1]:
            data_inputs = step.fit_transform(data_inputs, None)
        processed_outputs = self.steps_as_tuple[-1][-1].fit_transform(data_inputs, expected_outputs)
        return processed_outputs

    def fit(self, data_inputs, expected_outputs=None):
        for step_name, step in self.steps_as_tuple[:-1]:
            data_inputs = step.fit_transform(data_inputs, None)
        processed_outputs = self.steps_as_tuple[-1][-1].fit(data_inputs, expected_outputs)
        return processed_outputs

    def transform(self, data_inputs):
        for step_name, step in self.steps_as_tuple:
            data_inputs = step.transform(data_inputs)
        processed_outputs = data_inputs
        return processed_outputs


class Pipeline(TruncableSteps):

    def __init__(
            self,
            steps: NamedTupleList,
            pipeline_runner: PipelineRunner = BlockPipelineRunner(),
    ):
        BaseStep.__init__(self)
        TruncableSteps.__init__(self, steps)
        self.pipeline_runner: PipelineRunner = pipeline_runner

    def get_hyperparams_space(self):
        all_hyperparams = OrderedDict()
        for step_name, step in self.steps_as_tuple:
            all_hyperparams.update(step.get_hyperparams_space())
        return all_hyperparams

    def fit_transform(self, data_inputs, expected_outputs=None):
        return self.pipeline_runner.set_steps(self.steps_as_tuple).fit_transform(data_inputs, expected_outputs)

    def fit(self, data_inputs, expected_outputs=None):
        self.pipeline_runner.set_steps(self.steps_as_tuple).fit(data_inputs, expected_outputs)
        return self

    def transform(self, data_inputs):
        return self.pipeline_runner.transform(data_inputs)
