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

from neuraxle.base import BaseStep, PipelineRunner, NamedTupleList


class BlockPipelineRunner(PipelineRunner):

    def fit_transform(self, data_inputs, expected_outputs=None):
        for step_name, step in self.named_pipeline_steps[:-1]:
            data_inputs = step.fit_transform(data_inputs, None)
        processed_outputs = self.named_pipeline_steps[-1][-1].fit_transform(data_inputs, expected_outputs)
        return processed_outputs

    def fit(self, data_inputs, expected_outputs=None):
        for step_name, step in self.named_pipeline_steps[:-1]:
            data_inputs = step.fit_transform(data_inputs, None)
        processed_outputs = self.named_pipeline_steps[-1][-1].fit(data_inputs, expected_outputs)
        return processed_outputs

    def transform(self, data_inputs):
        for step_name, step in self.named_pipeline_steps:
            data_inputs = step.transform(data_inputs)
        processed_outputs = data_inputs
        return processed_outputs


class Pipeline(BaseStep):

    def __init__(
            self,
            named_pipeline_steps: NamedTupleList,
            pipeline_runner: PipelineRunner = BlockPipelineRunner(),
    ):
        super().__init__()
        self.named_pipeline_steps: NamedTupleList = named_pipeline_steps
        self.pipeline_runner: PipelineRunner = pipeline_runner

    def get_default_hyperparams(self):
        all_hyperparams = OrderedDict()
        for step_name, step in self.named_pipeline_steps:
            all_hyperparams.update(step.get_default_hyperparams())
        return all_hyperparams

    def fit_transform(self, data_inputs, expected_outputs):
        return self.pipeline_runner.set_steps(self.named_pipeline_steps).fit_transform(data_inputs, expected_outputs)

    def fit(self, data_inputs, expected_outputs):
        self.pipeline_runner.set_steps(self.named_pipeline_steps).fit(data_inputs)
        return self

    def transform(self, data_inputs):
        return self.pipeline_runner.transform(data_inputs)
