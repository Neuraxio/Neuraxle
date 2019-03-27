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

from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseStep(ABC):

    def __init__(self, **hyperparams):
        self.hyperparams: dict = hyperparams

    def set_hyperparams(self, **hyperparams):
        self.hyperparams: dict = hyperparams

    def get_hyperparams(self) -> dict:
        return self.hyperparams

    def get_default_hyperparams(self) -> dict:
        return dict()

    def fit_transform(self, data_inputs, expected_outputs=None):
        return self.fit(data_inputs, expected_outputs).transform(data_inputs)

    def fit_transform_one(self, data_input, expected_output=None):
        return self.fit_one(data_input, expected_output).transform_one(data_input)

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        if expected_outputs is None:
            expected_outputs = [None] * len(data_inputs)
        for data_input, expected_output in zip(data_inputs, expected_outputs):
            self.fit_one(data_input, expected_output)
        return self

    def fit_one(self, data_input, expected_output=None):
        # return self
        raise NotImplementedError("TODO")

    def transform(self, data_inputs):
        processed_outputs = [self.transform_one(data_input) for data_input in data_inputs]
        return processed_outputs

    def transform_one(self, data_input):
        # return processed_output
        raise NotImplementedError("TODO")


NamedTupleList = List[Tuple[str, BaseStep]]


class BaseBarrier(ABC):
    # TODO: a barrier is between steps and manages how they interact (e.g.: a checkpoint).
    pass


class BaseBlockBarrier(BaseBarrier, ABC):
    # TODO: a block barrier forces not using any "_one" functions past that barrier.
    pass


class BaseStreamingBarrier(BaseBarrier, ABC):
    # TODO: a block barrier forces using the "_one" functions past that barrier.
    pass


class PipelineRunner(BaseStep, ABC):

    def __init__(self, **pipeline_hyperparams):
        BaseStep.__init__(self, **pipeline_hyperparams)
        self.named_pipeline_steps: NamedTupleList = None

    def set_steps(self, named_pipeline_steps: NamedTupleList) -> 'PipelineRunner':
        self.named_pipeline_steps: NamedTupleList = named_pipeline_steps
        return self

    @abstractmethod
    def fit_transform(self, data_inputs, expected_outputs=None):
        pass

    @abstractmethod
    def fit(self, data_inputs, expected_outputs=None):
        pass

    @abstractmethod
    def transform(self, data_inputs):
        pass
