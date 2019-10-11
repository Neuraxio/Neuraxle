"""
Pipeline Steps For Looping
=====================================

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
from typing import List, Any

from neuraxle.base import MetaStepMixin, BaseStep
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace


class ForEachDataInputs(MetaStepMixin, BaseStep):
    """
    Truncable step that fits/transforms each step for each of the data inputs, and expected outputs.
    """

    def __init__(
            self,
            wrapped: BaseStep
    ):
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)

    def fit(self, data_inputs, expected_outputs=None):
        """
        Fit each step for each data inputs, and expected outputs

        :param data_inputs:
        :param expected_outputs:
        :return: self
        """
        if expected_outputs is None:
            expected_outputs = [None] * len(data_inputs)

        for di, eo in zip(data_inputs, expected_outputs):
            self.wrapped = self.wrapped.fit(di, eo)

        return self

    def transform(self, data_inputs):
        """
        Transform each step for each data inputs, and expected outputs

        :param data_inputs:
        :return: self
        """
        outputs = []
        for di in data_inputs:
            output = self.wrapped.transform(di)
            outputs.append(output)

        return outputs

    def fit_transform(self, data_inputs, expected_outputs=None):
        """
        Fit transform each step for each data inputs, and expected outputs

        :param data_inputs:
        :param expected_outputs:
        :return: self
        """
        if expected_outputs is None:
            expected_outputs = [None] * len(data_inputs)

        outputs = []
        for di, eo in zip(data_inputs, expected_outputs):
            self.wrapped, output = self.wrapped.fit_transform(di, eo)
            outputs.append(output)

        return self, outputs


class StepClonerForEachDataInput(MetaStepMixin, BaseStep):
    def __init__(self, wrapped: BaseStep, copy_op=copy.deepcopy):
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        self.set_step(wrapped)
        self.steps: List[BaseStep] = []
        self.copy_op = copy_op

    def set_hyperparams(self, hyperparams: HyperparameterSamples) -> BaseStep:
        super().set_hyperparams(hyperparams)
        self.steps = [s.set_hyperparams(self.wrapped.hyperparams) for s in self.steps]
        return self

    def set_hyperparams_space(self, hyperparams_space: HyperparameterSpace) -> 'BaseStep':
        super().set_hyperparams_space(hyperparams_space)
        self.steps = [s.set_hyperparams_space(self.wrapped.hyperparams_space) for s in self.steps]
        return self

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BaseStep', Any):
        # One copy of step per data input:
        self.steps = [self.copy_op(self.wrapped) for _ in range(len(data_inputs))]

        if expected_outputs is None:
            expected_outputs = [None] * len(data_inputs)

        fit_transform_result = [self.steps[i].fit_transform(di, eo) for i, (di, eo) in
                                enumerate(zip(data_inputs, expected_outputs))]
        self.steps = [step for step, di in fit_transform_result]
        data_inputs = [di for step, di in fit_transform_result]

        return self, data_inputs

    def fit(self, data_inputs: List, expected_outputs: List = None) -> 'StepClonerForEachDataInput':
        # One copy of step per data input:
        self.steps = [self.copy_op(self.wrapped) for _ in range(len(data_inputs))]

        # Fit them all.
        if expected_outputs is None:
            expected_outputs = [None] * len(data_inputs)
        self.steps = [self.steps[i].fit(di, eo) for i, (di, eo) in enumerate(zip(data_inputs, expected_outputs))]

        return self

    def transform(self, data_inputs: List) -> List:
        return [self.steps[i].transform(di) for i, di in enumerate(data_inputs)]

    def inverse_transform(self, data_output):
        return [self.steps[i].inverse_transform(di) for i, di in enumerate(data_output)]