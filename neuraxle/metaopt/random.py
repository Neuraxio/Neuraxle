"""
Random
====================================
Meta steps for hyperparameter tuning, such as random search.

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

import copy
from abc import ABC, abstractmethod
from typing import List

from neuraxle.base import MetaStepMixin, BaseStep
from neuraxle.steps.util import StepClonerForEachDataInput


class BaseCrossValidation(MetaStepMixin, BaseStep, ABC):
    # TODO: assert that set_step was called.

    def __init__(self, k_fold=3):
        super().__init__()
        self.k_fold = k_fold

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseCrossValidation':
        # TODO: assert that set_step was called.
        data_inputs, expected_outputs = self.split(data_inputs, expected_outputs)
        self.step = StepClonerForEachDataInput(self.step)
        self.step = self.step.fit(data_inputs, expected_outputs)
        return self

    def transform(self, data_inputs):
        data_inputs = self.split(data_inputs)
        return self.step.transform(data_inputs)

    @abstractmethod
    def split(self, data_inputs, expected_outputs=None) -> List:
        # return a list if size k
        raise NotImplementedError("TODO")  # TODO.


class KFoldCrossValidation(BaseCrossValidation):

    def split(self, data_inputs, expected_outputs=None) -> List:
        splitted_data_inputs = self._split(data_inputs)
        if expected_outputs is not None:
            splitted_expected_outputs = self._split(expected_outputs)
            return splitted_data_inputs, splitted_expected_outputs
        return splitted_data_inputs

    def _split(self, data_inputs):
        splitted_data_inputs = []
        step = len(data_inputs) / float(self.k_fold)
        for i in range(self.k_fold):
            a = int(step * i)
            b = int(step * (i + 1))
            if i >= self.k_fold - 1:
                b = len(data_inputs)

            slice = data_inputs[a:b]
            splitted_data_inputs.append(slice)
        print(len(data_inputs), [len(s) for s in splitted_data_inputs])
        return splitted_data_inputs

    def merge(self, data_inputs, expected_outputs=None):
        if expected_outputs is None:
            return sum(data_inputs, [])
        else:
            return sum(data_inputs, []), sum(expected_outputs, [])


class RandomSearch(MetaStepMixin, BaseStep):
    """Perform a random hyperparameter search."""

    # TODO: CV and rename to RandomSearchCV.

    def __init__(
            self,
            n_iter: int,
            scoring_function,
            higher_score_is_better: bool,
            cross_validation_technique: BaseCrossValidation = KFoldCrossValidation()
    ):
        super().__init__()
        self.n_iter = n_iter
        self.scoring_function = scoring_function
        self.higher_score_is_better = higher_score_is_better
        self.cross_validation_technique: BaseCrossValidation = cross_validation_technique

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        # TODO: assert that set_step was called.
        started = False
        for _ in range(self.n_iter):

            step = copy.copy(self.step)

            new_hyperparams = step.get_hyperparams_space().rvs()
            step.set_hyperparams(new_hyperparams)

            step = copy.copy(self.cross_validation_technique).set_step(step)

            step, generated_outputs = step.fit_transform(data_inputs, expected_outputs)
            score = self.scoring_function(generated_outputs, expected_outputs)

            if not started or self.higher_score_is_better == (score > self.score):
                started = True
                self.score = score
                self.best_model = step

        return self

    def transform(self, data_inputs):
        # TODO: check this again to be sure.
        return self.best_model.transform(data_inputs)
