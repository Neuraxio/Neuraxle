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

import numpy as np
from sklearn.metrics import r2_score

from neuraxle.base import MetaStepMixin, BaseStep
from neuraxle.steps.numpy import NumpyConcatenateOuterBatch
from neuraxle.steps.util import StepClonerForEachDataInput


class BaseCrossValidation(MetaStepMixin, BaseStep, ABC):
    # TODO: assert that set_step was called.
    # TODO: change default argument of scoring_function...
    def __init__(self, scoring_function=r2_score, k_fold=3, joiner=NumpyConcatenateOuterBatch()):
        super().__init__()
        self.scoring_function = scoring_function
        self.k_fold = k_fold
        self.joiner = joiner

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseCrossValidation':
        train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs = self.split(
            data_inputs, expected_outputs)

        step = StepClonerForEachDataInput(self.step)
        step = step.fit(train_data_inputs, train_expected_outputs)

        results = step.transform(validation_data_inputs)
        self.scores = [self.scoring_function(a, b) for a, b in zip(results, validation_expected_outputs)]
        self.scores_mean = np.mean(self.scores)
        self.scores_std = np.std(self.scores)

        return self

    @abstractmethod
    def split(self, data_inputs, expected_outputs):
        raise NotImplementedError("TODO")

    def transform(self, data_inputs):
        # TODO: use the splits and average the results?? instead of picking best model...
        raise NotImplementedError("TODO: code this method in Neuraxle.")
        data_inputs = self.split(data_inputs)
        predicted_outputs_splitted = self.step.transform(data_inputs)
        return self.joiner.transform(predicted_outputs_splitted)


class KFoldCrossValidation(BaseCrossValidation):

    def split(self, data_inputs, expected_outputs):
        validation_data_inputs, validation_expected_outputs = self.validation_split(
            data_inputs, expected_outputs)

        train_data_inputs, train_expected_outputs = self.train_split(
            validation_data_inputs, validation_expected_outputs)

        return train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs

    def train_split(self, validation_data_inputs, validation_expected_outputs) -> (List, List):
        train_data_inputs = []
        train_expected_outputs = []
        for i in range(len(validation_data_inputs)):
            inputs = validation_data_inputs[:i] + validation_data_inputs[i + 1:]
            outputs = validation_expected_outputs[:i] + validation_expected_outputs[i + 1:]

            inputs = self.joiner.transform(inputs)
            outputs = self.joiner.transform(outputs)

            train_data_inputs.append(inputs)
            train_expected_outputs.append(outputs)

        return train_data_inputs, train_expected_outputs

    def validation_split(self, data_inputs, expected_outputs=None) -> List:
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
            if b > len(data_inputs):
                b = len(data_inputs)

            slice = data_inputs[a:b]
            splitted_data_inputs.append(slice)
        return splitted_data_inputs


class RandomSearch(MetaStepMixin, BaseStep):
    """Perform a random hyperparameter search."""

    # TODO: CV and rename to RandomSearchCV.

    def __init__(
            self,
            n_iter: int,
            higher_score_is_better: bool,
            validation_technique: BaseCrossValidation = KFoldCrossValidation(),
            refit=True
    ):
        super().__init__()
        self.n_iter = n_iter
        self.higher_score_is_better = higher_score_is_better
        self.validation_technique: BaseCrossValidation = validation_technique
        self.refit = refit

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        # TODO: assert that set_step was called.
        started = False
        for _ in range(self.n_iter):

            step = copy.copy(self.step)

            new_hyperparams = step.get_hyperparams_space().rvs()
            step.set_hyperparams(new_hyperparams)

            step: BaseCrossValidation = copy.copy(self.validation_technique).set_step(step)

            # TODO: skip on error???
            step = step.fit(data_inputs, expected_outputs)
            score = step.scores_mean

            if not started or self.higher_score_is_better == (score > self.score):
                started = True
                self.score = score
                self.best_validation_wrapper_of_model = step

        if self.refit:
            self.best_model = self.best_validation_wrapper_of_model.step.fit(
                data_inputs, expected_outputs)

        return self

    def transform(self, data_inputs):
        # TODO: check this again to be sure.
        return self.best_validation_wrapper_of_model.transform(data_inputs)
