"""
Random
====================================
Meta steps for hyperparameter tuning, such as random search.
"""
import copy
from abc import ABC, abstractmethod
from typing import List

from neuraxle.base import MetaStepMixin, BaseStep
from neuraxle.steps.util import StepClonerForEachDataInput


class BaseCrossValidation(MetaStepMixin, MetaStepMixin, BaseStep, ABC):
    # TODO: assert that set_step was called.

    def __init__(self, k_fold=3):
        super().__init__()
        self.k_fold = k_fold

    def fit(self, data_inputs, expected_outputs=None) -> BaseStep:
        # TODO: assert that set_step was called.
        data_inputs = self.split(data_inputs)
        self.step = StepClonerForEachDataInput(self.step)
        self.step.fit(data_inputs, expected_outputs)
        return self

    def transform(self, data_inputs):
        data_inputs = self.split(data_inputs)
        return self.step.transform(data_inputs)

    @abstractmethod
    def split(self, data_inputs) -> List:
        # return a list if size k
        raise NotImplementedError("TODO")  # TODO.


class KFoldCrossValidation(BaseCrossValidation):

    def split(self, data_inputs) -> List:
        raise NotImplementedError("TODO")  # TODO.


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

            generated_outputs = step.fit_transform(data_inputs, expected_outputs)
            score = self.scoring_function(generated_outputs, expected_outputs)

            if not started or self.higher_score_is_better == (score > self.score):
                started = True
                self.score = score
                self.best_model = step

        return self

    def transform(self, data_inputs):
        # TODO: check this again to be sure.
        return self.best_model.transform(data_inputs)
