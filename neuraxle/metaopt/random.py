"""
Random
====================================
Meta steps for hyperparameter tuning, such as random search.
"""
import copy

from neuraxle.base import MetaStepMixin, BaseStep


class RandomSearch(MetaStepMixin, BaseStep):
    """Perform a random hyperparameter search."""

    # TODO: CV and rename to RandomSearchCV.

    def __init__(self, n_iter: int, scoring_function, higher_score_is_better: bool):
        super().__init__()
        self.n_iter = n_iter
        self.scoring_function = scoring_function
        self.higher_score_is_better = higher_score_is_better

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        started = False
        for _ in range(self.n_iter):

            step = copy.copy(self.step)

            new_hyperparams = step.get_hyperparams_space().rvs()
            step.set_hyperparams(new_hyperparams)

            generated_outputs = step.fit_transform(data_inputs, expected_outputs)
            score = self.scoring_function(generated_outputs, expected_outputs)

            if not started or score > self.score:  # TODO: use `self.higher_score_is_better`.
                started = True
                self.score = score
                self.best_model = step

        return self

    def transform(self, data_inputs):
        # TODO: check this again to be sure.
        return self.best_model.transform(data_inputs)
