from typing import Any

import numpy as np

from neuraxle.base import BaseStep, NonTransformableMixin, NonFittableMixin
from neuraxle.pipeline import Pipeline
from neuraxle.steps.misc import BaseCallbackStep, TapeCallbackFunction


def main():
    # Fit method is automatically implemented as changing nothing.

    class FitCallbackStep(NonTransformableMixin, BaseCallbackStep):
        """Call a callback method on fit."""

        def fit(self, data_inputs, expected_outputs=None) -> 'FitCallbackStep':
            self._callback((data_inputs, expected_outputs))
            return self

    # Transform method is automatically implemented as changing nothing.

    class TransformCallbackStep(NonFittableMixin, BaseCallbackStep):
        def fit_transform(self, data_inputs, expected_outputs=None) -> ('BaseStep', Any):
            self._callback(data_inputs)

            return self, data_inputs

        def transform(self, data_inputs):
            self._callback(data_inputs)
            if self.transform_function is not None:
                return self.transform_function(data_inputs)

            return data_inputs

        def inverse_transform(self, processed_outputs):
            self._callback(processed_outputs)
            return processed_outputs

    tape_fit = TapeCallbackFunction()
    tape_transform = TapeCallbackFunction()

    p = Pipeline([
        FitCallbackStep(tape_fit),
        TransformCallbackStep(tape_transform)
    ])

    p = p.fit(np.array([0, 1]), np.array([0, 1]))

    assert np.array_equal(tape_fit.data[0][0], np.array([0, 1]))
    assert np.array_equal(tape_fit.data[0][1], np.array([0, 1]))
    assert tape_transform.data == []

    tape_fit.data = []

    p = p.transform(np.array([0, 1]))

    assert tape_fit.data == []
    assert np.array_equal(tape_transform.data[0], np.array([0, 1]))

if __name__ == "__main__":
    main()
