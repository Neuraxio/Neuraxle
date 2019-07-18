"""
Union of Features
==========================
This module contains steps to perform various feature unions and model stacking, using parallelism is possible.

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

from joblib import Parallel, delayed

from neuraxle.base import BaseStep, TruncableSteps, NonFittableMixin, NamedTupleList, NonTransformableMixin
from neuraxle.steps.numpy import NumpyConcatenateInnerFeatures


class FeatureUnion(TruncableSteps):
    """Parallelize the union of many pipeline steps."""

    def __init__(
            self,
            steps_as_tuple: NamedTupleList,
            joiner: NonFittableMixin = NumpyConcatenateInnerFeatures(),
            n_jobs: int = None,
            backend: str = "threading"
    ):
        """
        Create a feature union.

        :param steps_as_tuple: the NamedTupleList of steps to process in parallel and to join.
        :param joiner: What will be used to join the features. For example, ``NumpyConcatenateInnerFeatures()``.
        :param n_jobs: The number of jobs for the parallelized ``joblib.Parallel`` loop in fit and in transform.
        :param backend: The type of parallelization to do with ``joblib.Parallel``. Possible values: "loky", "multiprocessing", "threading", "dask" if you use dask, and more.
        """
        super().__init__(steps_as_tuple)
        self.joiner = joiner  # TODO: add "other" types of step(s) to TuncableSteps or to another intermediate class. For example, to get their hyperparameters.
        self.n_jobs = n_jobs
        self.backend = backend

    def fit(self, data_inputs, expected_outputs=None) -> 'FeatureUnion':
        """
        Fit the parallel steps on the data. It will make use of some parallel processing.

        :param data_inputs: The input data to fit onto
        :param expected_outputs: The output that should be obtained when fitting.
        :return: self
        """
        # Actually fit:
        if self.n_jobs != 1:
            fitted = Parallel(backend=self.backend, n_jobs=self.n_jobs)(
                delayed(bro.fit)(data_inputs, expected_outputs)
                for _, bro in self.steps_as_tuple
            )
        else:
            fitted = [
                bro.fit(data_inputs, expected_outputs)
                for _, bro in self.steps_as_tuple
            ]

        # Save fitted steps
        for i, f in enumerate(fitted):
            self.steps_as_tuple[i] = (self.steps_as_tuple[i][0], f)
        self._refresh_steps()

        return self

    def transform(self, data_inputs):
        """
        Transform the data with the unions. It will make use of some parallel processing.

        :param data_inputs: The input data to fit onto
        :return: the transformed data_inputs.
        """
        if self.n_jobs != 1:
            results = Parallel(backend=self.backend, n_jobs=self.n_jobs)(
                delayed(bro.transform)(data_inputs)
                for _, bro in self.steps_as_tuple
            )
        else:
            results = [
                bro.transform(data_inputs)
                for _, bro in self.steps_as_tuple
            ]

        results = self.joiner.transform(results)
        return results


class Identity(NonTransformableMixin, NonFittableMixin, BaseStep):
    """A pipeline step that has no effect at all but to return the same data without changes.

    This can be useful to concatenate new features to existing features, such as what AddFeatures do.

    Identity inherits from ``NonTransformableMixin`` and from ``NonFittableMixin`` which makes it a class that has no
    effect in the pipeline: it doesn't require fitting, and at transform-time, it returns the same data it received.
    """
    pass  # Multi-class inheritance does the job here! See inside those other classes for more info.


class AddFeatures(FeatureUnion):
    """Parallelize the union of many pipeline steps AND concatenate the new features to the received inputs using Identity."""

    def __init__(self, steps_as_tuple: NamedTupleList, **kwargs):
        """
        Create a ``FeatureUnion`` where ``Identity`` is the first step so as to also keep
        the inputs to concatenate them to the outputs.

        :param steps_as_tuple: The steps to be sent to the ``FeatureUnion``. ``Identity()`` is prepended.
        :param kwargs: Other arguments to send to ``FeatureUnion``.
        """
        super().__init__([Identity()] + steps_as_tuple, **kwargs)


class ModelStacking(FeatureUnion):
    """Performs a ``FeatureUnion`` of steps, and then send the joined result to the above judge step."""

    def __init__(
            self,
            steps_as_tuple: NamedTupleList,
            judge: BaseStep,
            **kwargs
    ):
        """
        Perform model stacking. The steps will be merged with a FeatureUnion,
        and the judge will recombine the predictions.

        :param steps_as_tuple: the NamedTupleList of steps to process in parallel and to join.
        :param judge: a BaseStep that will learn to judge the best answer and who to trust out of every parallel steps.
        :param kwargs: Other arguments to send to ``FeatureUnion``.
        """
        super().__init__(steps_as_tuple, **kwargs)
        self.judge: BaseStep = judge  # TODO: add "other" types of step(s) to TuncableSteps or to another intermediate class. For example, to get their hyperparameters.

    def fit(self, data_inputs, expected_outputs=None) -> 'ModelStacking':
        """
        Fit the parallel steps on the data. It will make use of some parallel processing.
        Also, fit the judge on the result of the parallel steps.

        :param data_inputs: The input data to fit onto
        :param expected_outputs: The output that should be obtained when fitting.
        :return: self
        """
        super().fit(data_inputs, expected_outputs)
        results = super().transform(data_inputs)

        self.judge = self.judge.fit(results, expected_outputs)
        return self

    def transform(self, data_inputs):
        """
        Transform the data with the unions. It will make use of some parallel processing.
        Then, use the judge to refine the transformations.

        :param data_inputs: The input data to fit onto
        :return: the transformed data_inputs.
        """
        results = super().transform(data_inputs)
        return self.judge.transform(results)
