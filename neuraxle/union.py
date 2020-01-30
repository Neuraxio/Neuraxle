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

..
    Thanks to Umaneo Technologies Inc. for their contributions to this Machine Learning
    project, visit https://www.umaneo.com/ for more information on Umaneo Technologies Inc.

"""

from joblib import Parallel, delayed

from neuraxle.base import BaseStep, TruncableSteps, NonFittableMixin, NamedTupleList, Identity, ExecutionContext, \
    DataContainer, HandlerMixin
from neuraxle.steps.numpy import NumpyConcatenateInnerFeatures


class FeatureUnion(HandlerMixin, TruncableSteps):
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
        TruncableSteps.__init__(self, steps_as_tuple)
        self.joiner = joiner
        self.n_jobs = n_jobs
        self.backend = backend

    def apply(self, method_name: str, *kargs, **kwargs) -> 'BaseStep':
        BaseStep.apply(self, method_name, *kargs, **kwargs)
        self.joiner.apply(method_name, *kargs, **kwargs)

    def _fit_data_container(self, data_container, context):
        """
        Fit the parallel steps on the data. It will make use of some parallel processing.

        :param data_container: The input data to fit onto
        :param context: execution context
        :return: self
        """
        # Actually fit:
        if self.n_jobs != 1:
            fitted_steps = Parallel(backend=self.backend, n_jobs=self.n_jobs)(
                delayed(step.handle_fit)(data_container.copy(), context)
                for _, step in self.steps_as_tuple
            )
        else:
            fitted_steps = [
                step.handle_fit(data_container.copy(), context)
                for _, step in self.steps_as_tuple
            ]

        # Save fitted steps
        for i, fitted_step in enumerate(fitted_steps):
            self.steps_as_tuple[i] = (self.steps_as_tuple[i][0], fitted_step)
        self._refresh_steps()

        return self

    def _transform_data_container(self, data_container, context):
        """
        Transform the data with the unions. It will make use of some parallel processing.

        :param data_container: data container
        :param context: execution context
        :return: the transformed data_inputs.
        """
        if self.n_jobs != 1:
            data_containers = Parallel(backend=self.backend, n_jobs=self.n_jobs)(
                delayed(step.handle_transform)(data_container.copy(), context)
                for _, step in self.steps_as_tuple
            )
        else:
            data_containers = [
                step.handle_transform(data_container.copy(), context)
                for _, step in self.steps_as_tuple
            ]

        return DataContainer(data_inputs=data_containers, current_ids=data_container.current_ids,
                             summary_id=data_container.summary_id, expected_outputs=data_container.expected_outputs)

    def _did_transform(self, data_container, context):
        data_container = self.joiner.handle_transform(data_container, context)
        return data_container

    def _fit_transform_data_container(self, data_container, context):
        """
        Transform the data with the unions. It will make use of some parallel processing.

        :param data_container: data container
        :param context: execution context
        :return: the transformed data_inputs.
        """
        new_self = self._fit_data_container(data_container, context)
        data_container = self._transform_data_container(data_container, context)
        return new_self, data_container

    def _did_fit_transform(self, data_container, context):
        self.joiner, data_container = self.joiner.handle_fit_transform(data_container, context)
        return data_container


class AddFeatures(FeatureUnion):
    """Parallelize the union of many pipeline steps AND concatenate the new features to the received inputs using Identity."""

    def __init__(self, steps_as_tuple: NamedTupleList, **kwargs):
        """
        Create a ``FeatureUnion`` where ``Identity`` is the first step so as to also keep
        the inputs to concatenate them to the outputs.

        :param steps_as_tuple: The steps to be sent to the ``FeatureUnion``. ``Identity()`` is prepended.
        :param kwargs: Other arguments to send to ``FeatureUnion``.
        """
        FeatureUnion.__init__(self, [Identity()] + steps_as_tuple, **kwargs)


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
        FeatureUnion.__init__(self, steps_as_tuple, **kwargs)
        self.judge: BaseStep = judge  # TODO: add "other" types of step(s) to TuncableSteps or to another intermediate class. For example, to get their hyperparameters.

    def _fit_transform_data_container(self, data_container, context) -> (BaseStep, DataContainer):
        new_self, _ = super()._fit_transform_data_container(data_container, context)

        new_self = new_self._fit_data_container(data_container, context)
        data_container = new_self._transform_data_container(data_container, context)

        return new_self, data_container

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> BaseStep:
        """
        Fit the parallel steps on the data. It will make use of some parallel processing.
        Also, fit the judge on the result of the parallel steps.

        :param data_container: data container to fit on
        :param context: execution context
        :return: self
        """
        new_self = super()._fit_data_container(data_container, context)
        new_self = new_self.fit(data_container.data_inputs, data_container.expected_outputs)

        return new_self

    def _transform_data_container(self, data_container, context) -> DataContainer:
        """
        Transform the data with the unions. It will make use of some parallel processing.
        Then, use the judge to refine the transformations.

        :param data_container: data container to transform
        :param context: execution context
        """
        data_container = super()._transform_data_container(data_container, context)
        results = self.judge.transform(data_container.data_inputs)
        data_container.set_data_inputs(results)

        return data_container

