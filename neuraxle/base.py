"""
Neuraxle's Base Classes
====================================
This is the core of Neuraxle. Most pipeline steps derive (inherit) from those classes. They are worth noticing.

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

import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import copy
from typing import Tuple, List, Union, Any

from neuraxle.hyperparams.space import HyperparameterSpace, HyperparameterSamples


class BaseStep(ABC):

    def __init__(
            self,
            hyperparams: HyperparameterSamples = None,
            hyperparams_space: HyperparameterSpace = None
    ):
        if hyperparams is None:
            hyperparams = dict()
        if hyperparams_space is None:
            hyperparams_space = dict()

        self.hyperparams: HyperparameterSamples = hyperparams
        self.hyperparams_space: HyperparameterSpace = hyperparams_space

        self.pending_mutate: ('BaseStep', str, str) = (None, None, None)

    def set_hyperparams(self, hyperparams: HyperparameterSamples) -> 'BaseStep':
        self.hyperparams = HyperparameterSamples(hyperparams)
        return self

    def get_hyperparams(self) -> HyperparameterSamples:
        return self.hyperparams

    def set_hyperparams_space(self, hyperparams_space: HyperparameterSpace) -> 'BaseStep':
        self.hyperparams_space = HyperparameterSpace(hyperparams_space)
        return self

    def get_hyperparams_space(self, flat=False) -> HyperparameterSpace:
        return self.hyperparams_space

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BaseStep', Any):
        new_self = self.fit(data_inputs, expected_outputs)
        out = new_self.transform(data_inputs)
        return new_self, out

    def fit_transform_one(self, data_input, expected_output=None) -> ('BaseStep', Any):
        new_self = self.fit_one(data_input, expected_output)
        out = new_self.transform_one(data_input)
        return new_self, out

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        if expected_outputs is None:
            expected_outputs = [None] * len(data_inputs)
        for data_input, expected_output in zip(data_inputs, expected_outputs):
            self.fit_one(data_input, expected_output)
        return self

    def transform(self, data_inputs):
        processed_outputs = [self.transform_one(data_input) for data_input in data_inputs]
        return processed_outputs

    def inverse_transform(self, processed_outputs):
        data_inputs = [self.inverse_transform_one(data_output) for data_output in processed_outputs]
        return data_inputs

    def predict(self, data_input):
        return self.transform(data_input)

    def meta_fit(self, X_train, y_train, metastep: 'MetaStepMixin'):
        """
        Uses a meta optimization technique (AutoML) to find the best hyperparameters in the given
        hyperparameter space.

        Usage: ``p = p.meta_fit(X_train, y_train, metastep=RandomSearch(n_iter=10, scoring_function=r2_score, higher_score_is_better=True))``

        Call ``.mutate(new_method="inverse_transform", method_to_assign_to="transform")``, and the
        current estimator will become

        :param X_train: data_inputs.
        :param y_train: expected_outputs.
        :param metastep: a metastep, that is, a step that can sift through the hyperparameter space of another estimator.
        :return: your best self.
        """
        metastep.set_step(self)
        metastep = metastep.fit(X_train, y_train)
        best_step = metastep.get_best_model()
        return best_step

    def mutate(self, new_method="inverse_transform", method_to_assign_to="transform", warn=True) -> 'BaseStep':
        """
        Replace the "method_to_assign_to" method by the "new_method" method, IF the present object has no pending calls to
        ``.will_mutate_to()`` waiting to be applied. If there is a pending call, the pending call will override the
        methods specified in the present call. If the change fails (such as if the new_method doesn't exist), then
        a warning is printed (optional). By default, there is no pending ``will_mutate_to`` call.

        This could for example be useful within a pipeline to apply ``inverse_transform`` to every pipeline steps, or
        to assign ``predict_probas`` to ``predict``, or to assign "inverse_transform" to "transform" to a reversed pipeline.

        :param new_method: the method to replace transform with, if there is no pending ``will_mutate_to`` call.
        :param method_to_assign_to: the method to which the new method will be assigned to, if there is no pending ``will_mutate_to`` call.
        :param warn: (verbose) wheter or not to warn about the inexistence of the method.
        :return: self, a copy of self, or even perhaps a new or different BaseStep object.
        """
        pending_new_base_step, pending_new_method, pending_method_to_assign_to = self.pending_mutate

        # Use everything that is pending if they are not none (ternaries).
        new_base_step = pending_new_base_step if pending_new_base_step is not None else copy(self)
        new_method = pending_new_method if pending_new_method is not None else new_method
        method_to_assign_to = pending_method_to_assign_to if pending_method_to_assign_to is not None else method_to_assign_to

        # We set "new_method" in place of "method_to_affect" to a copy of self:
        try:
            # 1. get new method's reference
            new_method = getattr(new_base_step, new_method)

            # 2. delete old method
            try:
                delattr(new_base_step, method_to_assign_to)
            except AttributeError as e:
                pass

            # 3. assign new method to old method
            setattr(new_base_step, method_to_assign_to, new_method)

        except AttributeError as e:
            if warn:
                import warnings
                warnings.warn(e)

        return new_base_step

    def will_mutate_to(
            self, new_base_step: 'BaseStep' = None, new_method: str = None, method_to_assign_to: str = None
    ) -> 'BaseStep':
        """
        This will change the behavior of ``self.mutate(<...>)`` such that when mutating, it will return the
        presently provided new_base_step BaseStep (can be left to None for self), and the ``.mutate`` method
        will also apply the ``new_method`` and the  ``method_to_affect``, if they are not None, and after changing
        the object to new_base_step.

        This can be useful if your pipeline requires unsupervised pretraining. For example:

        .. code-block:: python

            X_pretrain = ...
            X_train = ...

            p = Pipeline(
                SomePreprocessing(),
                SomePretrainingStep().will_mutate_to(new_base_step=SomeStepThatWillUseThePretrainingStep),
                Identity().will_mutate_to(new_base_step=ClassifierThatWillBeUsedOnlyAfterThePretraining)
            )
            # Pre-train the pipeline
            p = p.fit(X_pretrain, y=None)

            # This will leave `SomePreprocessing()` untouched and will affect the two other steps.
            p = p.mutate(new_method="transform", method_to_affect="transform")

            # Pre-train the pipeline
            p = p.fit(X_train, y_train)  # Then fit the classifier and other new things

        :param new_base_step: if it is not None, upon calling ``mutate``, the object it will mutate to will be this provided new_base_step.
        :param method_to_assign_to: if it is not None, upon calling ``mutate``, the method_to_affect will be the one that is used on the provided new_base_step.
        :param new_method: if it is not None, upon calling ``mutate``, the new_method will be the one that is used on the provided new_base_step.
        :return: self
        """
        if new_method is None or method_to_assign_to is None:
            new_method = method_to_assign_to = "transform"  # No changes will be applied (transform will stay transform).

        self.pending_mutate = (new_base_step, new_method, method_to_assign_to)

        return self

    def fit_one(self, data_input, expected_output=None) -> 'BaseStep':
        # return self
        raise NotImplementedError("TODO: Implement this method in {}.".format(self.__class__.__name__))

    def transform_one(self, data_input):
        # return processed_output
        raise NotImplementedError("TODO: Implement this method in {}.".format(self.__class__.__name__))

    def inverse_transform_one(self, data_output):
        # return data_input
        raise NotImplementedError("TODO: Implement this method in {}.".format(self.__class__.__name__))

    def tosklearn(self) -> 'NeuraxleToSKLearnPipelineWrapper':
        from sklearn.base import BaseEstimator

        class NeuraxleToSKLearnPipelineWrapper(BaseEstimator):
            def __init__(self, neuraxle_step):
                self.p: Union[BaseStep, TruncableSteps] = neuraxle_step

            def set_params(self, **params) -> BaseEstimator:
                self.p.set_hyperparams(HyperparameterSpace(params))
                return self

            def get_params(self, deep=True):
                neuraxle_params = HyperparameterSamples(self.p.get_hyperparams()).to_flat_as_dict_primitive()
                return neuraxle_params

            def get_params_space(self, deep=True):
                neuraxle_params = HyperparameterSpace(self.p.get_hyperparams_space()).to_flat_as_dict_primitive()
                return neuraxle_params

            def fit(self, **args) -> BaseEstimator:
                self.p = self.p.fit(**args)
                return self

            def transform(self, **args):
                return self.p.transform(**args)

            def fit_transform(self, **args) -> Any:
                self.p, out = self.p.fit_transform(**args)
                # Careful: 1 return value.
                return out

            def inverse_transform(self, **args):
                return self.p.reverse().transform(**args)

            def predict(self, **args):
                return self.p.transform(**args)

        return NeuraxleToSKLearnPipelineWrapper(self)

    def reverse(self) -> 'BaseStep':
        """
        The object will mutate itself such that the ``.transform`` method (and of all its underlying objects
        if applicable) be replaced by the ``.inverse_transform`` method.

        Note: the reverse may fail if there is a pending mutate that was set earlier with ``.will_mutate_to``.

        :return: a copy of self, reversed. Each contained object will also have been reversed if self is a pipeline.
        """
        return self.mutate(new_method="inverse_transform", method_to_assign_to="transform")

    def __reversed__(self) -> 'BaseStep':
        """
        The object will mutate itself such that the ``.transform`` method (and of all its underlying objects
        if applicable) be replaced by the ``.inverse_transform`` method.

        Note: the reverse may fail if there is a pending mutate that was set earlier with ``.will_mutate_to``.

        :return: a copy of self, reversed. Each contained object will also have been reversed if self is a pipeline.
        """
        return self.reverse()


class MetaStepMixin:
    """A class to represent a meta step which is used to optimize another step."""

    # TODO: how to set_params on contained step?

    def set_step(self, step: BaseStep) -> BaseStep:
        self.step: BaseStep = step
        return self

    def get_best_model(self) -> BaseStep:
        return self.best_model


NamedTupleList = List[Union[Tuple[str, 'BaseStep'], 'BaseStep']]


class MetaStepsMixin:
    def __init__(self, **pipeline_hyperparams):
        BaseStep.__init__(self, **pipeline_hyperparams)
        self.steps_as_tuple: NamedTupleList = None

    def set_steps(self, steps_as_tuple: NamedTupleList) -> 'BasePipelineRunner':
        self.steps_as_tuple: NamedTupleList = steps_as_tuple
        return self


class NonFittableMixin:
    """A pipeline step that requires no fitting: fitting just returns self when called to do no action.

    Note: fit methods are not implemented"""

    def fit(self, data_inputs, expected_outputs=None) -> 'NonFittableMixin':
        """
        Don't fit.

        :param data_inputs: the data that would normally be fitted on.
        :param expected_outputs: the data that would normally be fitted on.
        :return: self
        """
        return self

    def fit_one(self, data_input, expected_output=None) -> 'NonFittableMixin':
        """
        Don't fit.

        :param data_input: the data that would normally be fitted on.
        :param expected_output: the data that would normally be fitted on.
        :return: self
        """
        return self


class NonTransformableMixin:
    """A pipeline step that has no effect at all but to return the same data without changes.

    Note: fit methods are not implemented"""

    def transform(self, data_inputs):
        """
        Do nothing - return the same data.

        :param data_inputs: the data to process
        :return: the ``data_inputs``, unchanged.
        """
        return data_inputs

    def transform_one(self, data_input):
        """
        Do nothing - return the same data.

        :param data_input: the data to process
        :return: the ``data_input``, unchanged.
        """
        return data_input

    def inverse_transform(self, processed_outputs):
        """
        Do nothing - return the same data.

        :param processed_outputs: the data to process
        :return: the ``processed_outputs``, unchanged.
        """
        return processed_outputs

    def inverse_transform_one(self, processed_output):
        """
        Do nothing - return the same data.

        :param processed_output: the data to process
        :return: the ``data_output``, unchanged.
        """
        return processed_output


class TruncableSteps(BaseStep, ABC):

    def __init__(
            self,
            steps_as_tuple: NamedTupleList,
            hyperparams: HyperparameterSamples = dict(),
            hyperparams_space: HyperparameterSpace = dict()
    ):
        super().__init__(hyperparams, hyperparams_space)
        self.steps_as_tuple: NamedTupleList = self.patch_missing_names(steps_as_tuple)
        self._refresh_steps()
        assert isinstance(self, BaseStep), "Classes that inherit from TruncableMixin must also inherit from BaseStep."

    def patch_missing_names(self, steps_as_tuple: List) -> NamedTupleList:
        names_yet = set()
        patched = []
        for step in steps_as_tuple:

            class_name = step.__class__.__name__
            if isinstance(step, tuple):
                class_name = step[0]
                step = step[1]
                if class_name in names_yet:
                    warnings.warn(
                        "Named pipeline tuples must be unique. "
                        "Will rename '{}' because it already exists.".format(class_name))

            # Add suffix number to name if it is already used to ensure name uniqueness.
            _name = class_name
            i = 1
            while _name in names_yet:
                _name = class_name + str(i)
                i += 1

            step = (_name, step)
            names_yet.add(step[0])
            patched.append(step)
        return patched

    def _refresh_steps(self):
        """
        Private method to refresh inner state after having edited ``self.steps_as_tuple``
        (recreate ``self.steps`` from ``self.steps_as_tuple``).
        """
        self.steps: OrderedDict = OrderedDict(self.steps_as_tuple)

    def get_hyperparams(self, flat=False) -> HyperparameterSamples:
        ret = dict()

        for k, v in self.steps.items():
            hparams = v.get_hyperparams()  # TODO: oop diamond problem?
            if hasattr(v, "hyperparams"):
                hparams.update(v.hyperparams)
            if len(hparams) > 0:
                ret[k] = hparams

        if flat:
            ret = HyperparameterSamples(ret)
        return ret

    def set_hyperparams(self, hyperparams: Union[HyperparameterSamples, OrderedDict, dict]) -> BaseStep:
        hyperparams: HyperparameterSamples = HyperparameterSamples(hyperparams).to_nested_dict()

        remainders = dict()
        for name, hparams in hyperparams.items():
            if name in self.steps.keys():
                self.steps[name].set_hyperparams(hparams)
            else:
                remainders[name] = hparams
        self.hyperparams = remainders

        return self

    def set_hyperparams_space(self, hyperparams_space: Union[HyperparameterSpace, OrderedDict, dict]) -> BaseStep:
        hyperparams_space: HyperparameterSpace = HyperparameterSpace(hyperparams_space).to_nested_dict()

        remainders = dict()
        for name, hparams in hyperparams_space.items():
            if name in self.steps.keys():
                self.steps[name].set_hyperparams_space(hparams)
            else:
                remainders[name] = hparams
        self.hyperparams = remainders

        return self

    def get_hyperparams_space(self, flat=False):
        all_hyperparams = HyperparameterSpace()
        for step_name, step in self.steps_as_tuple:
            hspace = step.get_hyperparams_space(flat=flat)
            all_hyperparams.update({
                step_name: hspace
            })
        all_hyperparams.update(
            super().get_hyperparams_space()
        )
        if flat:
            all_hyperparams = all_hyperparams.to_flat()
        else:
            all_hyperparams = all_hyperparams.to_nested_dict()
        return all_hyperparams

    def mutate(self, new_method="inverse_transform", method_to_assign_to="transform", warn=True) -> 'BaseStep':
        """
        Call mutate on every steps the the present truncable step contains.

        :param new_method: the method to replace transform with.
        :param method_to_assign_to: the method to which the new method will be assigned to.
        :param warn: (verbose) wheter or not to warn about the inexistence of the method.
        :return: self, a copy of self, or even perhaps a new or different BaseStep object.
        """
        if self.pending_mutate[0] is None:
            new_base_step = self
            self.pending_mutate = (new_base_step, self.pending_mutate[1], self.pending_mutate[2])

            new_base_step.steps_as_tuple = [
                (
                    k,
                    v.mutate(new_method, method_to_assign_to, warn)
                )
                for k, v in new_base_step.steps_as_tuple
            ]
            new_base_step._refresh_steps()
            return super().mutate(new_method, method_to_assign_to, warn)
        else:
            return super().mutate(new_method, method_to_assign_to, warn)

    def __getitem__(self, key):
        if isinstance(key, slice):

            self_shallow_copy = copy(self)

            start = key.start
            stop = key.stop
            step = key.step
            if step is not None or (start is None and stop is None):
                raise KeyError("Invalid range: '{}'.".format(key))
            new_steps_as_tuple = []

            if start is None:
                if stop not in self.steps.keys():
                    raise KeyError("Stop '{}' not found in '{}'.".format(stop, self.steps.keys()))
                for key, val in self.steps_as_tuple:
                    if stop == key:
                        break
                    new_steps_as_tuple.append((key, val))
            elif stop is None:
                if start not in self.steps.keys():
                    raise KeyError("Start '{}' not found in '{}'.".format(stop, self.steps.keys()))
                for key, val in reversed(self.steps_as_tuple):
                    new_steps_as_tuple.append((key, val))
                    if start == key:
                        break
                new_steps_as_tuple = list(reversed(new_steps_as_tuple))
            else:
                started = False
                if stop not in self.steps.keys() or start not in self.steps.keys():
                    raise KeyError(
                        "Start or stop ('{}' or '{}') not found in '{}'.".format(start, stop, self.steps.keys()))
                for key, val in self.steps_as_tuple:
                    if stop == key:
                        break
                    if not started and start == key:
                        started = True
                    if started:
                        new_steps_as_tuple.append((key, val))

            self_shallow_copy.steps_as_tuple = new_steps_as_tuple
            self_shallow_copy.steps = OrderedDict(new_steps_as_tuple)
            return self_shallow_copy
        else:
            return self.steps[key]

    def items(self):
        return self.steps.items()

    def keys(self):
        return self.steps.keys()

    def values(self):
        return self.steps.values()

    def append(self, item: Tuple[str, 'BaseStep']):
        self.steps_as_tuple.append(item)
        self._refresh_steps()

    def pop(self) -> 'BaseStep':
        return self.popitem()[-1]

    def popitem(self, key=None) -> Tuple[str, 'BaseStep']:
        if key is None:
            item = self.steps_as_tuple.pop()
            self._refresh_steps()
        else:
            item = key, self.steps.pop(key)
            self.steps_as_tuple = list(self.steps.items())
        return item

    def popfront(self) -> 'BaseStep':
        return self.popfrontitem()[-1]

    def popfrontitem(self) -> Tuple[str, 'BaseStep']:
        item = self.steps_as_tuple.pop(0)
        self._refresh_steps()
        return item

    def __contains__(self, item):
        """
        Check wheter the ``item`` key or value (or key value tuple pair) is found in self.

        :param item: The key or value to check if is in self's keys or values.
        :return: True or False
        """
        return item in self.steps.keys() or item in self.steps.values() or item in self.items()

    def __iter__(self):
        """
        Iterate through the steps.

        :return: iter(self.steps_as_tuple)
        """
        return iter(self.steps_as_tuple)

    def __len__(self):
        """
        Return the number of contained steps.

        :return: len(self.steps_as_tuple)
        """
        return len(self.steps_as_tuple)


class BaseBarrier(ABC):
    # TODO: a barrier is between steps and manages how they interact (e.g.: a checkpoint).
    pass


class BaseBlockBarrier(BaseBarrier, ABC):
    # TODO: a block barrier forces not using any "_one" functions past that barrier.
    pass


class BaseStreamingBarrier(BaseBarrier, ABC):
    # TODO: a streaming barrier forces using the "_one" functions past that barrier.
    pass


class BasePipelineRunner(MetaStepsMixin, BaseStep, ABC):

    @abstractmethod
    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BasePipelineRunner', Any):
        pass

    @abstractmethod
    def fit(self, data_inputs, expected_outputs=None) -> 'BasePipelineRunner':
        pass

    @abstractmethod
    def transform(self, data_inputs):
        pass
