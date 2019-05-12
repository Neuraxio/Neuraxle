# Copyright 2019, The Neuraxle Authors
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

import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import copy
from typing import Tuple, List, Union

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

    def set_hyperparams(self, hyperparams: HyperparameterSamples):
        self.hyperparams = hyperparams

    def get_hyperparams(self) -> HyperparameterSamples:
        return self.hyperparams

    def set_hyperparams_space(self, hyperparams_space: HyperparameterSpace):
        self.hyperparams_space = hyperparams_space

    def get_hyperparams_space(self) -> HyperparameterSpace:
        return self.hyperparams_space

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

    def transform(self, data_inputs):
        processed_outputs = [self.transform_one(data_input) for data_input in data_inputs]
        return processed_outputs

    def inverse_transform(self, processed_outputs):
        data_inputs = [self.inverse_transform_one(data_output) for data_output in processed_outputs]
        return data_inputs

    def predict(self, data_input):
        return self.transform(data_input)

    def mutate(self, new_method="inverse_transform", method_to_assign_to="transform", warn=True) -> 'BaseStep':
        """
        Replace the "method_to_assign_to" method by the "new_method" method, IF the present object has no pending calls to
        `.will_mutate_to()` waiting to be applied. If there is a pending call, the pending call will override the
        methods specified in the present call. If the change fails (such as if the new_method doesn't exist), then
        a warning is printed (optional). By default, there is no pending `will_mutate_to` call.

        This could for example be useful within a pipeline to apply `inverse_transform` to every pipeline steps, or
        to assign `predict_probas` to `predict`, or to assign "inverse_transform" to "transform" to a reversed pipeline.

        :param new_method: the method to replace transform with, if there is no pending `will_mutate_to` call.
        :param method_to_assign_to: the method to which the new method will be assigned to, if there is no pending `will_mutate_to` call.
        :param warn: (verbose) wheter or not to warn about the inexistence of the method.
        :return: self, a copy of self, or even perhaps a new or different BaseStep object.
        """
        pending_new_base_step, pending_new_method, pending_method_to_assign_to = self.pending_mutate

        # Use everything that is pending if they are not none (ternaries).
        new_base_step = pending_new_base_step if pending_new_base_step is not None else self
        new_method = pending_new_method if pending_new_method is not None else new_method
        method_to_assign_to = pending_method_to_assign_to if pending_method_to_assign_to is not None else method_to_assign_to

        try:
            new_method = getattr(self, new_method)

            # We set "new_method" in place of "method_to_affect" to a copy of self:
            if id(self) == id(new_base_step):
                new_base_step = copy(self)  # shallow copy if the new_base_step is self.
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
        This will change the behavior of `self.mutate(<...>)` such that when mutating, it will return the
        presently provided new_base_step BaseStep (can be left to None for self), and the `.mutate` method
        will also apply the `new_method` and the  `method_to_affect`, if they are not None, and after changing
        the object to new_base_step.

        This can be useful if your pipeline requires unsupervised pretraining. For example:

        ```
        X_pretrain = ...
        X_train = ...

        p = Pipeline(
            SomePreprocessing(),
            SomePretrainingStep().will_mutate_to(new_base_step=SomeStepThatWillUseThePretrainingStep),
            Identity().will_mutate_to(new_base_step=ClassifierThatWillBeUsedOnlyAfterThePretraining)
        )
        # Pre-train the pipeline
        p.fit(X_pretrain, y=None)

        # This will leave `SomePreprocessing()` untouched and will affect the two other steps.
        p.mutate(new_method="transform", method_to_affect="transform")

        # Pre-train the pipeline
        p.fit(X_train, y_train)  # Then fit the classifier and other new things
        ```

        :param new_base_step: if it is not None, upon calling `mutate`, the object it will mutate to will be this provided new_base_step.
        :param method_to_assign_to: if it is not None, upon calling `mutate`, the method_to_affect will be the one that is used on the provided new_base_step.
        :param new_method: if it is not None, upon calling `mutate`, the new_method will be the one that is used on the provided new_base_step.
        :return: self
        """
        if new_method is None or method_to_assign_to is None:
            new_method = method_to_assign_to = "transform"  # No changes will be applied (transform will stay transform).

        self.pending_mutate = (new_base_step, new_method, method_to_assign_to)

        return self

    def fit_one(self, data_input, expected_output=None):
        # return self
        raise NotImplementedError("TODO")

    def transform_one(self, data_input):
        # return processed_output
        raise NotImplementedError("TODO")

    def inverse_transform_one(self, data_output):
        # return data_input
        raise NotImplementedError("TODO")

    def tosklearn(self) -> 'NeuraxleToSKLearnPipelineWrapper':
        from sklearn.base import BaseEstimator as be

        class NeuraxleToSKLearnPipelineWrapper(be):
            def __init__(self, neuraxle_step):
                self.p: Union[BaseStep, TruncableSteps] = neuraxle_step

            def set_params(self, **params):
                self.p.set_hyperparams(HyperparameterSpace(params))

            def get_params(self, deep=True):
                neuraxle_params = HyperparameterSamples(self.p.get_hyperparams()).to_flat_as_dict_primitive()
                return neuraxle_params

            def get_params_space(self, deep=True):
                neuraxle_params = HyperparameterSpace(self.p.get_hyperparams_space()).to_flat_as_dict_primitive()
                return neuraxle_params

            def fit(self, **args):
                return self.p.fit(**args)

            def transform(self, **args):
                return self.p.transform(**args)

            def fit_transform(self, **args):
                return self.p.fit_transform(**args)

            def inverse_transform(self, **args):
                return self.p.inverse_transform(**args)

            def predict(self, **args):
                return self.p.transform(**args)

        return NeuraxleToSKLearnPipelineWrapper(self)


class NonFittableMixin:
    def fit(self, data_inputs, expected_outputs=None):
        return self

    def fit_one(self, data_input, expected_output=None):
        return self


NamedTupleList = List[Union[Tuple[str, 'BaseStep'], 'BaseStep']]


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
        Private method to refresh inner state after having edited `self.steps_as_tuple`
        (recreate `self.steps` from `self.steps_as_tuple`).
        """
        self.steps: OrderedDict = OrderedDict(self.steps_as_tuple)

    def get_hyperparams(self, flat=False) -> HyperparameterSamples:
        ret = dict()

        for k, v in self.steps.items():
            hparams = v.get_hyperparams()  # TODO: oop diamond problem?
            if hasattr(v, "hyparparams"):
                hparams.update(v.hyperparams)
            if len(hparams) > 0:
                ret[k] = hparams

        if flat:
            ret = HyperparameterSamples(ret)
        return ret

    def set_hyperparams(self, hyperparams: Union[HyperparameterSamples, OrderedDict, dict]):
        hyperparams: HyperparameterSamples = HyperparameterSamples(hyperparams).to_nested_dict()

        remainders = dict()
        for name, hparams in hyperparams.items():
            if name in self.steps.keys():
                self.steps[name].set_hyperparams(hparams)
            else:
                remainders[name] = hparams
        self.hyperparams = remainders

    def set_hyperparams_space(self, hyperparams_space: Union[HyperparameterSpace, OrderedDict, dict]):
        hyperparams_space: HyperparameterSpace = HyperparameterSpace(hyperparams_space)

        remainders = dict()
        for name, hparams in hyperparams_space.items():
            if name in self.steps.keys():
                self.steps[name].set_hyperparams_space(hparams)
            else:
                remainders[name] = hparams
        self.hyperparams = remainders

    def get_hyperparams_space(self, flat=False):
        all_hyperparams = HyperparameterSpace()
        for step_name, step in self.steps_as_tuple:
            all_hyperparams.update(
                step.get_hyperparams_space(flat=flat)
            )
        all_hyperparams.update(
            super().get_hyperparams_space()
        )
        return all_hyperparams

    def mutate(self, new_method="inverse_transform", method_to_assign_to="transform", warn=True) -> 'BaseStep':
        """
        Call mutate on every steps the the present truncable step contains.

        :param new_method: the method to replace transform with.
        :param method_to_assign_to: the method to which the new method will be assigned to.
        :param warn: (verbose) wheter or not to warn about the inexistence of the method.
        :return: self, a copy of self, or even perhaps a new or different BaseStep object.
        """
        self.steps_as_tuple = [(k, v.mutate(new_method, method_to_assign_to, warn)) for k, v in self.steps_as_tuple]
        self._refresh_steps()

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
        Check wheter the `item` key or value (or key value tuple pair) is found in self.

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


class ReversibleTruncableSteps(TruncableSteps, ABC):
    """Inherit from this to make a `TruncableSteps` class that is reversible (such as for doing inverse_transform)."""

    def __reversed__(self):
        """
        Iterate the steps in reverse order.

        :return: an iterator for which every item is a tuple of (step_name, base_step), in reverse order.
        """
        return reversed(self.steps_as_tuple)


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
        self.steps_as_tuple: NamedTupleList = None

    def set_steps(self, steps_as_tuple: NamedTupleList) -> 'PipelineRunner':
        self.steps_as_tuple: NamedTupleList = steps_as_tuple
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
