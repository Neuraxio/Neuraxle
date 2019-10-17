"""
Neuraxle's Base Classes
====================================
This is the core of Neuraxle. Most pipeline steps derive (inherit) from those classes. They are worth noticing.

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

import hashlib
import inspect
import os
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import copy
from typing import Tuple, List, Union, Any, Iterable

from conv import convolved_1d
from joblib import dump, load

from neuraxle.hyperparams.space import HyperparameterSpace, HyperparameterSamples

DEFAULT_CACHE_FOLDER = os.path.join(os.getcwd(), 'cache')


class BaseHasher(ABC):
    @abstractmethod
    def hash(self, current_ids, hyperparameters: HyperparameterSamples, data_inputs: Any):
        raise NotImplementedError()

    def _hash_hyperparameters(self, hyperparams: HyperparameterSamples):
        hyperperams_dict = hyperparams.to_flat_as_dict_primitive()
        return hashlib.md5(str.encode(str(hyperperams_dict))).hexdigest()


class HashlibMd5Hasher:
    def hash(self, current_ids, hyperparameters, data_inputs: Any = None):
        if current_ids is None:
            current_ids = [str(i) for i in range(len(data_inputs))]

        if len(hyperparameters) == 0:
            return current_ids

        hyperperams_dict = hyperparameters.to_flat_as_dict_primitive()
        current_hyperparameters_hash = hashlib.md5(str.encode(str(hyperperams_dict))).hexdigest()

        new_current_ids = []
        for current_id in current_ids:
            m = hashlib.md5()
            m.update(str.encode(current_id))
            m.update(str.encode(current_hyperparameters_hash))
            new_current_ids.append(m.hexdigest())

        return new_current_ids


class DataContainer:
    def __init__(self,
                 current_ids,
                 data_inputs: Any,
                 expected_outputs: Any = None
                 ):
        self.current_ids = current_ids
        self.data_inputs = data_inputs
        if expected_outputs is None:
            self.expected_outputs = [None] * len(current_ids)
        else:
            self.expected_outputs = expected_outputs

    def set_data_inputs(self, data_inputs: Any):
        self.data_inputs = data_inputs

    def set_expected_outputs(self, expected_outputs: Any):
        self.expected_outputs = expected_outputs

    def set_current_ids(self, current_ids: Any):
        self.current_ids = current_ids

    def convolved_1d(self, stride, kernel_size) -> Iterable:
        conv_current_ids = convolved_1d(stride=stride, iterable=self.current_ids, kernel_size=kernel_size)
        conv_data_inputs = convolved_1d(stride=stride, iterable=self.data_inputs, kernel_size=kernel_size)
        conv_expected_outputs = convolved_1d(stride=stride, iterable=self.expected_outputs, kernel_size=kernel_size)

        for current_ids, data_inputs, expected_outputs in zip(conv_current_ids, conv_data_inputs,
                                                              conv_expected_outputs):
            yield DataContainer(
                current_ids=current_ids,
                data_inputs=data_inputs,
                expected_outputs=expected_outputs
            )

    def __iter__(self):
        current_ids = self.current_ids
        if self.current_ids is None:
            current_ids = [None] * len(self.data_inputs)

        expected_outputs = self.expected_outputs
        if self.expected_outputs is None:
            expected_outputs = [None] * len(self.data_inputs)

        return zip(current_ids, self.data_inputs, expected_outputs)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.__class__.__name__ + "(current_ids=" + repr(list(self.current_ids)) + ", ...)"

    def __len__(self):
        return len(self.data_inputs)


class ListDataContainer(DataContainer):
    @staticmethod
    def empty() -> 'ListDataContainer':
        return ListDataContainer([], [], [])

    def append(self, current_id, data_input, expected_output):
        self.current_ids.append(current_id)
        self.data_inputs.append(data_input)
        self.expected_outputs.append(expected_output)

    def concat(self, data_container: DataContainer):
        self.current_ids.extend(data_container.current_ids)
        self.data_inputs.extend(data_container.data_inputs)
        self.expected_outputs.extend(data_container.expected_outputs)


class BaseSaver(ABC):
    """
    Any saver must inherit from this one. Some savers just save parts of objects, some save it all or what remains.
    """

    @abstractmethod
    def save_step(self, step: 'BaseStep', context: 'ExecutionContext') -> 'BaseStep':
        """
        Save step with execution context.

        :param step: step to save
        :param context: execution context
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def can_load(self, step: 'BaseStep', context: 'ExecutionContext'):
        raise NotImplementedError()

    @abstractmethod
    def load_step(self, step: 'BaseStep', context: 'ExecutionContext') -> 'BaseStep':
        """
        Load step with execution context.

        :param step: step to load
        :param context: execution context to load from
        :return: loaded base step
        """
        raise NotImplementedError()


class JoblibStepSaver(BaseSaver):
    """
    This saver is a good default saver when the object is
    already stripped out of things that would make it unserializable.
    """

    def can_load(self, step: 'BaseStep', context: 'ExecutionContext'):
        return os.path.exists(
            self._create_step_path(context, step)
        )

    def _create_step_path(self, context, step):
        return os.path.join(context.get_path(), '{0}.joblib'.format(step.name))

    def save_step(self, step: 'BaseStep', context: 'ExecutionContext') -> 'BaseStep':
        """
        Saved step stripped out of things that would make it unserializable.

        :param step: stripped step to save
        :param context: execution context to save from
        :return:
        """
        context.mkdir()

        path = self._create_step_path(context, step)
        dump(step, path)

        return step

    def load_step(self, step: 'BaseStep', context: 'ExecutionContext') -> 'BaseStep':
        """
        Load stripped step.

        :param step: stripped step to load
        :param context: execution context to load from
        :return:
        """
        loaded_step = load(self._create_step_path(context, step))

        # we need to keep the current steps in memory because they have been deleted before saving...
        # the steps that have not been saved yet need to be in memory while loading a truncable steps...
        if isinstance(step, TruncableSteps):
            loaded_step.steps = step.steps

        return loaded_step


class ExecutionContext:
    """
    Execution context object containing all of the pipeline hierarchy steps.
    First item in execution context parents is root, second is nested, and so on. This is like a stack.
    """

    def __init__(
            self,
            root: str = DEFAULT_CACHE_FOLDER,
            stripped_saver: BaseSaver = None,
            parents=None
    ):
        if stripped_saver is None:
            stripped_saver: BaseSaver = JoblibStepSaver()

        self.stripped_saver = stripped_saver
        self.root: str = root
        if parents is None:
            parents = []
        self.parents: List[BaseStep] = parents

    @staticmethod
    def from_root(root_step: 'BaseStep', root_path) -> 'ExecutionContext':
        return ExecutionContext(root=root_path, parents=[root_step])

    def save_all_unsaved(self):
        """
        Save all unsaved steps in the parents of the execution context.

        :return:
        """
        copy_self = copy(self)
        while not copy_self.empty():
            if copy_self.should_save_last_step():
                copy_self.peek().save(copy_self)
            copy_self.pop()

    def should_save_last_step(self):
        """
        Returns True if the last step should be saved.

        :return:
        """
        if len(self.parents) > 0:
            return self.parents[-1].should_save()
        return False

    def pop_item(self) -> 'BaseStep':
        """
        Change the execution context to be the same as the latest parent context.

        :return:
        """
        return self.parents.pop()

    def pop(self) -> bool:
        """
        Pop the context.
        :return:
        """
        if len(self) == 0:
            return False
        self.pop_item()
        return True

    def push(self, step: 'BaseStep') -> 'ExecutionContext':
        """
        Pushes a step in the parents of the execution context.

        :param step: step to add to the execution context
        :return: self
        """
        return ExecutionContext(
            root=self.root,
            parents=self.parents + [step]
        )

    def peek(self):
        """
        Get last parent.

        :return:
        """
        return self.parents[-1]

    def mkdir(self):
        """
        Creates the directory to save the last parent step.

        :return:
        """
        path = self.get_path()
        if not os.path.exists(path):
            os.makedirs(path)

    def get_path(self):
        """
        Creates the directory path for the current execution context.

        :return: current context path
        :rtype: str
        """
        parents_with_path = [self.root] + [p.name for p in self.parents]
        return os.path.join(*parents_with_path)

    def get_names(self):
        """
        Returns a list of the parent names.

        :return: list of parents step names
        :rtype: list(str)
        """
        return [p.name for p in self.parents]

    def empty(self):
        """
        Return True if the context has parent steps.

        :return: if parents len is 0
        :rtype: bool
        """
        return len(self) == 0

    def __len__(self):
        return len(self.parents)


class BaseStep(ABC):
    """
    Base class for a pipeline step.

    Note : All heavy initialization logic should be done inside the *setup* method (e.g.: things inside GPU),
    and NOT in the constructor.

    """
    def __init__(
            self,
            hyperparams: HyperparameterSamples = None,
            hyperparams_space: HyperparameterSpace = None,
            name: str = None,
            savers: List[BaseSaver] = None,
            hashers: List[BaseHasher] = None
    ):
        if hyperparams is None:
            hyperparams = dict()
        if hyperparams_space is None:
            hyperparams_space = dict()
        if name is None:
            name = self.__class__.__name__
        if savers is None:
            savers = []
        if hashers is None:
            hashers = [HashlibMd5Hasher()]

        self.hyperparams: HyperparameterSamples = HyperparameterSamples(hyperparams)
        self.hyperparams = self.hyperparams.to_flat()

        self.hyperparams_space: HyperparameterSpace = HyperparameterSpace(hyperparams_space)
        self.hyperparams_space = self.hyperparams_space.to_flat()

        self.name: str = name

        self.savers: List[BaseSaver] = savers  # TODO: doc. First is the most stripped.
        self.hashers: List[BaseHasher] = hashers

        self.pending_mutate: ('BaseStep', str, str) = (None, None, None)
        self.is_initialized = False
        self.is_invalidated = True
        self.is_train: bool = True

    def hash(self, current_ids, hyperparameters, data_inputs: Any = None):
        for h in self.hashers:
            current_ids = h.hash(current_ids, hyperparameters, data_inputs)
        return current_ids

    def setup(self) -> 'BaseStep':
        """
        Initialize step before it runs. Only from here and not before that heavy things should be created
        (e.g.: things inside GPU), and NOT in the constructor.

        :return: self
        """
        self.is_initialized = True
        self.is_invalidated = True
        return self

    def teardown(self):
        """
        Teardown step after program execution. This is like the inverse of setup, and it clears memory.

        :return:
        """
        self.is_initialized = False
        return self

    def set_train(self, is_train: bool=True):
        """
        Set pipeline step mode to train or test.

        For instance, you can add a simple if statement to direct to the right implementation:
        `
            def transform(self, data_inputs):
                if self.is_train:
                    self.transform_train_(data_inputs)
                else:
                    self.transform_test_(data_inputs)

            def fit_transform(self, data_inputs, expected_outputs):
                if self.is_train:
                    self.fit_transform_train_(data_inputs, expected_outputs)
                else:
                    self.fit_transform_test_(data_inputs, expected_outputs)
        `

        :param is_train: bool
        :return:
        """
        self.is_train = is_train
        return self

    def set_name(self, name: str):
        """
        Set the name of the pipeline step.

        :param name: a string.
        :return: self
        """
        self.name = name
        self.is_invalidated = True
        return self

    def get_name(self) -> str:
        """
        Get the name of the pipeline step.

        :return: the name, a string.
        """
        return self.name

    def get_savers(self) -> List[BaseSaver]:
        """
        Get the step savers of a pipeline step.

        :return:
        """
        return self.savers

    def set_savers(self, savers: List[BaseSaver]) -> 'BaseStep':
        """
        Set the step savers of a pipeline step.

        :return: self
        :rtype: BaseStep
        """
        self.savers = savers
        return self

    def set_hyperparams(self, hyperparams: HyperparameterSamples) -> 'BaseStep':
        self.is_invalidated = True
        self.hyperparams = HyperparameterSamples(hyperparams).to_flat()
        return self

    def get_hyperparams(self) -> HyperparameterSamples:
        return self.hyperparams

    def set_params(self, **params) -> 'BaseStep':
        return self.set_hyperparams(HyperparameterSamples(params))

    def get_params(self) -> dict:
        return self.get_hyperparams().to_flat_as_ordered_dict_primitive()

    def set_hyperparams_space(self, hyperparams_space: HyperparameterSpace) -> 'BaseStep':
        self.is_invalidated = True
        self.hyperparams_space = HyperparameterSpace(hyperparams_space).to_flat()
        return self

    def get_hyperparams_space(self) -> HyperparameterSpace:
        return self.hyperparams_space

    def handle_fit(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        """
        Perform any needed side effects on the step or the data container before fitting the step.

        :param data_container: the data container to transform
        :param context: execution context
        :return: tuple(fitted pipeline, data_container)
        """
        self.is_invalidated = True

        new_self = self.fit(data_container.data_inputs, data_container.expected_outputs)

        current_ids = self.hash(data_container.current_ids, self.hyperparams, data_container.data_inputs)
        data_container.set_current_ids(current_ids)

        return new_self, data_container

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> (
            'BaseStep', DataContainer):
        """
        Update the data inputs inside DataContainer after fit transform,
        and update its current_ids.

        :param data_container: the data container to transform
        :param context: execution context
        :return: tuple(fitted pipeline, data_container)
        """
        self.is_invalidated = True

        new_self, out = self.fit_transform(data_container.data_inputs, data_container.expected_outputs)
        data_container.set_data_inputs(out)

        current_ids = self.hash(data_container.current_ids, self.hyperparams, out)
        data_container.set_current_ids(current_ids)

        return new_self, data_container

    def handle_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Update the data inputs inside DataContainer after transform.

        :param data_container: the data container to transform
        :param context: execution context
        :return: transformed data container
        """
        out = self.transform(data_container.data_inputs)
        data_container.set_data_inputs(out)

        current_ids = self.hash(data_container.current_ids, self.hyperparams, out)
        data_container.set_current_ids(current_ids)

        return data_container

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BaseStep', Any):
        self.is_invalidated = True

        new_self = self.fit(data_inputs, expected_outputs)
        out = new_self.transform(data_inputs)

        return new_self, out

    @abstractmethod
    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        raise NotImplementedError("TODO: Implement this method in {}, or have this class inherit from the NonFittableMixin.".format(self.__class__.__name__))

    @abstractmethod
    def transform(self, data_inputs):
        raise NotImplementedError("TODO: Implement this method in {}, or have this class inherit from the NonTransformableMixin.".format(self.__class__.__name__))

    def inverse_transform(self, processed_outputs):
        raise NotImplementedError("TODO: Implement this method in {}.".format(self.__class__.__name__))

    def predict(self, data_input):
        return self.transform(data_input)

    def should_save(self) -> bool:
        """
        Returns true if the step should be saved.

        :return: bool
        """
        return self.is_invalidated and self.is_initialized

    def save(self, context: ExecutionContext):
        """
        Save step using the execution context to create the directory to save the step into.

        :param context: context to save from
        :return:
        """
        if self.is_invalidated and self.is_initialized:
            self.is_invalidated = False

            context.mkdir()
            stripped_step = copy(self)

            # A final "visitor" saver will save anything that
            # wasn't saved customly after stripping the rest.
            savers_with_provided_default_stripped_saver = [context.stripped_saver] + self.savers

            for saver in reversed(savers_with_provided_default_stripped_saver):
                # Each saver strips the step a bit more if needs be.
                stripped_step = saver.save_step(stripped_step, context)

            return stripped_step

        return self

    def load(self, context: ExecutionContext) -> 'BaseStep':
        """
        Load step using the execution context to create the directory of the saved step.
        Warning: please do not override this method because on loading it is an identity
        step that will load whatever step you coded.

        :param context: execution context to load step from
        :return:
        """
        # A final "visitor" saver might reload anything that wasn't saved customly after stripping the rest.
        savers_with_provided_default_stripped_saver = [context.stripped_saver] + self.savers

        loaded_self = self
        for saver in savers_with_provided_default_stripped_saver:
            # Each saver unstrips the step a bit more if needed
            if saver.can_load(loaded_self, context):
                loaded_self = saver.load_step(loaded_self, context)
            else:
                warnings.warn('Cannot Load Step {0} ({1}:{2}) With Step Saver {3}.'.format(context.get_path(), self.name, self.__class__.__name__, saver.__class__.__name__))
                break

        return loaded_self

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
        self.is_invalidated = True
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
            self.is_invalidated = True

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
        self.is_invalidated = True

        if new_method is None or method_to_assign_to is None:
            new_method = method_to_assign_to = "transform"  # No changes will be applied (transform will stay transform).

        self.pending_mutate = (new_base_step, new_method, method_to_assign_to)

        return self

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

    # TODO: remove equal None, and fix random search at the same time ?
    def __init__(
            self,
            wrapped: BaseStep = None
    ):
        self.wrapped: BaseStep = wrapped

    def setup(self) -> BaseStep:
        """
        Initialize step before it runs

        :param context: execution context
        :return: self
        :rtype: BaseStep
        """
        BaseStep.setup(self)
        self.wrapped.setup()
        return self

    def set_train(self, is_train: bool=True):
        self.is_train = is_train
        self.wrapped.set_train(is_train)
        return self

    def set_hyperparams(self, hyperparams: HyperparameterSamples) -> BaseStep:
        """
        Set meta step and wrapped step hyperparams using the given hyperparams.

        :param hyperparams: ordered dict containing all hyperparameters
        :type hyperparams: HyperparameterSamples

        :return:
        """
        self.is_invalidated = True

        hyperparams: HyperparameterSamples = HyperparameterSamples(hyperparams).to_nested_dict()

        remainders = dict()
        for name, hparams in hyperparams.items():
            if name == self.wrapped.name:
                self.wrapped.set_hyperparams(hparams)
            else:
                remainders[name] = hparams

        self.hyperparams = HyperparameterSamples(remainders)

        return self

    def get_hyperparams(self) -> HyperparameterSamples:
        """
        Get meta step and wrapped step hyperparams as a flat hyperparameter samples.

        :return: hyperparameters
        """
        return HyperparameterSamples({
            **self.hyperparams.to_flat_as_dict_primitive(),
            self.wrapped.name: self.wrapped.hyperparams.to_flat_as_dict_primitive()
        }).to_flat()

    def set_hyperparams_space(self, hyperparams_space: HyperparameterSpace) -> 'BaseStep':
        """
        Set meta step and wrapped step hyperparams space using the given hyperparams space.

        :param hyperparams_space: ordered dict containing all hyperparameter spaces
        :type hyperparams_space: HyperparameterSpace

        :return: self
        """
        self.is_invalidated = True

        hyperparams_space: HyperparameterSpace = HyperparameterSpace(hyperparams_space.to_nested_dict())

        remainders = dict()
        for name, hparams in hyperparams_space.items():
            if name == self.wrapped.name:
                self.wrapped.set_hyperparams_space(hparams)
            else:
                remainders[name] = hparams

        self.hyperparams_space = HyperparameterSpace(remainders)

        return self

    def get_hyperparams_space(self) -> HyperparameterSpace:
        """
        Get meta step and wrapped step hyperparams as a flat hyperparameter space

        :return: hyperparameters_space
        """
        return HyperparameterSpace({
            **self.hyperparams_space.to_flat_as_dict_primitive(),
            self.wrapped.name: self.wrapped.hyperparams_space.to_flat_as_dict_primitive()
        }).to_flat()

    def set_step(self, step: BaseStep) -> BaseStep:
        self.is_invalidated = True
        self.wrapped: BaseStep = step
        return self

    def get_best_model(self) -> BaseStep:
        return self.best_model


NamedTupleList = List[Union[Tuple[str, 'BaseStep'], 'BaseStep']]


class NonFittableMixin:
    """
    A pipeline step that requires no fitting: fitting just returns self when called to do no action.
    Note: fit methods are not implemented
    """

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext):
        return self, self.handle_transform(data_container, context)

    def fit(self, data_inputs, expected_outputs=None) -> 'NonFittableMixin':
        """
        Don't fit.

        :param data_inputs: the data that would normally be fitted on.
        :param expected_outputs: the data that would normally be fitted on.
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

    def inverse_transform(self, processed_outputs):
        """
        Do nothing - return the same data.

        :param processed_outputs: the data to process
        :return: the ``processed_outputs``, unchanged.
        """
        return processed_outputs


class TruncableJoblibStepSaver(JoblibStepSaver):
    """
    Step saver for a TruncableSteps.
    TruncableJoblibStepSaver saves, and loads all of the sub steps using their savers.
    """

    def __init__(self):
        super().__init__()

    def save_step(self, step: 'TruncableSteps', context: ExecutionContext):
        """
        Save truncable steps.

        :param step: step to save
        :param context: execution context
        :return:
        """
        # First, save all of the sub steps with the right execution context.
        sub_steps_savers = []
        for i, (_, sub_step) in enumerate(step):
            if sub_step.should_save():
                sub_context = context.push(sub_step)
                sub_step.save(sub_context)
                sub_steps_savers.append((step[i].name, step[i].get_savers()))
            else:
                sub_steps_savers.append((step[i].name, None))

        step.sub_steps_savers = sub_steps_savers

        # Third, strip the sub steps from truncable steps before saving
        del step.steps
        del step.steps_as_tuple

        return step

    def load_step(self, step: 'TruncableSteps', context: ExecutionContext) -> 'TruncableSteps':
        """
        Load truncable steps.

        :param step: step to load
        :param context: execution context
        :return:
        """
        step.steps_as_tuple = []

        for step_name, savers in step.sub_steps_savers:
            if savers is None:
                # keep step as it is if it hasn't been saved
                step.steps_as_tuple.append((step_name, step[step_name]))
            else:
                # Load each sub step with their savers
                sub_step_to_load = Identity(name=step_name, savers=savers)
                subcontext = context.push(sub_step_to_load)

                sub_step = sub_step_to_load.load(subcontext)
                step.steps_as_tuple.append((step_name, sub_step))

        step._refresh_steps()

        return step


class TruncableSteps(BaseStep, ABC):

    def __init__(
            self,
            steps_as_tuple: NamedTupleList,
            hyperparams: HyperparameterSamples = dict(),
            hyperparams_space: HyperparameterSpace = dict()
    ):
        BaseStep.__init__(self, hyperparams=hyperparams, hyperparams_space=hyperparams_space)
        self._set_steps(steps_as_tuple)

        self.set_savers([TruncableJoblibStepSaver()] + self.savers)

    def are_steps_before_index_the_same(self, other: 'TruncableSteps', index: int):
        """
        Returns true if self.steps before index are the same as other.steps before index.

        :param other: other truncable steps to compare
        :param index: max step index to compare

        :return: bool
        """
        steps_before_index = self[:index]
        for current_index, (step_name, step) in enumerate(steps_before_index):
            source_current_step = inspect.getsource(step.__class__)
            source_cached_step = inspect.getsource(other[current_index].__class__)

            if source_current_step != source_cached_step:
                return False

        return True

    def _load_saved_pipeline_steps_before_index(self, saved_pipeline: 'TruncableSteps', index: int):
        """
        Load the cached pipeline steps
        before the index into the current steps

        :param saved_pipeline:
        :param index:
        :return:
        """
        self.set_hyperparams(saved_pipeline.get_hyperparams())
        self.set_hyperparams_space(saved_pipeline.get_hyperparams_space())

        new_truncable_steps = saved_pipeline[:index] + self[index:]
        self._set_steps(new_truncable_steps.steps_as_tuple)

    def _set_steps(self, steps_as_tuple: NamedTupleList):
        """
        Set pipeline steps

        :param steps_as_tuple: List[Tuple[str, BaseStep]] list of tuple containing step name and step
        :return:
        """
        self.steps_as_tuple: NamedTupleList = self.patch_missing_names(steps_as_tuple)
        self._refresh_steps()

    def setup(self) -> 'BaseStep':
        """
        Initialize step before it runs

        :return: self
        """
        if self.is_initialized:
            return self

        self.is_initialized = True

        return self

    def teardown(self) -> 'BaseStep':
        for step_name, step in self.steps_as_tuple:
            step.teardown()

        return self

    def patch_missing_names(self, steps_as_tuple: List) -> NamedTupleList:
        names_yet = set()
        patched = []
        for step in steps_as_tuple:

            if isinstance(step, tuple):
                class_name = step[0]
                step = step[1]
            else:
                class_name = step.get_name()

            _name = class_name
            if class_name in names_yet:
                warnings.warn(
                    "Named pipeline tuples must be unique. "
                    "Will rename '{}' because it already exists.".format(class_name))

                _name = self._rename_step(step_name=_name, class_name=class_name, names_yet=names_yet)
                step.set_name(_name)

            step = (_name, step)
            names_yet.add(step[0])
            patched.append(step)
        self.is_invalidated = True
        return patched

    def _rename_step(self, step_name, class_name, names_yet):
        """
        Rename step by adding a number suffix after the class name.
        Ensure uniqueness with the names yet parameter.
        :param step_name:
        :param class_name:
        :param names_yet:
        :return:
        """
        # Add suffix number to name if it is already used to ensure name uniqueness.
        i = 1
        while step_name in names_yet:
            step_name = class_name + str(i)
            i += 1
        self.is_invalidated = True
        return step_name

    def _refresh_steps(self):
        """
        Private method to refresh inner state after having edited ``self.steps_as_tuple``
        (recreate ``self.steps`` from ``self.steps_as_tuple``).
        """
        self.is_invalidated = True
        self.steps: OrderedDict = OrderedDict(self.steps_as_tuple)
        for name, step in self.items():
            step.name = name

    def get_hyperparams(self) -> HyperparameterSamples:
        hyperparams = dict()

        for k, v in self.steps.items():
            hparams = v.get_hyperparams()  # TODO: oop diamond problem?
            if hasattr(v, "hyperparams"):
                hparams.update(v.hyperparams)
            if len(hparams) > 0:
                hyperparams[k] = hparams

        hyperparams = HyperparameterSamples(hyperparams)

        return hyperparams.to_flat()

    def set_hyperparams(self, hyperparams: Union[HyperparameterSamples, OrderedDict, dict]) -> BaseStep:
        self.is_invalidated = True

        hyperparams: HyperparameterSamples = HyperparameterSamples(hyperparams).to_nested_dict()

        remainders = dict()
        for name, hparams in hyperparams.items():
            if name in self.steps.keys():
                self.steps[name].set_hyperparams(hparams)
            else:
                remainders[name] = hparams
        self.hyperparams = HyperparameterSamples(remainders)

        return self

    def get_hyperparams_space(self):
        all_hyperparams = HyperparameterSpace()
        for step_name, step in self.steps_as_tuple:
            hspace = step.get_hyperparams_space()
            all_hyperparams.update({
                step_name: hspace
            })
        all_hyperparams.update(
            super().get_hyperparams_space()
        )

        return all_hyperparams.to_flat()

    def set_hyperparams_space(self, hyperparams_space: Union[HyperparameterSpace, OrderedDict, dict]) -> BaseStep:
        self.is_invalidated = True

        hyperparams_space: HyperparameterSpace = HyperparameterSpace(hyperparams_space).to_nested_dict()

        remainders = dict()
        for name, hparams in hyperparams_space.items():
            if name in self.steps.keys():
                self.steps[name].set_hyperparams_space(hparams)
            else:
                remainders[name] = hparams
        self.hyperparams = HyperparameterSpace(remainders)

        return self


    def should_save(self):
        if BaseStep.should_save(self):
            return True

        for _, step in self.items():
            if step.should_save():
                return True
        return False

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

    def _step_name_to_index(self, step_name):
        for index, (current_step_name, step) in self.steps_as_tuple:
            if current_step_name == step_name:
                return index

    def _step_index_to_name(self, step_index):
        if step_index == len(self.items()):
            return None

        name, _ = self.steps_as_tuple[step_index]
        return name

    def __getitem__(self, key):
        if isinstance(key, slice):
            self_shallow_copy = copy(self)

            if isinstance(key.start, int):
                start = self._step_index_to_name(key.start)
            else:
                start = key.start

            if isinstance(key.stop, int):
                stop = self._step_index_to_name(key.stop)
            else:
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
                    if start == stop == key:
                        new_steps_as_tuple.append((key, val))

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
            if isinstance(key, int):
                key = self._step_index_to_name(key)

            return self.steps[key]

    def __add__(self, other: 'TruncableSteps') -> 'TruncableSteps':
        self._set_steps(self.steps_as_tuple + other.steps_as_tuple)
        return self

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

    def split(self, step_type: type) -> List['TruncableSteps']:
        """
        Split truncable steps by a step class name.

        :param step_type: step class type to split from.
        :return: list of truncable steps containing the splitted steps
        """
        sub_pipelines = []

        previous_sub_pipeline_end_index = 0
        for index, (step_name, step) in enumerate(self.items()):
            if isinstance(step, step_type):
                sub_pipelines.append(
                    self[previous_sub_pipeline_end_index:index + 1]
                )
                previous_sub_pipeline_end_index = index + 1

        if previous_sub_pipeline_end_index < len(self.items()):
            sub_pipelines.append(
                self[previous_sub_pipeline_end_index:-1]
            )

        return sub_pipelines

    def ends_with(self, step_type: type):
        return isinstance(self[-1], step_type)

    def set_train(self, is_train: bool=True) -> 'BaseStep':
        """
        Set pipeline step mode to train or test.

        In the pipeline steps functions, you can add a simple if statement to direct to the right implementation:
        `
            def transform(self, data_inputs):
                if self.is_train:
                    self.transform_train_(data_inputs)
                else:
                    self.transform_test_(data_inputs)

            def fit_transform(self, data_inputs, expected_outputs):
                if self.is_train:
                    self.fit_transform_train_(data_inputs, expected_outputs)
                else:
                    self.fit_transform_test_(data_inputs, expected_outputs)
        `

        :param is_train: bool
        :return: self
        """
        self.is_train = is_train
        for _, step in self.items():
            step.set_train(is_train)
        return self



class ResumableStepMixin:
    """
    A step that can be resumed, for example a checkpoint on disk.
    """

    @abstractmethod
    def should_resume(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        raise NotImplementedError()


class Barrier(NonFittableMixin, NonTransformableMixin, BaseStep, ABC):
    """
    A Barrier step to be used in a minibatch sequential pipeline. It forces all the
    data inputs to get to the barrier in a sub pipeline before going through to the next sub-pipeline.


    ``p = MiniBatchSequentialPipeline([
        SomeStep(),
        SomeStep(),
        Barrier(), # must be a concrete Barrier ex: Joiner()
        SomeStep(),
        SomeStep(),
        Barrier(), # must be a concrete Barrier ex: Joiner()
    ], batch_size=10)``

    """

    @abstractmethod
    def join_transform(self, step: BaseStep, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        raise NotImplementedError()

    @abstractmethod
    def join_fit_transform(self, step: BaseStep, data_container: DataContainer, context: ExecutionContext) -> Tuple[
        'Any', Iterable]:
        raise NotImplementedError()


class Joiner(Barrier):
    """
    A Special Barrier step that joins the transformed mini batches together with list.extend method.
    """

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def join_transform(self, step: BaseStep, data_container: DataContainer, context: ExecutionContext) -> Iterable:
        """
        Concatenate the pipeline transform output of each batch of self.batch_size together.

        :param step: pipeline to transform on
        :param data_container: data container to transform
        :param context: execution context
        :return:
        """
        data_container_batches = data_container.convolved_1d(
            stride=self.batch_size,
            kernel_size=self.batch_size
        )

        output_data_container = ListDataContainer.empty()
        for data_container_batch in data_container_batches:
            output_data_container.concat(
                step._transform_core(data_container_batch, context)
            )

        return output_data_container

    def join_fit_transform(self, step: BaseStep, data_container: DataContainer, context: ExecutionContext) -> \
            Tuple['Any', Iterable]:
        """
        Concatenate the pipeline fit transform output of each batch of self.batch_size together.

        :param step: pipeline to fit transform on
        :param data_container: data container to fit transform on
        :param context: execution context
        :return: fitted self, transformed data inputs
        """
        data_container_batches = data_container.convolved_1d(
            stride=self.batch_size,
            kernel_size=self.batch_size
        )

        output_data_container = ListDataContainer.empty()
        for data_container_batch in data_container_batches:
            step, data_container_batch = step._fit_transform_core(data_container_batch, context)
            output_data_container.concat(
                data_container_batch
            )

        return step, output_data_container


class Identity(NonTransformableMixin, NonFittableMixin, BaseStep):
    """A pipeline step that has no effect at all but to return the same data without changes.

    This can be useful to concatenate new features to existing features, such as what AddFeatures do.

    Identity inherits from ``NonTransformableMixin`` and from ``NonFittableMixin`` which makes it a class that has no
    effect in the pipeline: it doesn't require fitting, and at transform-time, it returns the same data it received.
    """
    pass  # Multi-class inheritance does the job here! See inside those other classes for more info.
