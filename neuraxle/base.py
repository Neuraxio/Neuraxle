"""
Neuraxle's Base Classes
====================================
This is the core of Neuraxle. They are worth noticing.
Most classes inherit from these classes, composing them differently.

This package ensures proper respect of the Interface Segregation Principle (ISP),
That is a SOLID principle of OOP programming suggesting to segregate interfaces.
This is what is done here as the project gained in abstraction, and that the base
classes needed to compose other base classes.

..
    Copyright 2022, Neuraxio Inc.

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
import datetime
import inspect
import logging
import os
import pickle
import pprint
import shutil
import sys
import tempfile
import traceback
import warnings
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import copy, deepcopy
from enum import Enum
from multiprocessing import RLock
from operator import attrgetter
from typing import (Any, Callable, Dict, Generic, ItemsView, Iterable, KeysView, List, Optional, Set, Tuple, Type,
                    TypeVar, Union, ValuesView)

from joblib import dump, load

from neuraxle.data_container import ARG_X_INPUTTED, ARG_Y_EXPECTED, ARG_Y_PREDICTD, DIT, EOT, IDT
from neuraxle.data_container import DataContainer as DACT
from neuraxle.data_container import PredsDACT, TrainDACT
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace, RecursiveDict
from neuraxle.logging.logging import NEURAXLE_LOGGER_NAME, NeuraxleLogger, ParallelLoggingConsumerThread
from neuraxle.logging.warnings import warn_deprecated_arg


class BaseSaver(ABC):
    """
    Any saver must inherit from this one. Some savers just save parts of objects, some save it all or what remains.
    Each :class`BaseStep` can potentially have multiple savers to make serialization possible.

    .. seealso::
        :func:`~neuraxle.base._HasSavers.save`,
        :func:`~neuraxle.base._HasSavers.load`
    """

    @abstractmethod
    def save_step(self, step: 'BaseTransformer', context: 'CX') -> 'BaseTransformer':
        """
        Save a step or a step's parts using the execution context.

        :param step: step to save
        :param context: execution context
        :param save_savers:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def can_load(self, step: 'BaseTransformer', context: 'CX'):
        """
        Returns true if we can load the given step with the given execution context.

        :param step: step to load
        :param context: execution context to load from
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def load_step(self, step: 'BaseTransformer', context: 'CX') -> 'BaseTransformer':
        """
        Load step with execution context.

        :param step: step to load
        :param context: execution context to load from
        :return: loaded base step
        """
        raise NotImplementedError()


class JoblibStepSaver(BaseSaver):
    """
    Saver that can save, or load a step with
    `joblib.load <https://joblib.readthedocs.io/en/latest/generated/joblib.load.html>`_,
    and `joblib.dump <https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html>`_.

    This saver is a good default saver when the object is
    already stripped out of things that would make it unserializable.

    It is the default stripped_saver for the :class:`ExecutionContext`.
    The stripped saver is the first to load the step, and the last to save the step.
    The saver receives a *stripped* version of the step so that it can be saved by joblib.

    .. seealso::
        :class:`~neuraxle.base.BaseSaver`,
        :class:`~neuraxle.base.ExecutionContext`
    """

    def can_load(self, step: 'BaseTransformer', context: 'CX') -> bool:
        """
        Returns true if the given step has been saved with the given execution context.

        :param step: step that might have been saved
        :param context: execution context
        :return: if we can load the step with the given context
        """
        return os.path.exists(
            self._get_step_path(context, step)
        )

    def _get_step_path(self, context, step):
        """
        Create step path for the given context.

        :param context: execution context
        :param step: step to save, or load
        :return: path
        """
        return os.path.join(context.get_path(), '{0}.joblib'.format(step.name))

    def save_step(self, step: 'BaseTransformer', context: 'CX') -> 'BaseTransformer':
        """
        Saved step stripped out of things that would make it unserializable.

        :param step: stripped step to save
        :param context: execution context to save from
        :return:
        """
        context.mkdir()

        path = self._get_step_path(context, step)
        dump(step, path)

        return step

    def load_step(self, step: 'BaseTransformer', context: 'CX') -> 'BaseTransformer':
        """
        Load stripped step.

        :param step: stripped step to load
        :param context: execution context to load from
        :return:
        """
        loaded_step = load(self._get_step_path(context, step))

        # we need to keep the current steps in memory because they have been deleted before saving...
        # the steps that have not been saved yet need to be in memory while loading a truncable steps...
        if isinstance(loaded_step, TruncableSteps) and hasattr(step, 'steps'):
            loaded_step.steps = step.steps

        return loaded_step


class ExecutionMode(Enum):
    """
    This enum defines the execution mode of a :class:`~neuraxle.base.BaseStep`.
    It is available in the :class:`~neuraxle.base.ExecutionContext` as :attr:`~neuraxle.base.BaseStep.execution_mode`.
    """
    FIT_OR_FIT_TRANSFORM_OR_TRANSFORM = 'fit_or_fit_transform_or_transform'
    FIT_OR_FIT_TRANSFORM = 'fit_or_fit_transform'
    TRANSFORM = 'transform'
    FIT = 'fit'
    FIT_TRANSFORM = 'fit_transform'
    INVERSE_TRANSFORM = 'inverse_transform'


class ExecutionPhase(Enum):
    """
    This enum defines the execution phase of a :class:`~neuraxle.base.BaseStep`.
    It is available in the :class:`~neuraxle.base.ExecutionContext` as :attr:`~neuraxle.base.BaseStep.execution_mode`.
    """
    UNSPECIFIED = None
    PRETRAIN = "pretraining"
    TRAIN = "training"
    VALIDATION = "validation"
    TEST = "test"
    PROD = "production"


class MixinForBaseService:
    """
    Any steps/transformers within a pipeline that inherits of this class should implement BaseStep/BaseTransformer and
    initialize it before any mixin. This class checks that its the case at initialization.
    """

    def __init__(self):
        self._ensure_baseservice_init_called()

    def _ensure_baseservice_init_called(self):
        """
        Assert that BaseTransformer's init method has been called.
        """
        assert isinstance(self, BaseService), "This class should be of type BaseService."
        if not all(map(lambda x: hasattr(self, x), ('apply', 'get_config', 'set_config'))):
            raise RuntimeError(
                f'Please initialize Mixins in the good order. The present Mixin should '
                f'be initialized after BaseService. '
                f'Got: {inspect.getmro(self.__class__)}. '
                f'Visit https://www.neuraxle.org/stable/classes_and_modules_overview.html '
                f'for more information.'
            )


class _RecursiveArguments:
    """
    This class is used by :func:`~neuraxle.base.BaseStep.apply`, and :class:`_HasChildrenMixin`
    to pass the right arguments to steps with children.

    Two types of arguments:
    - args: arguments that are not named
    - kwargs: arguments that are named

    For the values of both args and kwargs, we use either values or recursive values:
    - value is not RecursiveDict: the value is replicated and passed to each sub step.
    - value is RecursiveDict: the value is sliced accordingly and decomposed into the next levels.

    As a shorthand, if another _RecursiveArguments (ra) is passed as an argument, it is used almost as is
    to merge different ways of using ra: using a past ra, or else some args.

    .. seealso::
        :class:`_HasChildrenMixin`,
        :func:`~neuraxle.base._HasHyperparamsSpace.get_hyperparams_space`,
        :func:`~neuraxle.base._HasHyperparamsSpace.set_hyperparams_space`,
        :func:`~neuraxle.base._HasHyperparamsSpace.update_hyperparams_space`,
        :func:`~neuraxle.base._HasHyperparams.get_hyperparams`,
        :func:`~neuraxle.base._HasHyperparams.set_hyperparams`,
        :func:`~neuraxle.base._HasHyperparams.update_hyperparams`,
        :func:`~neuraxle.base._HasConfig.get_config`,
        :func:`~neuraxle.base._HasConfig.set_config`,
        :func:`~neuraxle.base._HasConfig.update_config`,
        :func:`~neuraxle.base.BaseTransformer.invalidate`
    """

    def __init__(
        self,
        ra=None,
        args: Union[List[Any], List[RecursiveDict]] = None,
        kwargs: Union[Dict[str, Any], RecursiveDict] = None,
        current_level: int = 0
    ):
        if ra is not None:
            args = ra.args
            kwargs = ra.kwargs
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()

        self.args: Union[List[Any], List[RecursiveDict]] = args
        self.kwargs: Union[Dict[str, Any], RecursiveDict] = kwargs
        self.current_level: int = current_level

    def __getitem__(self, child_step_name: str):
        """
        Return recursive arguments for the given child step name.
        If child step name is None, return the values at the current root.

        :param child_step_name: child step name, or None if we want to get root values.
        :return: recursive argument for the given child step name
        """
        arguments = []

        for arg in self.args:
            if isinstance(arg, RecursiveDict):
                arguments.append(arg.get(child_step_name))
            else:
                arguments.append(arg)

        keyword_arguments = RecursiveDict()
        for key, arg in self.kwargs.items():
            if isinstance(arg, RecursiveDict):
                keyword_arguments[key] = arg.get(child_step_name)
            else:
                keyword_arguments[key] = arg

        next_level = self.current_level + 1

        return _RecursiveArguments(None, arguments, keyword_arguments, next_level)

    def children_names(self) -> List[str]:
        """
        Return the names of the children steps.

        :return: list of children step names
        """
        argvals = (list(self.args) + list(self.kwargs.values()))
        # Checking if any of the arguments (v) is a RecursiveDict and if so,
        # take all it's children names (keys) for childs who are also
        # RecursiveDict for being themselves steps:
        return set(sum([
            list([
                k
                for (k, vv) in v.items()
                if isinstance(vv, RecursiveDict)
            ])
            for v in argvals
            if isinstance(v, RecursiveDict)
        ], []))


class _HasRecursiveMethods:
    """
    An internal class to represent a step that has recursive methods.
    The apply :func:`apply` function is used to apply a method to a step and its children.

    Example usage:

    .. code-block:: python

        class _HasHyperparams:
            # ...
            def set_hyperparams(self, hyperparams: Union[HyperparameterSamples, Dict]) -> HyperparameterSamples:
                self.apply(method='_set_hyperparams', hyperparams=HyperparameterSamples(hyperparams))
                return self

            def _set_hyperparams(self, hyperparams: Union[HyperparameterSamples, Dict]) -> HyperparameterSamples:
                self._invalidate()
                hyperparams = HyperparameterSamples(hyperparams)
                self.hyperparams = hyperparams if len(hyperparams) > 0 else self.hyperparams
                return self.hyperparams

        pipeline = Pipeline([
            SomeStep()
        ])

        pipeline.set_hyperparams(HyperparameterSamples({
            'learning_rate': 0.1,
            'SomeStep__learning_rate': 0.05
        }))


    .. seealso::
        :class:`BaseStep`,
        :class:`BaseTransformer`,
        :class:`_HasChildrenMixin`
        :class:`_RecursiveArguments`
    """

    def __init__(self, name: str = None):
        if name is None:
            name = self.__class__.__name__
        self.name: str = name

    def set_name(self, name: str) -> 'BaseService':
        """
        Set the name of the service.

        :param name: a string.
        :return: self

        .. note::
            A step name is in the keys of :py:attr:`~neuraxle.base.TruncableSteps.steps_as_tuple`
        """
        self.name = name
        return self

    def get_name(self) -> str:
        """
        Get the name of the service.

        :return: the name, a string.

        .. note:: A step name is the same value as the one in the keys of :class:`Pipeline`.steps_as_tuple
        """
        return self.name

    def getattr(self, attr_name: str) -> RecursiveDict:
        """
        Get an attribute of the service or step, if it exists, returned as a :class:`RecursiveDict`.

        :param attr_name: the name of the attribute to get in each step or service.
        :return: A RecursiveDict with terminal leafs like ``RecursiveDict({attr_name: getattr(self, attr_name)})``.
        """
        return self.apply(method='_getattr', attr_name=attr_name)

    def _getattr(self, attr_name: str) -> 'RecursiveDict[str, str]':
        """
        Get an attribute if it exists, as a RecursiveDict({attr_name: getattr(self, attr_name)}).
        """
        if hasattr(self, attr_name):
            return RecursiveDict({attr_name: getattr(self, attr_name)})
        else:
            return RecursiveDict()

    def get_step_by_name(self, name: str) -> Optional['BaseServiceT']:
        if self.name == name:
            return self
        return None

    def apply(self, method: Union[str, Callable], ra: _RecursiveArguments = None, *args, **kwargs) -> RecursiveDict:
        """
        Apply a method to a step and its children.

        Here is an apply usage example to invalidate each steps.
        This example comes from the saving logic:

        .. code-block:: python

            # preparing to save steps and its nested children:
            if full_dump:
                # initialize and invalidate steps to make sure that all steps will be saved

                def _initialize_if_needed(step):
                    if not step.is_initialized:
                        step._setup(context=context)
                    if not step.is_initialized:
                        raise NotImplementedError(f"The `setup` method of the following class "
                                                f"failed to set `self.is_initialized` to True: {step.__class__.__name__}.")
                    return RecursiveDict()

                def _invalidate(step):
                    step._invalidate()
                    return RecursiveDict()

                self.apply(method=_initialize_if_needed)
                self.apply(method=_invalidate)

            # save steps:
            ...


        Here is another example. For instance, when setting the hyperparams space of a step,
        we use :func:`~neuraxle.base.BaseStep._set_hyperparams_space` to set the hyperparams of the step.
        The trick is that the space argument :class:`~neuraxle.hyperparams.space.HyperparameterSpace` is a recursive dict.
        The implementation is the same for setting the hyperparams and config
        of the step and its children, not only its space.
        The cool thing is that such hyperparameter spaces are recursive, inheriting from :class:`~neuraxle.hyperparams.space.RecursiveDict`.
        and applying recursive arguments to the step and its children with the :func:`_HasChildrenMixin.apply` of :class:`_HasChildrenMixin`.
        Here is the implementation, using apply:

        .. code-block:: python

            def set_hyperparams_space(self, hyperparams_space: HyperparameterSpace) -> 'BaseTransformer':
                self.apply(method='_set_hyperparams_space', hyperparams_space=HyperparameterSpace(hyperparams_space))
                return self

            def _set_hyperparams_space(self, hyperparams_space: Union[Dict, HyperparameterSpace]) -> HyperparameterSpace:
                self._invalidate()
                self.hyperparams_space = HyperparameterSpace(hyperparams_space)
                return self.hyperparams_space


        :param method: method name that need to be called on all steps
        :param ra: recursive arguments
        :param args: any additional arguments to be passed to the method
        :param kwargs: any additional positional arguments to be passed to the method
        :return: method outputs, or None if no method has been applied

        .. seealso::
            :class:`_RecursiveArguments`,
            :class:`_HasChildrenMixin`
        """
        ra = _RecursiveArguments(ra, args, kwargs)

        kargs = ra.args

        def _return_empty(*args, **kwargs):
            return RecursiveDict()

        _method = _return_empty
        if isinstance(method, str) and hasattr(self, method) and callable(getattr(self, method)):
            _method = getattr(self, method)

        if not isinstance(method, str):
            _method = method
            kargs = [self] + list(kargs)

        results = _method(*kargs, **ra.kwargs)
        if results is None:
            results = RecursiveDict()
        if not isinstance(results, RecursiveDict):
            raise ValueError(
                f'Method {method} of {self} must return None or a RecursiveDict, as it is applied recursively.')
        return results

    def __str__(self) -> str:
        """
        Return a pretty representation of the step or service.
        Use :func:`~neuraxle.base.BaseTransformer.__repr__`, for a more
        detailed string representation if needed.

        :return: return pretty representation such as ``StepName(name='StepName')``.
        """
        return self._repr(verbose=False)

    def __repr__(self) -> str:
        """
        Return a detailed and pretty representation of the pipeline step.
        Use :func:`~neuraxle.base.BaseTransformer.__str__`, for a less
        detailed string representation if needed.

        :return: return pretty representation, such as ``StepName(name='StepName', hyperparameters=HyperparameterSamples({...}))``.
        """
        return self._repr(verbose=True)

    def _repr(self, level=0, verbose=False):
        output = self.__class__.__name__ + '('
        output += self._repr_params(level=level, verbose=verbose).replace(', ', '', 1)
        return output + ')'

    def _repr_params(self, level=0, verbose=False) -> str:
        output = ''
        has_name = self.__class__.__name__ != self.name
        if has_name:
            output += ", name='" + self.name + "'"
        return output


class _HasConfig(ABC):
    """
    An internal class to represent a step that has config params.
    This is useful to store the config of a step.

    A config :class:`~neuraxle.hyperparams.space.RecursiveDict` config
    attribute is used when you don't want to use a
    :class:`~neuraxle.hyperparams.space.HyperparameterSamples` attribute.
    The reason sometimes is that you don't want to tune your config, whereas
    hyperparameters are used to tune your step in the AutoML
    from hyperparameter spaces, such as using hyperopt.

    A good example of a config parameter would be the number of threads,
    or an API key loaded from the OS' environment variables, since they won't
    be tuned but are changeable from the outside.

    Thus, this class looks a lot like :class:`~neuraxle.base._HasHyperparams`
    and :class:`~neuraxle.hyperparams.space.HyperparameterSpace`.

    .. seealso::
        :class:`BaseStep`,
        :class:`BaseTransformer`,
        :class:`~neuraxle.base._HasHyperparams`,
        :class:`~neuraxle.base._HasHyperparamsSpace`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`,
        :class:`~neuraxle.hyperparams.space.RecursiveDict`
    """

    def __init__(self, config: Union[Dict, RecursiveDict] = None):
        if config is None:
            config = dict()
        self.config: RecursiveDict = RecursiveDict(config)

    def set_config(self, config: Union[Dict, RecursiveDict]) -> 'BaseTransformer':
        """
        Set step config. See :func:`~neuraxle.base._HasHyperparams.set_hyperparams`
        for more usage examples and documentation, it works the same way.
        """
        if not isinstance(config, RecursiveDict):
            config = RecursiveDict(config)
        self.apply(method='_set_config', config=config)
        return self

    def _set_config(self, config: RecursiveDict) -> RecursiveDict:
        self.config = RecursiveDict(config)
        return self.config

    def update_config(self, config: Union[Dict, RecursiveDict]) -> 'BaseTransformer':
        """
        Update the step config variables without removing the already-set config variables.
        This method is similar to :func:`~neuraxle.base._HasHyperparams.update_hyperparams`.
        Refer to it for more documentation and usage examples, it works the same way.
        """
        if not isinstance(config, RecursiveDict):
            config = RecursiveDict(config)
        self.apply(method='_update_config', config=config)
        return self

    def _update_config(self, config: RecursiveDict) -> RecursiveDict:
        self.config.update(config)
        return self.config

    def get_config(self) -> RecursiveDict:
        """
        Get step config. Refer to :func:`~neuraxle.base._HasHyperparams.get_hyperparams`
        for more documentation and usage examples, it works the same way.
        """
        results: RecursiveDict = self.apply(method='_get_config')
        return results

    def _get_config(self) -> RecursiveDict:
        return self.config

    def _repr_params(self, level=0, verbose=False):
        if verbose:
            conf: RecursiveDict = self._get_config()
            if len(conf) > 0:
                return ", config=" + pprint.pformat(conf)
        return ''


class _CanMutate:
    """
    An internal class to represent a step that can be mutated.
    A step can replace some of its method by others.
    For example, you might want to reverse a step, and replace the transform method by the inverse transform method.
    .. seealso::
        :class:`BaseStep`,
        :class:`BaseTransformer`,
        :func:`~neuraxle.base._TransformerStep.transform`,
        :func:`~neuraxle.base._TransformerStep.inverse_transform`
    """

    def __init__(self):
        self.pending_mutate: Tuple['BaseTransformer', str, str] = (None, None, None)

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
        self._invalidate()
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
            except AttributeError:
                pass

            # 3. assign new method to old method
            setattr(new_base_step, method_to_assign_to, new_method)
            self._invalidate()

        except AttributeError as e:
            if warn:
                import warnings
                warnings.warn(repr(e))

        return new_base_step

    def will_mutate_to(self, new_base_step: 'BaseTransformer' = None, new_method: str = None,
                       method_to_assign_to: str = None) -> 'BaseTransformer':
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
        self._invalidate()

        if new_method is None or method_to_assign_to is None:
            # No changes will be applied (transform will stay transform).
            new_method = method_to_assign_to = "transform"

        self.pending_mutate = (new_base_step, new_method, method_to_assign_to)

        return self


class BaseService(
    _CanMutate,
    _HasConfig,
    _HasRecursiveMethods,
    ABC
):
    """
    Base class for all services registred into the :class:`ExecutionContext`.

    .. seealso::
        :class:`ExecutionContext`,
        :class:`BaseTransformer`,
        :class:`BaseStep`,
    """

    def __init__(self, config: Union[Dict, RecursiveDict] = None, name: str = None):
        _HasRecursiveMethods.__init__(self, name=name)
        _HasConfig.__init__(self, config=config)
        _CanMutate.__init__(self)

    def _repr_params(self, level=0, verbose=False):
        output = ""
        _name = _HasRecursiveMethods._repr_params(self, level=level, verbose=verbose)
        output += _name
        _config = _HasConfig._repr_params(self, level=level, verbose=verbose)
        if not (len(_name) > 0):
            _config = _config.replace(', ', '', 1)
        output += _config
        return output


BaseServiceT = TypeVar('BaseServiceT', bound=BaseService)


class _HasChildrenMixin(MixinForBaseService, Generic[BaseServiceT]):
    """
    Mixin to add behavior to the steps that have children (sub steps).

    .. seealso::
        :class:`MixinForBaseTransformer`
        :class:`~neuraxle.base.MetaStepMixin`,
        :class:`~neuraxle.base.TruncableSteps`
    """

    def apply(self, method: Union[str, Callable], ra: _RecursiveArguments = None, *args, **kwargs) -> RecursiveDict:
        """
        Apply method to root, and children steps.
        Split the root, and children values inside the arguments of type RecursiveDict.

        This method overrides the :func:`~neuraxle.base._HasRecursiveMethods.apply` method
        of :class:`~neuraxle.base._HasRecursiveMethods`. Read the documentation of the
        original method to learn more.

        Read more: `Steps containing other steps <https://www.neuraxle.org/stable/classes_and_modules_overview.html#steps-containing-other-steps-as-the-composite-design-pattern-in-machine-learning>`_.

        :param method: str or callable function to apply
        :param ra: recursive arguments
        :return:
        """
        ra: _RecursiveArguments = _RecursiveArguments(ra, args, kwargs)
        self._validate_children_exists(ra)
        # #################################################################
        results: RecursiveDict = self._apply_self(method=method, ra=ra)
        # Documentation: https://www.neuraxle.org/stable/classes_and_modules_overview.html#steps-containing-other-steps-as-the-composite-design-pattern-in-machine-learning
        results: RecursiveDict = self._apply_childrens(results=results, method=method, ra=ra)
        # #################################################################
        return results

    def _apply_self(self, method: Union[str, Callable], ra: _RecursiveArguments) -> RecursiveDict:
        terminal_ra: _RecursiveArguments = ra[None]
        self_results: RecursiveDict = BaseStep.apply(self, method=method, ra=terminal_ra)
        return self_results

    def _apply_childrens(
            self, results: RecursiveDict, method: Union[str, Callable], ra: _RecursiveArguments) -> RecursiveDict:

        children: List[BaseServiceT] = self.get_children()
        # Add context to the children steps if we are at the level 0:
        if ra.current_level == 0:
            cx: CX = [c for c in list(ra.args) + list(ra.kwargs.values())
                      if isinstance(c, CX)]
            if len(cx) > 0:
                children.append(cx[0])

        for child in self.get_children():
            child_ra: _RecursiveArguments = ra[child.get_name()]
            children_results: RecursiveDict = child.apply(method=method, ra=child_ra)
            results[child.get_name()] = children_results
        return results

    def _validate_children_exists(self, ra):
        """
        Validate that the provided childrens are in self, and if not, raise an error.
        """
        children_names: Set[str] = set(self.get_children_names())
        ra_children_names: Set[str] = ra.children_names()
        unknown_names: Set[str] = ra_children_names - children_names
        if len(unknown_names) > 0:
            raise KeyError(f'{unknown_names} not children of {self.name}. Available childrens are: {children_names}.')

    @abstractmethod
    def get_children(self) -> List[BaseServiceT]:
        """
        Get the list of all the childs for that step or service.

        :return: every child steps
        """
        pass

    def get_children_names(self) -> List[str]:
        """
        Get the list of all the childs names for that step.

        :return: every child steps' names
        """
        return [child.get_name() for child in self.get_children()]

    def get_step_by_name(self, name: str) -> Optional[BaseServiceT]:
        if self.name == name:
            return self
        # loop in self.get_children() to find the step with the given name:
        for child in self.get_children():
            subchild: BaseServiceT = child.get_step_by_name(name)
            if subchild is not None:
                return subchild
        return None


class MetaServiceMixin(_HasChildrenMixin):
    """
    A mixin for services containing other services
    """

    def __init__(self, wrapped: BaseServiceT):
        _HasChildrenMixin.__init__(self)
        self.wrapped: BaseServiceT = wrapped

    def set_step(self, step: BaseServiceT) -> BaseServiceT:
        """
        Set wrapped step to the given step.

        :param step: new wrapped step
        :return: self
        """
        self._invalidate()
        self.wrapped: BaseServiceT = _sklearn_to_neuraxle_step(step)
        return self

    def get_step(self) -> BaseServiceT:
        """
        Get wrapped step

        :return: self.wrapped
        """
        return self.wrapped

    def get_children(self) -> List[BaseServiceT]:
        """
        Get the list of all the childs for that step.
        :class:`_HasChildrenMixin` calls this method to apply methods to all of the childs for that step.

        :return: list of child steps

        .. seealso::
            :class:`_HasChildrenMixin`
        """
        return [self.wrapped]

    def _repr(self, level=0, verbose=False) -> str:
        output = self.__class__.__name__ + "("
        output += self.wrapped._repr(level=level + 1, verbose=verbose)
        output += self._repr_params(level, verbose)
        output += ")"
        return output

    def mutate(self, new_method="inverse_transform", method_to_assign_to="transform", warn=False) -> 'BaseTransformer':
        """
        Mutate self, and self.wrapped. Please refer to :func:`~neuraxle.base._CanMutate.mutate` for more information.
        :param new_method: the method to replace transform with, if there is no pending ``will_mutate_to`` call.
        :param method_to_assign_to: the method to which the new method will be assigned to, if there is no pending ``will_mutate_to`` call.
        :param warn: (verbose) wheter or not to warn about the inexistence of the method.
        :return: self, a copy of self, or even perhaps a new or different BaseStep object.
        """
        new_self = super().mutate(new_method, method_to_assign_to, warn)
        new_self.wrapped = self.wrapped.mutate(new_method, method_to_assign_to, warn)

        return new_self

    def will_mutate_to(
        self, new_base_step: 'BaseTransformer' = None, new_method: str = None, method_to_assign_to: str = None
    ) -> 'BaseTransformer':
        """
        Add pending mutate self, self.wrapped. Please refer to :func:`~neuraxle.base._CanMutate.will_mutate_to` for more information.
        :param new_base_step: if it is not None, upon calling ``mutate``, the object it will mutate to will be this provided new_base_step.
        :param method_to_assign_to: if it is not None, upon calling ``mutate``, the method_to_affect will be the one that is used on the provided new_base_step.
        :param new_method: if it is not None, upon calling ``mutate``, the new_method will be the one that is used on the provided new_base_step.
        :return: self
        """
        new_self = super().will_mutate_to(new_base_step, new_method, method_to_assign_to)
        return new_self


class MetaService(MetaServiceMixin, BaseService):
    """
    A service containing other services.
    """

    def __init__(
            self,
            wrapped: BaseServiceT = None,
            config: RecursiveDict = None,
            name: str = None
    ):
        BaseService.__init__(
            self,
            config=config,
            name=name,
        )
        MetaServiceMixin.__init__(self, wrapped=wrapped)


ServiceName = Union[str, Type[BaseServiceT]]
NamedServicesList = List[Union[Tuple[str, BaseServiceT], BaseServiceT]]


class _TruncableMixin(MixinForBaseService):
    # TODO: Merge common code of TruncableServiceMixin and TruncableStepsMixin into this.
    def __init__(self):
        MixinForBaseService.__init__(self)

    def mutate(self, new_method="inverse_transform", method_to_assign_to="transform", warn=False) -> 'BaseTransformer':
        """
        Call mutate on every steps the the present truncable step contains.

        :param new_method: the method to replace transform with.
        :param method_to_assign_to: the method to which the new method will be assigned to.
        :param warn: (verbose) wheter or not to warn about the inexistence of the method.
        :return: self, a copy of self, or even perhaps a new or different BaseStep object.
        """
        if self.pending_mutate[0] is None:
            new_base_step = BaseStep.mutate(self, new_method, method_to_assign_to, warn)
            self.pending_mutate = (new_base_step, self.pending_mutate[1], self.pending_mutate[2])

            new_base_step.steps_as_tuple = [
                (
                    k,
                    v.mutate(new_method, method_to_assign_to, warn)
                )
                for k, v in new_base_step.steps_as_tuple
            ]
            new_base_step._refresh_steps()
            return new_base_step
        else:
            # Since we're remplacing ourselves with a new step, we don't have to call mutate on our childrens since
            # they won't exist afterward.
            return BaseStep.mutate(self, new_method, method_to_assign_to, warn)

    def _repr(self, level=0, verbose=False) -> str:

        output = self.__class__.__name__ + "("
        output += self._repr_children(level, verbose)
        output += self._repr_params(level, verbose)
        output += ")"
        return output

    def _repr_children(self, level, verbose) -> str:
        """
        Returns a string representation of the children of the step like this:

        .. code-block:: python
            output = '''[
                ChildrenA,
                ChildrenB,
                ChildrenC
            ]'''

        """
        output = ""

        children: List[BaseService] = self.get_children()
        is_compact: bool = len(children) < 2

        tab0 = "    " * level
        tab1 = "    " * (level + 1)
        _nl = "\n"
        _nl2 = _nl
        if is_compact:
            tab1 = ""
            _nl = ""
            _nl2 = " "

        output += "["
        childs_reprs = []
        for child in children:
            try:
                if hasattr(child, '_repr'):
                    c_repr = child._repr(level=level + 1, verbose=verbose)
                else:
                    c_repr = repr(child)
            except:
                # raise Exception(f"Could not repr child `{child}` of self `{self}`:\n{e}") from e
                # c_repr = child._repr(level=level + 1, verbose=verbose)
                c_repr = repr(child)  # breakpoint here if needed.
            childs_reprs.append(c_repr)

        output += _nl + tab1 + ("," + _nl2 + tab1).join(childs_reprs)
        output += _nl + (tab1 if is_compact else tab0) + "]"

        return output


class _TruncableServiceWithBodyMixin(MixinForBaseService):
    """
    This is a mixin to enable the .joiner and .body methods to be
    used on a truncable step that has a joiner at its end.
    """

    def __init__(self):
        MixinForBaseService.__init__(self)

    @property
    def joiner(self) -> BaseService:
        """
        returns `self[-1]`
        """
        return self[-1]

    @property
    def body(self) -> List[BaseService]:
        """
        returns `list(self.values())[:-1]`, that is all the steps except the last joiner.
        """
        return list(self.values())[:-1]

    @property
    def named_body(self) -> List[BaseService]:
        """
        returns `list(self.values())[:-1]`, that is all the steps except the last joiner.
        """
        return self[:-1]


class TruncableServiceMixin(_TruncableMixin, _HasChildrenMixin):

    def __init__(self, services: Dict[ServiceName, 'BaseServiceT']):
        _HasChildrenMixin.__init__(self)
        _TruncableMixin.__init__(self)
        self.set_services(services)

    def set_services(self, services: Dict[ServiceName, 'BaseServiceT']):
        services = services or {}
        services = {self._sanitize_service_name(k): v for k, v in services.items()}
        self.services: Dict[str, 'BaseServiceT'] = services
        return self

    def _sanitize_service_name(self, service_name: ServiceName) -> str:
        if isinstance(service_name, str):
            return service_name
        return service_name.__name__  # is a class / type

    def register_service(
        self, service_type: ServiceName, service_instance: 'BaseServiceT'
    ) -> 'CX':
        """
        Register base class instance inside the services. This is useful to register services.
        Make sure the service is an instance of the class :class:`~neuraxle.base.BaseService`.

        :param service_type: base type
        :param service_instance:  instance
        :return: self
        """
        service_type: str = self._sanitize_service_name(service_type)
        self[service_type] = service_instance
        return self

    def get_services(self) -> Dict[ServiceName, 'BaseServiceT']:
        """
        Get the registered instances in the services.

        :return: self
        """
        return self.services

    def get_service(self, service_type: ServiceName) -> object:
        """
        Get the registered instance for the given abstract class :class:`~neuraxle.base.BaseService` type.
        It is common to use service types as keys in the services dictionary.

        :param service_type: service type
        :return: self
        """
        service_type: str = self._sanitize_service_name(service_type)
        return self[service_type]

    def has_service(self, service_type: ServiceName) -> bool:
        """
        Return a bool indicating if the service has been registered.

        :param service_type: base type
        :return: if the service registered or not
        """
        return service_type in self

    def get_children(self) -> List[BaseServiceT]:
        """
        Get the list of all the childs for that step.
        :class:`_HasChildrenMixin` calls this method to apply methods to all of the childs for that step.

        :return: list of child steps

        .. seealso::
            :class:`_HasChildrenMixin`
        """
        return self.services.values()

    def __contains__(self, item: ServiceName) -> bool:
        """
        Return a bool indicating if the service has been registered.

        :param item: base type
        :return: if the service registered or not
        """
        return self._sanitize_service_name(item) in self.services

    def __getitem__(self, service_type: ServiceName) -> 'BaseServiceT':
        """
        Get the service from its base type key (or string equivalent of this key).

        :param service_type: base type
        :return: service
        """
        service_type: str = self._sanitize_service_name(service_type)
        return self.services[service_type]

    def __setitem__(self, service_type: ServiceName, service_instance: 'BaseServiceT'):
        """
        Set the service in the services dictionary.

        :param service_type: base type that is a type of :class:`~neuraxle.base.BaseService`
        :param service_instance: instance that is an instance of :class:`~neuraxle.base.BaseService`
        :return: self
        """
        service_type: str = self._sanitize_service_name(service_type)
        self.services[service_type] = service_instance
        return self


class TruncableService(TruncableServiceMixin, BaseService):

    def __init__(self, services: Dict[Type['BaseServiceT'], 'BaseServiceT'] = None):
        BaseService.__init__(self)
        TruncableServiceMixin.__init__(self, services)


class TrialStatus(Enum):
    """
    Enum of the possible states of a trial.
    """
    PLANNED = 'PLANNED'
    RUNNING = 'RUNNING'
    ABORTED = 'ABORTED'
    FAILED = 'FAILED'
    SUCCESS = 'SUCCESS'


class Flow(BaseService):
    """
    This is like a news feed for pipelines where you post (log) info.
    Flow is a step that can be used to store the status, metrics, logs,
    and other information of the execution of the current run.

    Concrete implementations of this object may interact with repositories.
    """

    def __init__(self, context: 'CX' = None):
        BaseService.__init__(self)
        self.context = weakref.proxy(context) if context else None

    def link_context(self, context: 'CX') -> 'Flow':
        """
        Link the context to the flow with a weak ref.
        """
        self.context = weakref.proxy(context)
        return self

    def unlink_context(self) -> 'Flow':
        """
        Unlink the context from the flow.
        """
        del self.context
        self.context: ExecutionContext = None
        return self

    @property
    def logger(self) -> NeuraxleLogger:
        return self.context.logger

    def _copy(self) -> 'Flow':
        """
        Copy the flow.
        """
        return copy(self)

    def log(self, message: str, level: int = logging.INFO, stacklevel=4):
        if sys.version_info.major <= 3 and sys.version_info.minor <= 7:
            self.logger.log(level, message)
        else:
            self.logger.log(level, message, stacklevel=stacklevel)

    def log_train_metric(self, metric_name: str, metric_value: float):
        """
        Log training metric
        """
        self.log(f'Train Metric `{metric_name}`: {metric_value}')

    def log_valid_metric(self, metric_name: str, metric_value: float):
        """
        Log validation metric
        """
        self.log(f'Valid Metric `{metric_name}`: {metric_value}')

    def log_model(self, model: 'BaseStep'):
        self.log(f'Model: {model}')

    def log_status(self, status: TrialStatus):
        self.log(f'Status: {status}')

    def log_planned(self, trial_id: int, hps: HyperparameterSamples):
        self.log(f'Trial #{trial_id} planned!')
        self.log_hps(hps)
        self.log_status(TrialStatus.PLANNED)

    def log_continued(self, trial_id: int):
        self.log(f'Trial #{trial_id} continued!')

    def log_retraining(self, trial_id: int, hps: HyperparameterSamples):
        self.log(f'Trial #{trial_id} will retrain!')
        self.log_hps(hps)
        self.log_status(TrialStatus.PLANNED)

    def log_hps(self, hps: HyperparameterSamples, use_wildcards=True):
        hps_str = pprint.pformat(hps.to_flat_dict(use_wildcards=use_wildcards), indent=4)
        self.log(f'Hyperparameters: \n{hps_str}')

    def log_start(self):
        self.log('Started!')
        self.log_status(TrialStatus.RUNNING)

    def log_epoch(self, epoch: int, n_epochs: int):
        self.log(f'Epoch {epoch}/{n_epochs}.')

    def log_end(self, status: TrialStatus = TrialStatus.SUCCESS):
        self.log('Finished!')
        self.log_status(status)

    def log_success(self, best_val_score: float = None, n_epochs_to_val_score: int = None, metric_name: str = None):
        self.log_end(TrialStatus.SUCCESS)
        if best_val_score is not None:
            self.log(f' ==> With best {metric_name} validation score: {best_val_score}')
        if n_epochs_to_val_score is not None:
            self.log(f' ==> At epoch: {n_epochs_to_val_score}')

    def log_best_hps(self, main_metric_name, best_hps: HyperparameterSamples, avg_validation_score: float, avg_n_epoch_to_best_validation_score: int):
        self.log(
            f"Best hyperparameters found for metric '{main_metric_name}' with "
            f"best validation score '{avg_validation_score}' "
            f"obtained at epoch '{avg_n_epoch_to_best_validation_score}':")
        self.log_hps(best_hps)

    def log_failure(self, exception: Exception):
        if exception is not None:
            self.log_error(exception)
        self.log_end(TrialStatus.FAILED)

    def log_error(self, exception: Exception):
        """
        Log an exception or error. The stack trace is logged as well.
        """
        self.log(f'The following {type(exception).__name__} occurred: {exception}', level=logging.ERROR)
        if exception is not None and exception.__traceback__ is not None:
            self.logger.exception(exception)

    def log_warning(self, message: str):
        self.log(message, level=logging.WARNING)

    def log_aborted(self, exception: Exception):
        """
        Probably aborted with CTRL+C
        """
        self.log('Aborted!')
        self.log_error(exception)
        self.log_status(TrialStatus.ABORTED)
        # TODO:
        # repo_trial_split_number = 0 if repo_trial_split is None else repo_trial_split.split_number + 1
        # trial_split_description = '{}/{} split {}/{}\nhyperparams: {}'.format(
        # trial_number + 1,
        # n_trial,
        # repo_trial_split_number + 1,
        # len(validation_splits),
        # json.dumps(repo_trial.hyperparams, sort_keys=True, indent=4)


class ExecutionContext(TruncableService):
    """
    Execution context object containing all of the pipeline hierarchy steps.
    First item in execution context parents is root, second is nested, and so on. This is like a stack.
    It tracks the current step, the current phase, the current execution mode, and the current saver,
    as well as other information such as the current path and other caching information.

    For instance, it is used in the handle_fit and handle_transform methods of the :class:`~neuraxle.base.BaseStep` as follows:
    :func:`~neuraxle.base.BaseStep.handle_fit`, :func:`~neuraxle.base.BaseStep.handle_transform`.

    This class can save and load steps using :class:`~neuraxle.base.BaseSaver` objects and the given root saving path.

    Like a service locator, it is used to access some registered services to be made available to the pipeline
    at every step when they process data.
    If pipeline steps are composed like a tree, the execution context is used to pass information between steps.
    Thus, some domain services can be registered in the execution context, and then used by the pipeline steps.

    One could design a lazy data loader that loads data only when needed, and have only the data IDs pass into the
    pipeline steps prior to hitting a step that needs the data and loads it when needed.

    This way, a cache and several other contextual services can be used to store the data IDs and the data.

    The :class:`~neuraxle.metaopt.AutoML` class is an example of a step that uses this execution context extensively.

    The execution context is used for fitted step saving:
        * :func:`~neuraxle.base._HasSavers.save`
        * :func:`~neuraxle.base._HasSavers.load`

    .. seealso::
        :class:`~neuraxle.base.BaseStep`,
        :class:`~neuraxle.metaopt.auto_ml.AutoML`
    """

    def __init__(
        self,
        root: str = None,
        flow: Flow = None,
        execution_phase: ExecutionPhase = ExecutionPhase.UNSPECIFIED,
        execution_mode: ExecutionMode = ExecutionMode.FIT_OR_FIT_TRANSFORM_OR_TRANSFORM,
        stripped_saver: BaseSaver = None,
        parents: List['BaseStep'] = None,
        services: Dict[ServiceName, 'BaseServiceT'] = None,
    ):
        TruncableService.__init__(self, services=services)
        self.register_service(Flow, flow or Flow())
        self.execution_phase: ExecutionPhase = execution_phase
        self.execution_mode: ExecutionMode = execution_mode
        self.stripped_saver: BaseSaver = stripped_saver or JoblibStepSaver()
        self.parents: List[BaseStep] = parents or []
        self.root: str = root or self.get_new_cache_folder()

    @property
    def logger(self) -> NeuraxleLogger:
        return NeuraxleLogger.from_identifier(self.get_identifier(include_step_names=True))

    @property
    def flow(self) -> Flow:
        """
        Flow is a service that is used to log information about the execution of the pipeline.
        """
        f: Flow = self.get_service(Flow)
        return f.link_context(self)

    @staticmethod
    def get_new_cache_folder() -> str:
        return os.path.join(tempfile.gettempdir(), 'neuraxle-cache', datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss_%fμs"))

    def set_execution_phase(self, phase: ExecutionPhase) -> 'CX':
        """
        Set the instance's execution phase to given phase from the enum :class:`~neuraxle.base.ExecutionPhase`.

        :param phase: execution phase
        :return: self
        """
        self.execution_phase: ExecutionPhase = phase
        return self

    def set_service_locator(self, services: Dict[ServiceName, 'BaseServiceT']) -> 'CX':
        """
        Register abstract class type instances that inherit and implement
        the class :class:`~neuraxle.base.BaseService`.

        :param services: A dictionary of concrete services to register.
        :return: self
        """
        if Flow not in services:
            services[Flow] = self.flow
        self.set_services(services)
        return self

    def get_execution_mode(self) -> ExecutionMode:
        """
        Get the instance's execution mode from the enum :class:`~neuraxle.base.ExecutionMode`.
        """
        return self.execution_mode

    def save(self, full_dump=True):
        """
        Save all unsaved steps in the parents of the execution context using :func:`~neuraxle.base._HasSavers.save`.
        This method is called from a step checkpointer inside a :class:`Checkpoint`.

        :param full_dump: save full pipeline dump to be able to load everything without source code (false by default).
        :return:

        .. seealso::
            :class:`BaseStep`,
            :func:`~neuraxle.base._HasSavers.save`
        """
        # Documentation: https://www.neuraxle.org/stable/step_saving_and_lifecycle.html
        while not self.empty():
            should_save_last_step = self.should_save_last_step()
            last_step = self.peek()
            if full_dump:
                should_save_last_step = True

            self.pop()
            if should_save_last_step:
                last_step.save(self, full_dump)

    def save_last(self):
        """
        Save only the last step in the execution context.

        .. seealso::
            :func:`~neuraxle.base.ExecutionContext.save`
        """
        last_step = self.peek()
        self.pop()
        last_step.save(self, True)

    def should_save_last_step(self) -> bool:
        """
        Returns True if the last step should be saved.

        :return: if the last step should be saved
        """
        if len(self.parents) > 0:
            return self.parents[-1].should_save()
        return False

    def pop_item(self) -> 'BaseTransformer':
        """
        Change the execution context to be the same as the latest parent context.

        :return:
        """
        return self.parents.pop()

    def pop(self) -> bool:
        """
        Pop the context. Returns True if it successfully popped an item from the parents list.

        :return: if an item has been popped
        """
        if len(self) == 0:
            return False
        self.pop_item()
        return True

    def push(self, step: 'BaseTransformer') -> 'CX':
        """
        Pushes a step in the parents of the execution context.

        :param step: step to add to the execution context
        :return: self
        """
        return self.__class__(
            root=self.root,
            flow=self.flow,
            execution_mode=self.execution_mode,
            execution_phase=self.execution_phase,
            parents=self.parents + [step],
            services=self.services,
        )

    def _copy(self, copy_func: str = '_copy'):
        """
        Copy the execution context, and call a copy function on its services as well.

        :param copy_func: services' copy function. By default is `_copy`. Could also be: `copy`, `_copy_trial`, `_copy_trial_split`, `_copy_train`, `_copy_validation`, and more as needed.
        :return: a copy of the execution context (self) using the given copy function on the services.
        """
        copy_kwargs = self._get_copy_kwargs(copy_func)
        return self.__class__(**copy_kwargs)

    def _get_copy_kwargs(self, copy_func: str):
        possibly_copied_services = {
            k: (
                getattr(v, copy_func)() if hasattr(v, copy_func) else
                v._copy() if hasattr(v, '_copy') else
                v.copy() if hasattr(v, 'copy') else
                v
            )
            for k, v in self.services.items()
        }
        copy_kwargs = {
            'root': self.root,
            'flow': self.flow._copy(),
            'execution_mode': self.execution_mode,
            'execution_phase': self.execution_phase,
            'parents': copy(self.parents),
            'services': possibly_copied_services,
        }

        return copy_kwargs

    def synchroneous(self) -> 'CX':
        if self.has_service('HyperparamsRepository'):
            repo = self.get_service('HyperparamsRepository')
            repo = repo.with_lock()
            self.register_service('HyperparamsRepository', repo)
        return self

    def thread_safe(self) -> 'CX':
        """
        Prepare the context and its services to be thread safe

        :return: a tuple of the recursive dict to apply within thread, and the thread safe context
        """
        threaded_context = self._copy()
        self.flow.unlink_context()
        threaded_context.flow.unlink_context()

        # Not passing parents to threads. Could be refactored.
        threaded_context.parents = []
        # Compensate lost parents at the root:
        threaded_context.root = self.get_path()

        _cx2 = threaded_context._copy()
        try:
            for parent in reversed(_cx2.parents):
                pickle.dumps(parent.__reduce__())
        except Exception as e:
            raise pickle.PickleError(
                f"Couldn't pickle {parent} to send it to the next consumers, but its services are picklable. "
                f"From error: {e}"
            ) from e

        threaded_context = threaded_context.synchroneous()

        return threaded_context

    def process_safe(self) -> 'CX':
        """
        Prepare the context and its services to be process safe and
        reduce the current context for parallelization.

        It also does some pickling checks on the services for them to avoid deadlocking
        the multithreading queues by having picklables (parallelizeable) services.

        :return: a tuple of the recursive dict to apply within thread, and the thread safe context
        """
        # TODO: eventually this and the other method above will be an apply that returns a recursive dict of managed locks/things?

        process_safe_context = self._copy()
        self.flow.unlink_context()
        process_safe_context.flow.unlink_context()

        # TODO: code a way to serialize and deserialize such context services. Maybe use some new apply functions in context.process_safe() to pack and then un-pack within thread.
        process_safe_context.flow.unlink_context()
        _cx2 = process_safe_context._copy()
        try:
            for service in _cx2.get_services().values():
                # TODO: mixin that cleans things instead of ifs here.
                if 'Flow' in str(service):
                    continue
                if 'HyperparamsRepository' in str(service):
                    assert 'SynchronizedHyperparamsRepositoryWrapper' in str(
                        service), "Repo must have a lock before being parallelized."
                    service = service.wrapped
                pickle.dumps(service.__reduce__())
        except Exception as e:
            raise pickle.PickleError(f"Couldn't pickle service {service} to send context to other thread.") from e
        try:
            _cx2.set_services(dict())
            pickle.dumps(_cx2.__reduce__())
        except Exception as e:
            raise pickle.PickleError(
                f"Couldn't pickle {process_safe_context} to send it to the next consumers, but its services are picklable. "
                f"From error: {e}"
            ) from e

        return process_safe_context

    def peek(self) -> 'BaseTransformer':
        """
        Get last parent.

        :return: the last parent base step
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

    def get_path(self, is_absolute: bool = True):
        """
        Creates the directory path for the current execution context.

        The returned context path is the concatenation of:
        - the root path,
        - the AutoML trial subpath info,
        - the parent steps subpath info.

        :param is_absolute: bool to say if we want to add root to the path or not
        :return: current context path
        """
        parents_with_path = [self.root] if is_absolute else []
        parents_with_path += self.get_names()
        if len(parents_with_path) == 0:
            return '.' + os.sep
        return os.path.join(*parents_with_path)

    def get_identifier(self, include_step_names: bool = True) -> str:
        """
        Get an identifier depending on the ScopedLocation of the current context.

        Useful for logging. Example:

        .. code-block:: python
                context = ExecutionContext(tmpdir)
                logger = logging.getLogger(context.get_identifier())

        Example: "neuraxle.default_project.default_client.0" + ".".join(self.get_names())

        .. seealso::
            :class:`~neuraxle.metaopt.data.vanilla.ScopedLocation`
        """
        loc_attrs = self.get_service("ScopedLocation").as_list(
            stringify=True) if self.has_service("ScopedLocation") else []
        arr = [NEURAXLE_LOGGER_NAME] + loc_attrs
        if include_step_names:
            arr.extend(self.get_names())
        return ".".join(arr)

    def get_names(self) -> List[str]:
        """
        Returns a list of the parent names.

        :return: list of parents step names
        """
        return list(map(attrgetter('name'), self.parents))

    def empty(self):
        """
        Return True if the context doesn't have parent steps.

        :return: if parents len is 0
        """
        return len(self) == 0

    def load(self, path: str) -> 'BaseTransformer':
        """
        Load full dump at the given path.

        :param path: pipeline step path
        :return: loaded step

        .. seealso::
            :class:`FullDumpLoader`,
            :class:`Identity`
        """
        context_for_loading = self
        context_for_loading = context_for_loading.push(Identity(name=path))

        if os.sep in path:
            context_for_loading = context_for_loading.to_identity()
            path = path.split(os.sep)[-1]

        return FullDumpLoader(
            name=path,
            stripped_saver=self.stripped_saver
        ).load(context_for_loading, True)

    def to_identity(self) -> 'CX':
        """
        Create a fake execution context containing only identity steps.
        Create the parents by using the path of the current execution context.

        :return: fake identity execution context

        .. seealso::
            :class:`FullDumpLoader`,
            :class:`Identity`
        """
        step_names = self.get_path(False).split(os.sep)
        parents = list(map(Identity, step_names))

        return CX(
            root=self.root,
            execution_mode=self.execution_mode,
            stripped_saver=self.stripped_saver,
            parents=parents
        )

    def flush_cache_root(self):
        shutil.rmtree(self.root)

    def flush_cache_local(self):
        shutil.rmtree(self.get_path())

    def __len__(self):
        # TODO: on services instead maybe?
        return len(self.parents)

    def _repr(self, level=0, verbose=False) -> str:
        output = self.__class__.__name__
        output += f'<{self.get_identifier()}>('
        output += self._repr_children(level, verbose)
        level += 1
        if len(self.parents) > 0 and verbose is True:
            output += f",\n{'    ' * level}parents[0]={self.parents[0]._repr(level, verbose)}"
        output += ')'
        return output


CX = ExecutionContext


class _HasSetupTeardownLifecycle(MixinForBaseService):
    """
    Step that has a setup and a teardown lifecycle methods.

    .. note:: All heavy initialization logic should be done inside the *setup* method (e.g.: things inside GPU), and NOT in the constructor of your steps.

    """

    def __init__(self):
        self.is_initialized = False

    def _copy(self, context: CX = None, deep=True) -> '_HasSavers':
        """
        Copy the service or step.

        :param deep: if True, copy the savers as well
        :return: a copy of the step
        """
        self._assert(
            not self.is_initialized,
            "Can't copy an initialized step. "
            "There could be a way to copy for initialized steps using savers, "
            "but that is not implemented.",
            context
        )
        if deep:
            return deepcopy(self)
        return copy(self)

    def setup(self, context: 'CX') -> 'BaseTransformer':
        """
        Initialize the step before it runs. Only from here and not before that heavy things should be created
        (e.g.: things inside GPU), and NOT in the constructor.

        The _setup method is executed only if is self.is_initialized is False
        A setup function should set the self.is_initialized to True when called.

        .. warning::
            This setup method sets up the whole hierarchy of nested steps with children.
            If you want to setup progressively, use only self._setup() instead.
            The _setup method is called once for each step when
            handle_fit, handle_fit_transform or handle_transform is called.

        :param context: execution context
        :return: self
        """
        self.apply("_setup", context=context)
        return self

    def _setup(self, context: 'CX' = None) -> Optional[RecursiveDict]:
        """
        Internal method to setup the step. May be used by :class:`~neuraxle.pipeline.Pipeline`
        to setup the pipeline progressively instead of all at once.
        """
        self.is_initialized = True
        return RecursiveDict()

    def teardown(self) -> 'BaseTransformer':
        """
        Applies _teardown on the step and, if applicable, its children.
        :return: self
        """
        self.apply("_teardown")
        return self

    def _teardown(self) -> Optional[RecursiveDict]:
        """
        Teardown step after program execution. Inverse of setup, and it should clear memory.
        Override this method if you need to clear memory.

        :return: self
        """
        self.is_initialized = False
        return RecursiveDict()

    def __del__(self):
        try:
            self._teardown()
        except Exception:
            print(traceback.format_exc())


class _TransformerStep(MixinForBaseService):
    """
    An internal class to represent a step that can be transformed, or inverse transformed.
    See :class:`BaseTransformer`, for the complete transformer step that can be used inside a :class:`neuraxle.pipeline.Pipeline`.
    See :class:`BaseStep`, for a step that can also be fitted inside a :class:`neuraxle.pipeline.Pipeline`.

    Every step must implement :func:`~neuraxle.base._TransformerStep.transform`.
    If a step is not transformable, you can inherit from :class:`NonTransformableMixin`.

    Every transformer step has handle methods that can be overridden to add side effects or change the execution flow based on the execution context, and the data container :
        * :func:`~neuraxle.base._TransformerStep.handle_transform`
        * :func:`~neuraxle.base._TransformerStep.handle_fit_transform`
        * :func:`~neuraxle.base._TransformerStep.handle_fit`

    .. seealso::
        :class:`BaseStep`,
        :class:`BaseTransformer`,
        :class:`_FittableStep`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def _will_process(
        self, data_container: TrainDACT, context: CX
    ) -> Tuple[TrainDACT, CX]:
        """
        Apply side effects before any step method.
        :param data_container: data container
        :param context: execution context
        :return: (data container, execution context)
        """
        return data_container, context

    def handle_transform(self, data_container: TrainDACT, context: CX) -> PredsDACT:
        """
        Override this to add side effects or change the execution flow before (or after) calling * :func:`~neuraxle.base._TransformerStep.transform`.

        :param data_container: the data container to transform
        :param context: execution context
        :return: transformed data container
        """
        if not self.is_initialized:
            self._setup(context)

        data_container, context = self._will_process(data_container, context)
        data_container, context = self._will_transform(data_container, context)
        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        # Documentation: https://www.neuraxle.org/stable/handler_methods.html
        data_container = self._transform_data_container(data_container, context)
        # ////////////////////////////////////////////////////////////////////
        data_container = self._did_transform(data_container, context)
        data_container = self._did_process(data_container, context)

        return data_container

    def _will_transform(
        self, data_container: TrainDACT, context: CX
    ) -> Tuple[TrainDACT, CX]:
        """
        Apply side effects before transform.

        :param data_container: data container
        :param context: execution context
        :return: (data container, execution context)
        """
        return data_container, context.push(self)

    def _will_transform_data_container(
        self, data_container: TrainDACT, context: CX
    ) -> Tuple[TrainDACT, CX]:
        """
        This method is deprecated and will redirect to `_will_transform`. Use `_will_transform` instead.

        :param data_container: data container
        :param context: execution context
        :return: (data container, execution context)
        """
        return self._will_transform(data_container, context)

    def _transform_data_container(self, data_container: TrainDACT, context: CX) -> PredsDACT:
        """
        Transform data container.

        :param data_container: data container
        :param context: execution context
        :return: data container
        """
        data_container.di = self(data_container.di)
        return data_container

    def __call__(self, *args, **kwargs) -> Any:
        return self.transform(*args, **kwargs)

    @abstractmethod
    def transform(self, data_inputs: ARG_X_INPUTTED) -> ARG_Y_PREDICTD:
        """
        Transform given data inputs.

        :param data_inputs: data inputs
        :return: transformed data inputs
        """
        raise NotImplementedError(
            f"Implement this method in {self.__class__.__name__}, or have this class inherit from "
            f"the NonFittableMixin. You should otherwise ideally use handler methods. "
            f"Read more: https://www.neuraxle.org/stable/handler_methods.html")

    def _did_transform(self, data_container: PredsDACT, context: CX) -> PredsDACT:
        """
        Apply side effects after transform.

        :param data_container: data container
        :param context: execution context
        :return: data container
        """
        return data_container

    def _did_process(self, data_container: PredsDACT, context: CX) -> PredsDACT:
        """
        Apply side effects after any step method.

        :param data_container: data container
        :param context: execution context
        :return: (data container, execution context)
        """
        return data_container

    def handle_fit(self, data_container: TrainDACT, context: CX) -> 'BaseTransformer':
        """
        Override this to add side effects or change the execution flow before (or after) calling :func:`~neuraxle.base._FittableStep.fit`.

        :param data_container: the data container to transform
        :param context: execution context
        :return: tuple(fitted pipeline, data_container)

        .. seealso::
            :class:`~neuraxle.data_container.DataContainer`,
            :class:`~neuraxle.pipeline.Pipeline`
        """
        if not self.is_initialized:
            self._setup(context)

        self._did_process(data_container, context)
        return self

    def handle_fit_transform(
        self, data_container: TrainDACT, context: CX
    ) -> Tuple['BaseTransformer', PredsDACT]:
        """
        Override this to add side effects or change the execution flow before
        (or after) calling * :func:`~neuraxle.base._FittableStep.fit_transform`.

        :param data_container: the data container to transform
        :param context: execution context
        :return: tuple(fitted pipeline, data_container)
        """
        return self, self.handle_transform(data_container, context)

    def fit(self, data_inputs: ARG_X_INPUTTED, expected_outputs: ARG_Y_EXPECTED) -> '_TransformerStep':
        """
        Fit given data inputs. By default, a step only transforms in the fit transform method.
        To add fitting to your step, see class:`_FittableStep` for more info.
        :param data_inputs: data inputs
        :param expected_outputs: expected outputs to fit on
        :return: transformed data inputs
        """
        return self

    def fit_transform(
        self, data_inputs: ARG_X_INPUTTED, expected_outputs: ARG_Y_EXPECTED = None
    ) -> Tuple['_TransformerStep', ARG_Y_PREDICTD]:
        """
        Fit transform given data inputs. By default, a step only transforms in the fit transform method.
        To add fitting to your step, see class:`_FittableStep` for more info.

        :param data_inputs: data inputs
        :param expected_outputs: expected outputs to fit on
        :return: transformed data inputs
        """
        self = self.fit(data_inputs, expected_outputs)
        return self, self.transform(data_inputs)

    def handle_predict(self, data_container: TrainDACT, context: CX) -> PredsDACT:
        """
        Handle_transform in test mode.

        :param data_container: the data container to transform
        :param context: execution context
        :return: transformed data container
        """
        was_train: bool = self.is_train
        self.set_train(False)

        data_container = self.handle_transform(data_container, context)

        self.set_train(was_train)
        return data_container

    def predict(self, data_input: ARG_X_INPUTTED) -> ARG_Y_PREDICTD:
        """
        Predict the expected output in test mode using func:`~neuraxle.base._TransformerStep.transform`, but by setting self to test mode first and then reverting the mode.

        :param data_input: data input to predict
        :return: prediction
        """
        was_train: bool = self.is_train
        self.set_train(False)

        outputs = self(data_input)

        self.set_train(was_train)
        return outputs

    def handle_inverse_transform(self, data_container: PredsDACT, context: CX) -> TrainDACT:
        """
        Override this to add side effects or change the execution flow before (or after) calling :func:`~neuraxle.base._TransformerStep.inverse_transform`.

        :param data_container: the data container to inverse transform
        :param context: execution context
        :return: data_container

        .. seealso::
            :class:`~neuraxle.data_container.DataContainer`,
            :class:`~neuraxle.pipeline.Pipeline`
        """
        if not self.is_initialized:
            self._setup(context)

        data_container, context = self._will_process(data_container, context)
        data_container = self._inverse_transform_data_container(data_container, context)
        data_container = self._did_process(data_container, context)

        return data_container

    def _inverse_transform_data_container(self, data_container: PredsDACT, context: CX) -> TrainDACT:
        processed_outputs = self.inverse_transform(data_container.data_inputs)
        data_container.set_data_inputs(processed_outputs)

        return data_container

    def inverse_transform(self, processed_outputs: ARG_Y_PREDICTD) -> ARG_X_INPUTTED:
        """
        Inverse Transform the given transformed data inputs.

        .. code-block:: python

            p = Pipeline([MultiplyByN(2)])
            _in = np.array([1, 2])
            _out = p.transform(_in)
            _regenerated_in = p.inverse_transform(_out)
            assert np.array_equal(_regenerated_in, _in)
            assert np.array_equal(_out, _in * 2)


        :param processed_outputs: processed data inputs
        :return: inverse transformed processed outputs
        """
        raise NotImplementedError("Implement this method in {}.".format(self.__class__.__name__))


class _FittableStep(MixinForBaseService):
    """
    An internal class to represent a step that can be fitted.
    See :class:`BaseStep`, for a complete step that can be transformed, and fitted inside a :class:`neuraxle.pipeline.Pipeline`.

    .. seealso::
        :class:`BaseStep`,
        :class:`BaseTransformer`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def handle_fit(self, data_container: TrainDACT, context: CX) -> 'BaseStep':
        """
        Override this to add side effects or change the execution flow before (or after) calling :func:`~neuraxle.base._FittableStep.fit`.

        :param data_container: the data container to transform
        :param context: execution context
        :return: tuple(fitted pipeline, data_container)
        """
        if not self.is_initialized:
            self._setup(context)

        data_container, context = self._will_process(data_container, context)
        data_container, context = self._will_fit(data_container, context)
        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        # Documentation: https://www.neuraxle.org/stable/handler_methods.html
        new_self = self._fit_data_container(data_container, context)
        # ////////////////////////////////////////////////////////////////////
        self._did_fit(data_container, context)
        self._did_process(data_container, context)

        return new_self

    def _will_fit(
        self, data_container: TrainDACT, context: CX
    ) -> Tuple[TrainDACT, CX]:
        """
        Before fit is called, apply side effects on the step, the data container, or the execution context.

        :param data_container: data container
        :param context: execution context
        :return: (data container, execution context)
        """
        self._invalidate()
        return data_container, context.push(self)

    def _fit_data_container(self, data_container: TrainDACT, context: CX) -> '_FittableStep':
        """
        Fit data container.

        :param data_container: data container
        :param context: execution context
        :return: (fitted self, data container)
        """
        return self.fit(data_container.data_inputs, data_container.expected_outputs)

    def _did_fit(self, data_container: TrainDACT, context: CX) -> TrainDACT:
        """
        Apply side effects before fit is called.

        :param data_container: data container
        :param context: execution context
        :return: (data container, execution context)
        """
        return data_container

    @abstractmethod
    def fit(self, data_inputs: ARG_X_INPUTTED, expected_outputs: ARG_Y_EXPECTED) -> '_FittableStep':
        """
        Fit data inputs on the given expected outputs.

        :param data_inputs: data inputs
        :param expected_outputs: expected outputs to fit on.
        :return: self
        """
        raise NotImplementedError(
            f"Implement this method in {self.__class__.__name__}, or have this class inherit from "
            f"the NonFittableMixin. You should otherwise ideally use handler methods. "
            f"Read more: https://www.neuraxle.org/stable/handler_methods.html")

    def handle_fit_transform(
        self, data_container: TrainDACT, context: CX
    ) -> Tuple['BaseStep', PredsDACT]:
        """
        Override this to add side effects or change the execution flow before (or after) calling * :func:`~neuraxle.base._FittableStep.fit_transform`.

        :param data_container: the data container to transform
        :param context: execution context
        :return: tuple(fitted pipeline, data_container)
        """
        if not self.is_initialized:
            self._setup(context)

        data_container, context = self._will_process(data_container, context)
        data_container, context = self._will_fit_transform(data_container, context)
        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        # Documentation: https://www.neuraxle.org/stable/handler_methods.html
        new_self, data_container = self._fit_transform_data_container(data_container, context)
        # ////////////////////////////////////////////////////////////////////
        data_container = self._did_fit_transform(data_container, context)
        data_container = self._did_process(data_container, context)

        return new_self, data_container

    def _will_fit_transform(
        self, data_container: TrainDACT, context: CX
    ) -> Tuple[TrainDACT, CX]:
        """
        Apply side effects before fit_transform is called.

        :param data_container: data container
        :param context: execution context
        :return: (data container, execution context)
        """
        self._invalidate()
        return data_container, context.push(self)

    def _fit_transform_data_container(
        self, data_container: TrainDACT, context: CX
    ) -> Tuple['BaseStep', PredsDACT]:
        """
        Fit transform data container.

        :param data_container: data container
        :param context: execution context
        :return: (fitted self, data container)
        """
        new_self, out = self.fit_transform(data_container.data_inputs, data_container.expected_outputs)
        data_container.set_data_inputs(out)

        return new_self, data_container

    def fit_transform(
        self, data_inputs, expected_outputs=None
    ) -> Tuple['BaseStep', Any]:
        """
        Fit, and transform step with the given data inputs, and expected outputs.

        :param data_inputs: data inputs
        :param expected_outputs: expected outputs to fit on
        :return: (fitted self, tranformed data inputs)
        """
        self._invalidate()

        new_self = self.fit(data_inputs, expected_outputs)
        out = new_self(data_inputs)

        return new_self, out

    def _did_fit_transform(self, data_container: PredsDACT, context: CX) -> PredsDACT:
        """
        Apply side effects after fit transform.

        :param data_container: data container
        :param context: execution context
        :return: (fitted self, data container)
        """
        return data_container


class _CustomHandlerMethods(MixinForBaseService):
    """
    A class to represent a step that needs to add special behavior on top of the normal handler methods.
    It allows the step to apply side effects before calling the real handler method.

    Apply additional behavior (mini-batching, parallel processing, etc.) before calling the internal handler methods :
        - :func:`~neuraxle.base._FittableStep._fit_data_container`
        - :func:`~neuraxle.base._FittableStep._fit_transform_data_container`
        - :func:`~neuraxle.base._TransformerStep._transform_data_container`

    .. seealso::
        :class:`~neuraxle.base._FittableStep`,
        :class:`~neuraxle.base._TransformerStep`,
        :class:`~neuraxle.pipeline.MiniBatchSequentialPipeline`,
        :class:`~neuraxle.distributed.streaming.BaseQueuedPipeline`
    """

    def __init__(self):
        MixinForBaseService.__init__(self)

    def handle_fit(self, data_container: TrainDACT, context: CX) -> 'BaseStep':
        """
        Handle fit with a custom handler method for fitting the data container.
        The custom method to override is fit_data_container.
        The custom method fit_data_container replaces _fit_data_container.

        :param data_container: the data container to transform
        :param context: execution context
        :return: tuple(fitted pipeline, data_container)

        .. seealso::
            :class:`~neuraxle.base._FittableStep̀,
            :class:`~neuraxle.data_container.DataContainer`,
            :class:`~neuraxle.base.ExecutionContext`
        """
        if not self.is_initialized:
            self._setup(context)

        data_container, context = self._will_process(data_container, context)
        data_container, context = self._will_fit(data_container, context)
        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        # Documentation: https://www.neuraxle.org/stable/handler_methods.html
        new_self = self.fit_data_container(data_container, context)
        # ////////////////////////////////////////////////////////////////////
        self._did_fit(data_container, context)
        self._did_process(data_container, context)

        return new_self

    def handle_fit_transform(
        self, data_container: TrainDACT, context: CX
    ) -> Tuple['BaseStep', PredsDACT]:
        """
        Handle fit_transform with a custom handler method for fitting, and transforming the data container.
        The custom method to override is fit_transform_data_container.
        The custom method fit_transform_data_container replaces :func:`~neuraxle.base._FittableStep._fit_transform_data_container`.

        :param data_container: the data container to transform
        :param context: execution context
        :return: tuple(fitted pipeline, data_container)

        .. seealso::
            :class:`~neuraxle.base._FittableStep̀,
            :class:`~neuraxle.data_container.DataContainer`,
            :class:`~neuraxle.base.ExecutionContext`
        """
        if not self.is_initialized:
            self._setup(context)

        data_container, context = self._will_process(data_container, context)
        data_container, context = self._will_fit_transform(data_container, context)
        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        # Documentation: https://www.neuraxle.org/stable/handler_methods.html
        new_self, data_container = self.fit_transform_data_container(data_container, context)
        # ////////////////////////////////////////////////////////////////////
        data_container = self._did_fit_transform(data_container, context)
        data_container = self._did_process(data_container, context)

        return new_self, data_container

    def handle_transform(self, data_container: TrainDACT, context: CX) -> PredsDACT:
        """
        Handle transform with a custom handler method for transforming the data container.
        The custom method to override is transform_data_container.
        The custom method transform_data_container replaces :func:`~neuraxle.base._TransformerStep._transform_data_container`.

        :param data_container: the data container to transform
        :param context: execution context
        :return: transformed data container

        .. seealso::
            :class:`~neuraxle.base._TransformerStep̀,
            :class:`~neuraxle.data_container.DataContainer`,
            :class:`~neuraxle.base.ExecutionContext`
        """
        if not self.is_initialized:
            self._setup(context)

        data_container, context = self._will_process(data_container, context)
        data_container, context = self._will_transform(data_container, context)
        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        # Documentation: https://www.neuraxle.org/stable/handler_methods.html
        data_container = self.transform_data_container(data_container, context)
        # ////////////////////////////////////////////////////////////////////
        data_container = self._did_transform(data_container, context)
        data_container = self._did_process(data_container, context)

        return data_container

    @abstractmethod
    def fit_data_container(self, data_container: TrainDACT, context: CX):
        """
        Custom fit data container method.

        :param data_container: data container to fit on
        :param context: execution context
        :return: fitted self
        """
        raise NotImplementedError()

    @abstractmethod
    def fit_transform_data_container(self, data_container: TrainDACT, context: CX):
        """
        Custom fit transform data container method.

        :param data_container: data container to fit on
        :param context: execution context
        :return: fitted self, transformed data container
        """
        raise NotImplementedError()

    @abstractmethod
    def transform_data_container(self, data_container: TrainDACT, context: CX):
        """
        Custom transform data container method.

        :param data_container: data container to transform
        :param context: execution context
        :return: transformed data container
        """
        raise NotImplementedError()


class _HasHyperparamsSpace(MixinForBaseService):
    """
    An internal class to represent a step that has hyperparameter spaces of type :class:`~neuraxle.hyperparams.space.HyperparameterSpace`.
    See :class:`BaseStep`, for a complete step that can be transformed, and fitted inside a :class:`neuraxle.pipeline.Pipeline`.

    .. seealso::
        :class:`BaseStep`,
        :class:`BaseTransformer`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(self, hyperparams_space: HyperparameterSpace = None):
        if hyperparams_space is None:
            hyperparams_space = dict()

        self.hyperparams_space: HyperparameterSpace = HyperparameterSpace(hyperparams_space)

    def set_hyperparams_space(self, hyperparams_space: Union[Dict, HyperparameterSpace]) -> 'BaseTransformer':
        """
        Set step hyperparameters space.

        Example :

        .. code-block:: python

            step.set_hyperparams_space(HyperparameterSpace({
                'hp': RandInt(0, 10)
            }))


        :param hyperparams_space: hyperparameters space
        :return: self

        .. note::
            This is a recursive method that will call :func:`BaseStep._set_hyperparams_space` in the end.
        .. seealso::
            :class:`~neuraxle.hyperparams.space.HyperparameterSamples`,
            :class:`~neuraxle.hyperparams.space.HyperparameterSpace`,
            :class:`~neuraxle.hyperparams.distributions.HyperparameterDistribution`
            :class:̀_HasChildrenMixin`,
            :func:`BaseStep.apply`,
            :func:`_HasChildrenMixin._apply`,
            :func:`_HasChildrenMixin._get_params`
        """
        if not isinstance(hyperparams_space, HyperparameterSpace):
            hyperparams_space = HyperparameterSpace(hyperparams_space)
        self.apply(method='_set_hyperparams_space', hyperparams_space=hyperparams_space)
        return self

    def _set_hyperparams_space(self, hyperparams_space: HyperparameterSpace) -> HyperparameterSpace:
        self._invalidate()
        self.hyperparams_space = HyperparameterSpace(hyperparams_space)
        return self.hyperparams_space

    def update_hyperparams_space(self, hyperparams_space: Union[Dict, HyperparameterSpace]) -> 'BaseTransformer':
        """
        Update the step hyperparameter spaces without removing the already-set hyperparameters.
        This can be useful to add more hyperparameter spaces to the existing ones without flushing the ones that were already set.

        Example :

        .. code-block:: python

            step.set_hyperparams_space(HyperparameterSpace({
                'learning_rate': LogNormal(0.5, 0.5)
                'weight_decay': LogNormal(0.001, 0.0005)
            }))

            step.update_hyperparams_space(HyperparameterSpace({
                'learning_rate': LogNormal(0.5, 0.1)
            }))

            assert step.get_hyperparams_space()['learning_rate'] == LogNormal(0.5, 0.1)
            assert step.get_hyperparams_space()['weight_decay'] == LogNormal(0.001, 0.0005)


        :param hyperparams_space: hyperparameters space
        :return: self

        .. note::
            This is a recursive method that will call :func:`BaseStep._update_hyperparams_space` in the end.
        .. seealso::
            :func:`~BaseStep.update_hyperparams`,
            :class:`~neuraxle.hyperparams.space.HyperparameterSpace`
        """
        if not isinstance(hyperparams_space, HyperparameterSpace):
            hyperparams_space = HyperparameterSpace(hyperparams_space)
        self.apply(method='_update_hyperparams_space', hyperparams_space=hyperparams_space)
        return self

    def _update_hyperparams_space(self, hyperparams_space: HyperparameterSpace) -> HyperparameterSpace:
        self._invalidate()
        self.hyperparams_space.update(hyperparams_space)
        return self.hyperparams_space

    def get_hyperparams_space(self) -> HyperparameterSpace:
        """
        Get step hyperparameters space.

        Example :

        .. code-block:: python

            step.get_hyperparams_space()

        :return: step hyperparams space

        .. note::
            This is a recursive method that will call :func:`BaseStep._get_hyperparams_space` in the end.
        .. seealso::
            :class:`~neuraxle.hyperparams.space.HyperparameterSpace`,
            :class:`~neuraxle.hyperparams.distributions.HyperparameterDistribution`
        """
        results: HyperparameterSpace = self.apply(method='_get_hyperparams_space')
        return results

    def _get_hyperparams_space(self) -> HyperparameterSpace:
        return self.hyperparams_space


class _HasHyperparams(MixinForBaseService):
    """
    An internal class to represent a step that has hyperparameters of type :class:`~neuraxle.hyperparams.space.HyperparameterSamples`.
    See :class:`BaseStep`, for a complete step that can be transformed, and fitted inside a :class:`neuraxle.pipeline.Pipeline`.

    Every step has hyperparemeters, and hyperparameters spaces that can be set before the learning process begins.
    Hyperparameters can not only be passed in the constructor, but also be set by the pipeline that contains all of the steps :


    .. code-block:: python

        pipeline = Pipeline([
            SomeStep()
        ])

        pipeline.set_hyperparams(HyperparameterSamples({
            'learning_rate': 0.1,
            'SomeStep__learning_rate': 0.05
        }))


    .. seealso::
        :class:`BaseStep`,
        :class:`BaseTransformer`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(self, hyperparams: HyperparameterSamples = None):
        if hyperparams is None:
            hyperparams = dict()

        self.hyperparams: HyperparameterSamples = HyperparameterSamples(hyperparams)

    def set_hyperparams(self, hyperparams: Union[Dict, HyperparameterSamples]) -> 'BaseTransformer':
        """
        Set the step hyperparameters.

        Example :

        .. code-block:: python

            step.set_hyperparams(HyperparameterSamples({
                'learning_rate': 0.10
            }))


        :param hyperparams: hyperparameters
        :return: self

        .. note::
            This is a recursive method that will call :func:`~neuraxle.base._HasHyperparams._set_hyperparams`.
        .. seealso::
            :class:`~neuraxle.hyperparams.space.HyperparameterSamples`,
            :class:̀_HasChildrenMixin`,
            :func:`BaseStep.apply`,
            :func:`_HasChildrenMixin._apply`,
            :func:`_HasChildrenMixin._set_train`
        """
        if not isinstance(hyperparams, HyperparameterSamples):
            hyperparams = HyperparameterSamples(hyperparams)
        self.apply(method='_set_hyperparams', hyperparams=hyperparams)
        return self

    def _set_hyperparams(self, hyperparams: HyperparameterSamples) -> HyperparameterSamples:
        self._invalidate()
        hyperparams = HyperparameterSamples(hyperparams)
        self.hyperparams = hyperparams if len(hyperparams) > 0 else self.hyperparams
        return hyperparams

    def update_hyperparams(self, hyperparams: Union[Dict, HyperparameterSamples]) -> 'BaseTransformer':
        """
        Update the step hyperparameters without removing the already-set hyperparameters.
        This can be useful to add more hyperparameters to the existing ones without flushing the ones that were already set.

        Example :

        .. code-block:: python

            step.set_hyperparams(HyperparameterSamples({
                'learning_rate': 0.10
                'weight_decay': 0.001
            }))

            step.update_hyperparams(HyperparameterSamples({
                'learning_rate': 0.01
            }))

            assert step.get_hyperparams()['learning_rate'] == 0.01
            assert step.get_hyperparams()['weight_decay'] == 0.001

        :param hyperparams: hyperparameters
        :return: self

        .. seealso::
            :class:`~neuraxle.hyperparams.space.HyperparameterSamples`,
            :class:̀_HasChildrenMixin`,
            :func:`BaseStep.apply`,
            :func:`_HasChildrenMixin._apply`,
            :func:`_HasChildrenMixin._update_hyperparams`
        """
        if not isinstance(hyperparams, HyperparameterSamples):
            hyperparams = HyperparameterSamples(hyperparams)
        self.apply(method='_update_hyperparams', hyperparams=hyperparams)
        return self

    def _update_hyperparams(self, hyperparams: HyperparameterSamples) -> HyperparameterSamples:
        self.hyperparams.update(hyperparams)
        return self.hyperparams

    def get_hyperparams(self) -> HyperparameterSamples:
        """
        Get step hyperparameters as :class:`~neuraxle.hyperparams.space.HyperparameterSamples`.

        :return: step hyperparameters

        .. note::
            This is a recursive method that will call :func:`~neuraxle.base._HasHyperparams._get_hyperparams`.
        .. seealso::
            :class:`~neuraxle.hyperparams.space.HyperparameterSamples`,
            :class:̀_HasChildrenMixin`,
            :func:`BaseStep.apply`,
            :func:`_HasChildrenMixin._apply`,
            :func:`_HasChildrenMixin._get_hyperparams`
        """
        results: HyperparameterSamples = self.apply(method='_get_hyperparams')
        return results

    def _get_hyperparams(self) -> HyperparameterSamples:
        return self.hyperparams

    def set_params(self, **params: dict) -> 'BaseTransformer':
        """
        Set step hyperparameters with a dictionary.

        Example :

        .. code-block:: python

            s.set_params(learning_rate=0.1)
            hyperparams = s.get_params()
            assert hyperparams == {"learning_rate": 0.1}

        :param arbitrary number of arguments for hyperparameters

        .. note::
            This is a recursive method that will call :func:`~neuraxle.base._HasHyperparams._set_params` in the end.
        .. seealso::
            :class:`~neuraxle.hyperparams.space.HyperparameterSamples`,
            :class:̀_HasChildrenMixin`,
            :func:`~neuraxle.base._HasRecursiveMethods.apply`,
            :func:`~neuraxle.base._HasChildrenMixin._apply`,
            :func:`~neuraxle.base._HasChildrenMixin._set_params`
        """
        self.apply(method='_set_hyperparams', hyperparams=HyperparameterSamples(params))
        return self

    def get_params(self, deep=False) -> dict:
        """
        Get step hyperparameters as a flat primitive dict.
        The "deep" parameter is ignored.

        Example :

        .. code-block:: python

            s.set_params(learning_rate=0.1)
            hyperparams = s.get_params()
            assert hyperparams == {"learning_rate": 0.1}

        :return: hyperparameters

        .. seealso::
            :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
            :class:̀_HasChildrenMixin`,
            :func:`~neuraxle.base._HasRecursiveMethods.apply`,
            :func:`~neuraxle.base._HasChildrenMixin.apply`,
            :func:`~neuraxle.base._HasHyperparams._get_params`
        """
        results: HyperparameterSamples = self.apply(method='_get_hyperparams')
        return results.to_flat_dict()

    def _repr_params(self, level: int = 0, verbose: bool = False) -> str:
        output = ''
        if verbose:
            hps: HyperparameterSamples = self._get_hyperparams()
            if not hps.is_empty():
                # hps = hps.to_flat_dict(use_wildcards=not verbose)
                output += ", hyperparams=" + pprint.pformat(hps)
        return output


class _HasSavers(MixinForBaseService):
    """
    An internal class to represent a step that can be saved.
    A step with savers is saved using its list of savers.
    Each saver saves some parts of the step.

    A pipeline can save the step that need to be saved (see :func:`~neuraxle.base._HasSavers.save`) can be saved :

    .. code-block:: python

        step = Pipeline([
            Identity()
        ])
        step.save()
        step = step.load()


    Or, it can also save a full dump that can be reloaded without any source code :

    .. code-block:: python

        step = Identity().set_name('step_name')
        step.save(full_dump=True)
        step = ExecutionContext().load('step_name')


    .. seealso::
        :class:`BaseStep`,
        :class:`BaseTransformer`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`
    """

    def __init__(self, savers: List[BaseSaver] = None):
        if savers is None:
            savers = []
        self.savers: List[BaseSaver] = savers
        self.is_invalidated = False

    def invalidate(self) -> 'BaseTransformer':
        """
        Invalidate a step, and all of its children. Invalidating a step makes it eligible to be saved again.

        A step is invalidated when any of the following things happen :
            * an hyperparameter has changed func: `~neuraxle.base._HasHyperparams.set_hyperparams`
            * an hyperparameter space has changed func: `~neuraxle.base._HasHyperparamsSpace.set_hyperparams_space`
            * a call to the fit method func:`~neuraxle.base._FittableStep.handle_fit`
            * a call to the fit_transform method func:`~neuraxle.base._FittableStep.handle_fit_transform`
            * the step name has changed func:`~neuraxle.base.BaseStep.set_name`

        :return: self

        .. note::
            This is a recursive method used in :class:̀_HasChildrenMixin`.

        .. seealso::
            :func:`~neuraxle.base._HasRecursiveMethods.apply`,
            :func:`~neuraxle.base._HasChildrenMixin._apply`
        """
        self.apply(method='_invalidate')
        return self

    def _invalidate(self) -> Optional[RecursiveDict]:
        self.is_invalidated = True
        return RecursiveDict()

    def get_savers(self) -> List[BaseSaver]:
        """
        Get the step savers of a pipeline step.

        :return: step savers

        .. seealso::
            :class:`BaseSaver`
        """
        return self.savers

    def set_savers(self, savers: List[BaseSaver]) -> 'BaseTransformer':
        """
        Set the step savers of a pipeline step.

        :return: self

        .. seealso::
            :class:`BaseSaver`
        """
        self.savers: List[BaseSaver] = savers
        return self

    def add_saver(self, saver: BaseSaver) -> 'BaseTransformer':
        """
        Add a step saver of a pipeline step.

        :return: self

        .. seealso::
            :class:`BaseSaver`
        """
        self.savers.append(saver)
        return self

    def should_save(self) -> bool:
        """
        Returns true if the step should be saved.
        If the step has been initialized and invalidated, then it must be saved.

        A step is invalidated when any of the following things happen :

            * a mutation has been performed on the step :func:`~.mutate`
            * an hyperparameter has changed func:`~neuraxle.base._HasHyperparams.set_hyperparams`
            * an hyperparameter space has changed func:`~neuraxle.base._HasHyperparamsSpace.set_hyperparams_space`
            * a call to the fit method func:`~neuraxle.base._FittableStep.handle_fit`
            * a call to the fit_transform method func:`~neuraxle.base._FittableStep.handle_fit_transform`
            * the step name has changed func:`~neuraxle.base.BaseStep.set_name`

        :return: if the step should be saved
        """
        return self.is_invalidated and self.is_initialized

    def save(self, context: CX, full_dump=True) -> 'BaseTransformer':
        """
        Save step using the execution context to create the directory to save the step into.
        The saving happens by looping through all of the step savers in the reversed order.

        Some savers just save parts of objects, some save it all or what remains.
        The :class:`ExecutionContext`.stripped_saver has to be called last because it needs a
        stripped version of the step.

        :param context: context to save from
        :param full_dump: save full pipeline dump to be able to load everything without source code (false by default).
        :return: self

        .. seealso::
            :class:`ExecutionContext`,
            :class:`BaseSaver`
        """
        # Documentation: https://www.neuraxle.org/stable/step_saving_and_lifecycle.html

        context = context.push(self)
        if not full_dump and not self.should_save():
            return self
        self.is_invalidated = False

        if full_dump:
            self.setup(context=context)
            self.invalidate()

        context.mkdir()
        stripped_step = copy(self)

        # A final "visitor" saver will save anything that
        # wasn't saved customly after stripping the rest.
        savers_with_provided_default_stripped_saver = [context.stripped_saver] + self.savers

        for saver in reversed(savers_with_provided_default_stripped_saver):
            # Each saver strips the step a bit more if needs be.
            stripped_step = saver.save_step(stripped_step, context)

        return stripped_step

    def load(self, context: CX, full_dump=True) -> 'BaseTransformer':
        """
        Load step using the execution context to create the directory of the saved step.
        Warning:

        :param context: execution context to load step from
        :param full_dump: save full dump bool
        :return: loaded step

        .. warning::
            Please do not override this method because on loading it is an identity
            step that will load whatever step you coded.

        .. seealso::
            :class:`ExecutionContext`,
            :class:`BaseSaver`
        """
        context = context.push(self)

        savers = [context.stripped_saver] + self.savers
        return self._load_step(context, savers)

    def _load_step(self, context, savers):
        # A final "visitor" saver might reload anything that wasn't saved customly after stripping the rest.
        loaded_self = self
        has_been_loaded_by_a_saver: bool = False
        for saver in savers:
            # Each saver unstrips the step a bit more if needed
            if saver.can_load(loaded_self, context):
                loaded_self = saver.load_step(loaded_self, context)
                has_been_loaded_by_a_saver = True
            else:
                has_been_loaded_by_a_saver and warnings.warn(
                    'Cannot Load Step {0} ({1}:{2}) With Step Saver {3}.'.format(
                        context.get_path(),
                        self.name,
                        self.__class__.__name__,
                        saver.__class__.__name__)
                )
                break
        return loaded_self


class _CouldHaveContext(MixinForBaseService):
    """
    Step that can have a context.
    It has "has service assertions" to ensure that the context has registered all the necessary services.

    A context can be injected with the with_context method:

    .. code-block:: python

        context = ExecutionContext(root=tmpdir)
        service = SomeService()
        context.set_service_locator({SomeBaseService: service})

        p = Pipeline([
            SomeStep().assert_has_services(SomeBaseService)
        ]).with_context(context=context)

    Or alternatively,

    .. code-block:: python

        p = Pipeline([
            RegisterSomeService(),
            SomeStep().assert_has_services_at_execution(SomeBaseService)
        ])

    Context services can be used inside any step with handler methods:

    .. code-block:: python

        class SomeStep(ForceHandleMixin, Identity):
            def __init__(self):
                Identity.__init__(self)
                ForceHandleMixin.__init__(self)

            def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
                service: SomeBaseService = context.get_service(SomeBaseService)
                service.service_method(data_container.data_inputs)
                return data_container


    .. seealso::
        :class:`~neuraxle.base.BaseTransformer`,
        :class:`~neuraxle.base._TransformerStep`,
    """

    def with_context(self, context: CX) -> 'StepWithContext':
        """
        An higher order step to inject a context inside a step.
        A step with a context forces the pipeline to use that context through handler methods.
        This is useful for dependency injection because you can register services inside the :class:`ExecutionContext`.
        It also ensures that the context has registered all the necessary services.

        .. code-block:: python

            context = ExecutionContext(tmpdir)
            context.set_service_locator(ServiceLocator().services)
            # where services is of type Dict[Type['BaseService'], 'BaseService']

            p = WithContext(Pipeline([
                # When the context will be processing the SomeStep,
                # it will be asserted that the context will be able to access the SomeBaseService
                SomeStep().with_assertion_has_services(SomeBaseService)
            ]), context)


        .. seealso::
            :class:`BaseStep`,
            :class:`ExecutionContext`,
            :class:`BaseTransformer`
        """
        return StepWithContext(wrapped=self, context=context)

    def assert_has_services(self, *service_assertions: List[Type['BaseService']]) -> 'GlobalyRetrievableServiceAssertionWrapper':
        """
        Set all service assertions to be made at the root of the pipeline and before processing the step.

        :param service_assertions: base types that need to be available in the execution context at the root of the pipeline
        :type service_assertions: List[Type]
        """
        return GlobalyRetrievableServiceAssertionWrapper(wrapped=self, service_assertions=service_assertions)

    def assert_has_services_at_execution(self, *service_assertions: List[Type['BaseService']]) -> 'LocalServiceAssertionWrapper':
        """
        Set all service assertions to be made before processing the step.

        :param service_assertions: base types that need to be available in the execution context before the execution of the step
        :type service_assertions: List[Type]
        """
        return LocalServiceAssertionWrapper(wrapped=self, service_assertions=service_assertions)

    def _assert_at_lifecycle(self, context: CX):
        """
        Assert that the context has all the services required to process the step.
        This method will be registred within a handler method's _will_process or _did_process,
        or other lifecycle methods like these.
        """
        for service_assertion in self.service_assertions:
            self._assert(
                context.has_service(service_assertion),
                'Missing Service {0}'.format(service_assertion.__name__)
            )

    def _assert(self, condition: bool, err_message: str, context: CX = None):
        """
        Assert that the ``condition`` is true.
        If not, raise an exception with the ``message``.
        The exception will be logged with the logger in the ``context``.
        If the ``context`` is in :class:`ExecutionPhase` ``.PROD``,
        the exception will not be raised and only logged.

        it is good to call assertions here in a context-dependent way.
        For more information on contextual validation, read
        `Martin Fowler's article on Contextual Validation <https://martinfowler.com/bliki/ContextualValidation.html>`_.

        :param condition: condition to assert
        :param err_message: message to log and raise if the condition is false
        :param context: execution context to log the exception, and not raise it if in ``PROD`` mode.
        """
        return self._assert_equals(condition, True, err_message, context)

    def _assert_equals(self, a: Any, b: Any, err_message: str, context: CX):
        """
        Assert that the ``condition`` is true.
        If not, raise an exception with the ``message``.
        The exception will be logged with the logger in the ``context``.
        If the ``context`` is in :class:`ExecutionPhase` ``.PROD``,
        the exception will not be raised and only logged.

        :param a: element to compare to b with `==`
        :param b: element to compare to a with `==`
        :param err_message: message to log and raise if the condition is false
        :param context: execution context to log the exception, and not raise it if in ``PROD`` mode.
        """
        if context is None:
            context = CX()

        try:
            assert a == b, err_message.strip() + f" - raised from {context.get_identifier()}<{str(self)}>."
        except AssertionError as e:
            context.flow.log_error(e)

            # Crash when not in production context:
            if context.execution_phase != ExecutionPhase.PROD:
                raise e  # A base service or base step assertion failed.


class BaseTransformer(
    _CouldHaveContext,
    _HasSavers,
    _TransformerStep,
    _HasHyperparamsSpace,
    _HasHyperparams,
    _HasSetupTeardownLifecycle,
    BaseService,
    ABC
):
    """
    Base class for a pipeline step that can only be transformed.

    Every step has hyperparemeters, and hyperparameters spaces that can be set before the
    learning process begins (see :class:`_HasHyperparams`, and :class:`_HasHyperparamsSpace` for more info).

    Example usage :

    .. code-block:: python

        class AddN(BaseTransformer):
            def __init__(self, add=1):
                super().__init__(hyperparams=HyperparameterSamples({'add': add}))

            def transform(self, data_inputs):
                if not isinstance(data_inputs, np.ndarray):
                    data_inputs = np.array(data_inputs)

                return data_inputs + self.hyperparams['add']

            def inverse_transform(self, processed_outputs):
                if not isinstance(data_inputs, np.ndarray):
                    data_inputs = np.array(data_inputs)

                return data_inputs - self.hyperparams['add']


    .. note:: All heavy initialization logic should be done inside the *setup* method (e.g.: things inside GPU), and NOT in the constructor.

    .. seealso::
        :class:`~neuraxle.pipeline.Pipeline`,
        :class:`~neuraxle.base.BaseTransformer`,
        :class:`~neuraxle.base._TransformerStep`,
        :class:`~neuraxle.base.NonFittableMixin`,
        :class:`~neuraxle.base.NonTransformableMixin`
        :class:`~neuraxle.steps.numpy.AddN`,
    """

    def __init__(
            self,
            hyperparams: HyperparameterSamples = None,
            hyperparams_space: HyperparameterSpace = None,
            config: Union[Dict, RecursiveDict] = None,
            name: str = None,
            savers: List[BaseSaver] = None,
    ):
        # Must read: https://www.neuraxle.org/stable/classes_and_modules_overview.html
        BaseService.__init__(self, config=config, name=name)
        _HasSetupTeardownLifecycle.__init__(self)
        _HasHyperparams.__init__(self, hyperparams=hyperparams)
        _HasHyperparamsSpace.__init__(self, hyperparams_space=hyperparams_space)
        _TransformerStep.__init__(self)
        _HasSavers.__init__(self, savers=savers)
        _CouldHaveContext.__init__(self)

        self.is_train: bool = True

    def set_train(self, is_train: bool = True):
        """
        This method overrides the method of BaseStep to also consider the wrapped step as well as self.
        Set pipeline step mode to train or test.

        .. note::
            This is a recursive method used in :class:̀_HasChildrenMixin`.
        .. seealso::
            :func:`~neuraxle.base._HasRecursiveMethods.apply`,
            :func:`~neuraxle.base._HasChildrenMixin._apply`
            :func:`~neuraxle.base.BaseTransformer._set_train`
        """
        self.apply(method='_set_train', is_train=is_train)
        return self

    def _set_train(self, is_train) -> Optional[RecursiveDict]:
        self.is_train = is_train
        return RecursiveDict()

    def _repr(self, level=0, verbose=False) -> str:
        output = self.__class__.__name__ + "("
        _params = BaseService._repr_params(self, level=level, verbose=verbose)
        if _params.startswith(", "):
            _params = _params.replace(', ', '', 1)
        output += _params
        _hparams = _HasHyperparams._repr_params(self, level=level, verbose=verbose)
        if _hparams.startswith(", ") and len(_params) == 0:
            _hparams = _hparams.replace(', ', '', 1)
        output += _hparams
        output += ")"
        return output


def _sklearn_to_neuraxle_step(step) -> BaseTransformer:
    if step is None:
        step = Identity()
    elif hasattr(step, '_get_param_names') and hasattr(step, '_more_tags') \
            and hasattr(step, '_check_n_features') and hasattr(step, '_validate_data'):
        import neuraxle.steps.sklearn
        step = neuraxle.steps.sklearn.SKLearnWrapper(step)
        step.set_name(step.get_wrapped_sklearn_predictor().__class__.__name__)
    return step


BaseTransformerT = TypeVar('BaseTransformerT', bound=BaseTransformer)


class BaseStep(_FittableStep, BaseTransformer, ABC):
    """
    Base class for a transformer step that can also be fitted.

    If a step is not fittable, you can inherit from :class:`BaseTransformer` instead.
    If a step is not transformable, you can inherit from :class:`NonTransformableMixin`.
    A step should only change its state inside :func:`~neuraxle.base._FittableStep.fit` or :func:`~neuraxle.base._FittableStep.fit_transform` (see :class:`_FittableStep` for more info).
    Every step has hyperparemeters, and hyperparameters spaces that can be set before the learning process begins (see :class:`_HasHyperparams`, and :class:`_HasHyperparamsSpace` for more info).

    Example usage :

    .. code-block:: python

        class Normalize(BaseStep):
            def __init__(self):
                BaseStep.__init__(self)
                self.mean = None
                self.std = None

            def fit(
                self, data_inputs: ARG_X_INPUTTED, expected_outputs: ARG_Y_EXPECTED = None
            ) -> 'BaseStep':
                self._calculate_mean_std(data_inputs)
                return self

            def _calculate_mean_std(self, data_inputs: ARG_X_INPUTTED):
                self.mean = np.array(data_inputs).mean(axis=0)
                self.std = np.array(data_inputs).std(axis=0)

            def fit_transform(
                self, data_inputs: ARG_X_INPUTTED, expected_outputs: ARG_Y_EXPECTED = None
            ) -> Tuple['BaseStep', ARG_Y_PREDICTD]:
                self.fit(data_inputs, expected_outputs)
                return self, (np.array(data_inputs) - self.mean) / self.std

            def transform(self, data_inputs: ARG_X_INPUTTED) -> ARG_Y_PREDICTD:
                if self.mean is None or self.std is None:
                    self._calculate_mean_std(data_inputs)
                return (np.array(data_inputs) - self.mean) / self.std

        p = Pipeline([
            Normalize()
        ])

        p, outputs = p.fit_transform(data_inputs, expected_outputs)


    .. seealso::
        :class:`~neuraxle.base.BaseTransformer`,
        :class:`~neuraxle.base._TransformerStep`,
        :class:`~neuraxle.base.NonFittableMixin`,
        :class:`~neuraxle.base.NonTransformableMixin`
        :class:`~neuraxle.base.ExecutionContext`,
        :class:`~neuraxle.pipeline.Pipeline`,
    """
    # Documentation: https://www.neuraxle.org/stable/classes_and_modules_overview.html
    pass


BaseStepT = TypeVar('BaseStepT', bound=BaseStep)


class MixinForBaseTransformer:
    """
    Any steps/transformers within a pipeline that inherits of this class should implement BaseStep/BaseTransformer and
    initialize it before any mixin. This class checks that its the case at initialization.
    """

    def __init__(self):
        self._ensure_basetransformer_init_called()

    def _ensure_basetransformer_init_called(self):
        """
        Assert that BaseTransformer's init method has been called.
        """
        assert isinstance(self, BaseTransformer), "This class should be of type BaseTransformer."
        if not all(map(
            lambda x: hasattr(self, x),
            ('name', 'savers', 'is_initialized', 'is_train', 'is_invalidated', 'setup', '_teardown')
        )):
            raise RuntimeError(
                f'Please initialize Mixins in the good order. The present Mixin should '
                f'be initialized after BaseTransformer. '
                f'Got: {inspect.getmro(self.__class__)}. '
                f'Visit https://www.neuraxle.org/stable/classes_and_modules_overview.html '
                f'for more information.'
            )


class MetaStepMixin(MixinForBaseTransformer, MetaServiceMixin):
    """
    A class to represent a step that wraps another step. It can be used for many things.
    For example, :class:`~neuraxle.steps.loop.ForEachDataInput` adds a loop before any calls to the wrapped step :

    .. code-block:: python

        class ForEachDataInput(MetaStepMixin, BaseStep):
            def __init__(
                self,
                wrapped: BaseStep
            ):
                BaseStep.__init__(self)
                MetaStepMixin.__init__(self, wrapped)

            def fit(self, data_inputs: ARG_X_INPUTTED, expected_outputs: ARG_Y_EXPECTED = None) -> 'BaseStep':
                if expected_outputs is None:
                    expected_outputs = [None] * len(data_inputs)
                for di, eo in zip(data_inputs, expected_outputs):
                    self.wrapped = self.wrapped.fit(di, eo)
                return self

            def transform(self, data_inputs: ARG_X_INPUTTED) -> ARG_Y_PREDICTD:
                outputs = []
                for di in data_inputs:
                    output = self.wrapped.transform(di)
                    outputs.append(output)
                return outputs

            def fit_transform(
                self, data_inputs: ARG_X_INPUTTED, expected_outputs: ARG_Y_EXPECTED = None
            ) -> Tuple['BaseStep', ARG_Y_PREDICTD]:
                if expected_outputs is None:
                    expected_outputs = [None] * len(data_inputs)
                outputs = []
                for di, eo in zip(data_inputs, expected_outputs):
                    self.wrapped, output = self.wrapped.fit_transform(di, eo)
                outputs.append(output)
                return self, outputs

    .. seealso::
        :class:`~neuraxle.steps.loop.ForEachDataInput`,
        :class:`~neuraxle.steps.loop.StepClonerForEachDataInput`
    """

    def __init__(self, wrapped: BaseServiceT = None, savers: List[BaseSaver] = None):
        if savers is None:
            savers = []
        wrapped = _sklearn_to_neuraxle_step(wrapped)
        MetaServiceMixin.__init__(self, wrapped)
        MixinForBaseTransformer.__init__(self)
        savers.append(MetaStepJoblibStepSaver())
        self.savers.extend(savers)

    def handle_fit_transform(
        self, data_container: TrainDACT, context: CX
    ) -> Tuple[BaseStep, PredsDACT]:
        new_self, data_container = super().handle_fit_transform(data_container, context)

        return new_self, data_container

    def handle_transform(self, data_container: TrainDACT, context: CX) -> PredsDACT:
        data_container = super().handle_transform(data_container, context)

        return data_container

    def _fit_transform_data_container(
        self, data_container: TrainDACT, context: CX
    ) -> Tuple[BaseStep, PredsDACT]:
        self.wrapped, data_container = self.wrapped.handle_fit_transform(data_container, context)
        return self, data_container

    def _fit_data_container(self, data_container: TrainDACT, context: CX) -> BaseStep:
        self.wrapped = self.wrapped.handle_fit(data_container, context)
        return self

    def _transform_data_container(self, data_container: TrainDACT, context: CX) -> PredsDACT:
        data_container = self.wrapped.handle_transform(data_container, context)
        return data_container

    def _inverse_transform_data_container(self, data_container: PredsDACT, context: CX) -> TrainDACT:
        data_container = self.wrapped.handle_inverse_transform(data_container, context)
        return data_container

    def fit_transform(
        self, data_inputs: ARG_X_INPUTTED, expected_outputs: ARG_Y_EXPECTED = None
    ) -> Tuple[BaseStep, ARG_Y_PREDICTD]:
        self.wrapped, data_inputs = self.wrapped.fit_transform(data_inputs, expected_outputs)
        return self, data_inputs

    def fit(self, data_inputs: ARG_X_INPUTTED, expected_outputs: ARG_Y_EXPECTED = None) -> BaseStep:
        self.wrapped = self.wrapped.fit(data_inputs, expected_outputs)
        return self

    def transform(self, data_inputs: ARG_X_INPUTTED) -> ARG_Y_PREDICTD:
        data_inputs = self.wrapped.transform(data_inputs)
        return data_inputs

    def inverse_transform(self, processed_outputs: ARG_Y_PREDICTD) -> ARG_X_INPUTTED:
        data_inputs = self.wrapped.inverse_transform(processed_outputs)
        return data_inputs


class MetaStep(MetaStepMixin, BaseStep):
    def __init__(
            self,
            wrapped: BaseServiceT = None,
            hyperparams: HyperparameterSamples = None,
            hyperparams_space: HyperparameterSpace = None,
            name: str = None,
            savers: List[BaseSaver] = None,
    ):
        BaseStep.__init__(
            self,
            hyperparams=hyperparams,
            hyperparams_space=hyperparams_space,
            name=name,
            savers=savers,
        )
        MetaStepMixin.__init__(self, wrapped=wrapped)


class MetaStepJoblibStepSaver(JoblibStepSaver):
    """
    Custom saver for meta step mixin.
    """

    def __init__(self):
        JoblibStepSaver.__init__(self)

    def save_step(self, step: 'MetaStep', context: CX) -> MetaStep:
        """
        Save MetaStepMixin.

        # . Save wrapped step.
        # . Strip wrapped step form the meta step mixin.
        # . Save meta step with wrapped step savers.

        :param step: meta step to save
        :param context: execution context
        :return:
        """
        # First, save the wrapped step savers
        wrapped_step_savers = []
        if step.wrapped.should_save():
            wrapped_step_savers.extend(step.wrapped.get_savers())
        else:
            wrapped_step_savers.append(None)

        # Second, save the wrapped step
        step.wrapped.save(context)

        step.wrapped_step_name_and_savers = (step.wrapped.name, wrapped_step_savers)

        # Third, strip the wrapped step from the meta step
        del step.wrapped

        return step

    def load_step(self, step: 'MetaStep', context: CX) -> 'MetaStep':
        """
        Load MetaStepMixin.

        # . Loop through all of the sub steps savers, and only load the sub steps that have been saved.
        # . Refresh steps

        :param step: step to load
        :param context: execution context
        :return: loaded truncable steps
        """
        step_name, savers = step.wrapped_step_name_and_savers

        if savers is None:
            # keep wrapped step as it is if it hasn't been saved
            pass
        else:
            # load each sub step with their savers
            sub_step_to_load = Identity(name=step_name, savers=savers)
            sub_step = sub_step_to_load.load(context)
            step.wrapped = sub_step

        return step


NamedStepsList = List[Union[Tuple[str, BaseTransformerT], BaseTransformerT]]


class NonFittableMixin(MixinForBaseTransformer):
    """
    A pipeline step that requires no fitting: fitting just returns self when called to do no action.
    Note: fit methods are not implemented
    """

    def _fit_data_container(self, data_container: TrainDACT, context: CX) -> BaseStep:
        return self

    def _fit_transform_data_container(self, data_container: TrainDACT, context: CX) -> Tuple[BaseStep, PredsDACT]:
        return self, self._transform_data_container(data_container, context)

    def fit(self, data_inputs: ARG_X_INPUTTED, expected_outputs: ARG_Y_EXPECTED = None) -> BaseStep:
        """
        Don't fit.

        :param data_inputs: the data that would normally be fitted on.
        :param expected_outputs: the data that would normally be fitted on.
        :return: self
        """
        return self


class NonTransformableMixin(MixinForBaseTransformer):
    """
    A pipeline step that has no effect at all but to return the same data without changes.
    Transform method is automatically implemented as changing nothing.

    Example :

    .. code-block:: python

        class PrintOnFit(NonTransformableMixin, BaseStep):
            def __init__(self):
                BaseStep.__init__(self)

            def fit(self, data_inputs: ARG_X_INPUTTED, expected_outputs: ARG_Y_EXPECTED = None) -> BaseStep:
                print((data_inputs, expected_outputs))
                return self

    .. note::
        Fit methods are not implemented.
    """

    def _fit_transform_data_container(self, data_container: TrainDACT, context: CX):
        return self._fit_data_container(data_container, context), data_container

    def _transform_data_container(self, data_container: TrainDACT, context: CX) -> PredsDACT:
        """
        Do nothing - return the same data.

        :param data_container: data container
        :param context: execution context
        :return: data container
        """
        return data_container

    def transform(self, data_inputs: ARG_X_INPUTTED) -> ARG_Y_PREDICTD:
        """
        Do nothing - return the same data.

        :param data_inputs: the data to process
        :return: the ``data_inputs``, unchanged.
        """
        return data_inputs

    def inverse_transform(self, processed_outputs: ARG_Y_PREDICTD) -> ARG_X_INPUTTED:
        """
        Do nothing - passthrough to return the same data.

        :param processed_outputs: the data to process
        :return: the ``processed_outputs``, unchanged.
        """
        return processed_outputs


class TruncableJoblibStepSaver(JoblibStepSaver):
    """
    Step saver for a TruncableSteps.
    TruncableJoblibStepSaver saves, and loads all of the sub steps using their savers.

    .. seealso::
        :class:`JoblibStepSaver`,
        :class:`TruncableSteps`,
        :class:`BaseSaver`
    """

    def __init__(self):
        JoblibStepSaver.__init__(self)

    def save_step(self, step: 'TruncableSteps', context: CX):
        """
        # . Loop through all the steps, and save the ones that need to be saved.
        # . Add a new property called sub step savers inside truncable steps to be able to load sub steps when loading.
        # . Strip steps from truncable steps at the end.

        :param step: step to save
        :param context: execution context
        :return:
        """

        # First, save all of the sub steps with the right execution context.
        sub_steps_savers = []
        for i, (name, sub_step) in enumerate(step.items()):
            if sub_step.should_save():
                sub_steps_savers.append((name, sub_step.get_savers()))
                sub_step.save(context)
            else:
                sub_steps_savers.append((name, None))

        step.sub_steps_savers = sub_steps_savers

        # Third, strip the sub steps from truncable steps before saving
        if hasattr(step, 'steps'):
            del step.steps
            del step.steps_as_tuple

        return step

    def load_step(self, step: 'TruncableSteps', context: CX) -> 'TruncableSteps':
        """
        # . Loop through all of the sub steps savers, and only load the sub steps that have been saved.
        # . Refresh steps

        :param step: step to load
        :param context: execution context
        :return: loaded truncable steps
        """
        step.steps_as_tuple = []

        for step_name, savers in step.sub_steps_savers:
            if (savers is None):
                # keep step as it is if it hasn't been saved
                step.steps_as_tuple.append((step_name, step[step_name]))
            else:
                # Load each sub step with their savers
                sub_step_to_load = Identity(name=step_name, savers=savers)
                sub_step = sub_step_to_load.load(context)
                step.steps_as_tuple.append((step_name, sub_step))

        operation = getattr(step, '_refresh_steps', None)
        if callable(operation):
            step._refresh_steps()

        return step


class TruncableStepsMixin(_TruncableMixin, _HasChildrenMixin):
    """
    A mixin for steps that can be truncated,
    such as for instance a :class:`~neuraxle.pipeline.Pipeline`.
    """

    def __init__(
        self,
        steps_as_tuple: NamedStepsList,
        mute_step_renaming_warning: bool = True,
    ):
        _HasChildrenMixin.__init__(self)
        _TruncableMixin.__init__(self)
        self.warn_step_renaming = not mute_step_renaming_warning
        self.set_steps(steps_as_tuple, invalidate=False)

    def set_steps(self, steps_as_tuple: NamedStepsList, invalidate=True) -> 'TruncableStepsMixin':
        """
        Set steps as tuple.

        :param steps_as_tuple: list of tuple containing step name and step
        :return:
        """
        steps_as_tuple = self._wrap_non_base_steps(steps_as_tuple)
        self.steps_as_tuple: NamedStepsList = self._patch_missing_names(steps_as_tuple)
        self._refresh_steps(invalidate=invalidate)
        return self

    def get_children(self) -> List[BaseServiceT]:
        """
        Get the list of sub step inside the step with children.

        :return: children steps
        """
        return list(self.values())

    def _wrap_non_base_steps(self, steps_as_tuple: NamedStepsList) -> NamedStepsList:
        """
        If some steps are not of type BaseStep, we'll try to make them of this type. For instance, sklearn objects
        will be wrapped by a SKLearnWrapper here.

        :param steps_as_tuple: a list of steps or of named tuples of steps (e.g.: NamedStepsList)
        :return: a NamedStepsList
        """
        wrapped = []
        for step in steps_as_tuple:
            class_name = None
            if isinstance(step, tuple):
                class_name = step[0]
                step = step[1]

            step = _sklearn_to_neuraxle_step(step)

            if class_name is None:
                class_name = step.get_name()

            wrapped.append((class_name, step))
        return wrapped

    def _patch_missing_names(self, steps_as_tuple: NamedStepsList) -> NamedStepsList:
        """
        Make sure that each sub step has a unique name, and add a name to the sub steps that don't have one already.

        :param steps_as_tuple: a NamedStepsList
        :return: a NamedStepsList with fixed names
        """
        names_yet = set()
        patched = []
        for class_name, step in steps_as_tuple:
            _name = class_name
            if class_name in names_yet:
                if self.warn_step_renaming:
                    warnings.warn(
                        "Named pipeline tuples must be unique. "
                        "Will rename '{}' because it already exists.".format(class_name))

                _name = self._rename_step(step_name=_name, class_name=class_name, names_yet=names_yet)
                step.set_name(_name)

            step = (_name, step)
            names_yet.add(step[0])
            patched.append(step)
        return patched

    def _rename_step(self, step_name, class_name, names_yet: set):
        """
        Rename step by adding a number suffix after the class name.
        Ensure uniqueness with the names yet parameter.

        :param step_name: step name
        :param class_name: class name
        :param names_yet: names already taken
        :return: new step name
        """
        # Add suffix number to name if it is already used to ensure name uniqueness.
        i = 1
        while step_name in names_yet:
            step_name = class_name + str(i)
            i += 1
        return step_name

    def _refresh_steps(self, invalidate=True):
        """
        Private method to refresh inner state after having edited ``self.steps_as_tuple``
        (recreate ``self.steps`` from ``self.steps_as_tuple``).
        """
        self.steps: OrderedDict = OrderedDict(self.steps_as_tuple)
        for name, step in self.items():
            step.name = name
        if invalidate:
            self._invalidate()

    def _step_index_to_name(self, step_index):
        if step_index == len(self.items()):
            return None

        name, _ = self.steps_as_tuple[step_index]
        return name

    def __setitem__(self, key: Union[slice, int, str], new_step: BaseServiceT):
        """
        Set one step with a key, and a value.

        :param key: slice, index, or step name
        :param new_step: step
        """
        if isinstance(key, str):
            index = 0
            for step_index, current_step_name in enumerate(self.keys()):
                if current_step_name == key:
                    index = step_index

            new_step.set_name(key)
            self.steps[index] = new_step
            self.steps_as_tuple[index] = (key, new_step)
        else:
            raise ValueError(
                'type {0} not supported yet in TruncableSteps.__setitem__, please implement it if you need it'.format(
                    type(key)))

    def __getitem__(self, key: Union[slice, int, str]) -> Union[BaseServiceT, List[BaseServiceT]]:
        """
        Truncate self with a slice, an index or a step name.

        Example :

        .. code-block:: python

            p = Pipeline([
                ('1', SomeStep()),
                ('2', SomeStep()),
                ('3', SomeStep())
            ])
            p[0] # returns the first SomeStep()
            p[0:2] # returns a TruncableSteps containing the first, and second SomeStep()
            p['2'] # returns the second SomeStep()


        :param key: slice, index, or step name
        :return: truncated self
        .. seealso::
            :class:`~neuraxle.data_container.DataContainer`,
            `Getting model attributes from scikit-learn pipeline on stackoverflow <https://stackoverflow.com/questions/28822756/getting-model-attributes-from-scikit-learn-pipeline/58359509#58359509>`_
        """
        if isinstance(key, slice):
            self_shallow_copy = copy(self)

            start = key.start
            stop = key.stop
            step = key.step
            if isinstance(key.start, int):
                start = self._step_index_to_name(key.start)
            if isinstance(key.stop, int):
                stop = self._step_index_to_name(key.stop)

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

    def items(self) -> ItemsView:
        """
        Returns all of the steps as tuples items (step_name, step).

        :return: step items tuple : (step name, step)
        """
        return copy(self.steps_as_tuple)

    def keys(self) -> KeysView:
        """
        Returns the step names.

        :return: list of step names
        """
        return self.steps.keys()

    def values(self) -> ValuesView:
        """
        Get step values.

        :return: all of the steps
        """
        return self.steps.values()

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


class TruncableSteps(TruncableStepsMixin, BaseStep, ABC):
    """
    Step that contains multiple steps. :class:`Pipeline` inherits form this class.
    It is possible to truncate this step * :func:`~neuraxle.base.TruncableSteps.__getitem__`

    * self.steps contains the actual steps
    * self.steps_as_tuple contains a list of tuple of step name, and step

    .. seealso::
        :class:`~neuraxle.pipeline.Pipeline`,
        :class:`~neuraxle.union.FeatureUnion`
    """

    def __init__(
            self,
            steps_as_tuple: NamedStepsList,
            hyperparams: HyperparameterSamples = dict(),
            hyperparams_space: HyperparameterSpace = dict(),
            mute_step_renaming_warning: bool = True,
    ):
        BaseStep.__init__(self, hyperparams=hyperparams, hyperparams_space=hyperparams_space)
        TruncableStepsMixin.__init__(self, steps_as_tuple, mute_step_renaming_warning=mute_step_renaming_warning)
        self.set_savers([TruncableJoblibStepSaver()] + self.savers)

    def append(self, item: Tuple[str, 'BaseTransformer']) -> 'TruncableSteps':
        """
        Add an item to steps as tuple.

        :param item: item tuple (step name, step)
        :return: self
        """
        self.steps_as_tuple.append(item)
        self._refresh_steps()
        return self

    def pop(self) -> 'BaseTransformer':
        """
        Pop the last step.

        :return: last step
        """
        return self.popitem()[-1]

    def popitem(self, key=None) -> Tuple[str, 'BaseTransformer']:
        """
        Pop the last step, or the step with the given key

        :param key: step name to pop, or None
        :return: last step item

        """
        if key is None:
            item = self.steps_as_tuple.pop()
            self._refresh_steps()
        else:
            item = key, self.steps.pop(key)
            self.steps_as_tuple = list(self.steps.items())
            self._invalidate()
        return item

    def popfront(self) -> 'BaseTransformer':
        """
        Pop the first step.

        :return: first step
        """
        return self.popfrontitem()[-1]

    def popfrontitem(self) -> Tuple[str, 'BaseTransformer']:
        """
        Pop the first step.

        :return: first step item
        """
        item = self.steps_as_tuple.pop(0)
        self._refresh_steps()
        return item

    def split(self, step_type: type) -> List['TruncableSteps']:
        """
        Split truncable steps by a step class (type).

        :param step_type: step class type to split on.
        :return: list of truncable steps containing the splitted steps
        """
        sub_pipelines = []

        previous_sub_pipeline_end_index = 0
        for index, step in enumerate(self.values()):
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

    def ends_with(self, step_type: Type) -> bool:
        """
        Returns true if truncable steps end with a step of the given type.

        :param step_type: step type
        :return: if truncable steps ends with the given step type
        """
        return isinstance(self[-1], step_type)

    def should_save(self):
        """
        Returns if the step needs to be saved or not.

        :return: If self or any of his sub steps should be saved, returns True.

        .. seealso::
            :class:`TruncableJoblibStepSaver`
        """
        if super().should_save():
            return True

        for step in self.values():
            if step.should_save():
                return True
        return False

    def __add__(self, other: 'TruncableSteps') -> 'TruncableSteps':
        """
        Concatenate the given truncable steps to self.

        :param other: other truncable steps
        :return: new truncable steps with concatenated steps
        """
        new_self = copy(self)
        new_self = new_self.set_steps(self.steps_as_tuple + other.steps_as_tuple)
        return new_self


class Identity(NonTransformableMixin, NonFittableMixin, BaseStep):
    """
    A pipeline step that has no effect at all but to return the same data without changes.

    This can be useful to concatenate new features to existing features, such as what AddFeatures do.

    Identity inherits from :class:`NonTransformableMixin` and from :class:`NonFittableMixin` which makes it a class that has no
    effect in the pipeline: it doesn't require fitting, and at transform-time, it returns the same data it received.

    .. seealso::
        :class:`NonTransformableMixin`,
        :class:`NonFittableMixin`,
        :class:`BaseStep`
    """

    def __init__(self, name=None, savers=None):
        if savers is None:
            savers = [JoblibStepSaver()]
        BaseStep.__init__(self, name=name, savers=savers)
        NonFittableMixin.__init__(self)
        NonTransformableMixin.__init__(self)


class TransformHandlerOnlyMixin(MixinForBaseTransformer):
    """
    A pipeline step that only requires the implementation of _transform_data_container.

    .. seealso::
        :class:`BaseStep`,
        :class:`MixinForBaseTransformer`
        :class:`NonFittableMixin`
    """

    @abstractmethod
    def _transform_data_container(self, data_container: TrainDACT, context: CX) -> PredsDACT:
        """
        Transform data container with the given execution context.

        :param data_container: data container
        :param context: execution context
        :return: transformed data container
        """
        raise NotImplementedError('Must implement _transform_data_container in {0}'.format(self.name))

    def transform(self, data_inputs) -> 'HandleOnlyMixin':
        raise Exception(
            'Transform method is not supported for {0}, because it inherits from HandlerMixin. Please use handle_transform instead.'.format(
                self.name))


class HandleOnlyMixin(MixinForBaseTransformer):
    """
    A pipeline step that only requires the implementation of handler methods :

        - _fit_data_container
        - _transform_data_container
        - _fit_transform_data_container (by default, will call the above fit and then transform)

    If forbids only implementing fit or transform or fit_transform without the handles. So it forces the handles.

    .. seealso::
        :class:`BaseStep`,
        :class:`MixinForBaseTransformer`
        :class:`TransformHandlerOnlyMixin`,
        :class:`NonTransformableMixin`,
        :class:`NonFittableMixin`,
        :class:`ForceHandleMixin`,
        :class:`ForceHandleOnlyMixin`
    """

    @abstractmethod
    def _fit_data_container(self, data_container: TrainDACT, context: CX) -> 'BaseTransformer':
        raise NotImplementedError('Must implement _fit_data_container in {0}'.format(self.name))

    @abstractmethod
    def _transform_data_container(self, data_container: TrainDACT, context: CX) -> PredsDACT:
        raise NotImplementedError('Must implement _transform_data_container in {0}'.format(self.name))

    def _fit_transform_data_container(
        self, data_container: TrainDACT, context: CX
    ) -> Tuple['BaseTransformer', PredsDACT]:
        """
        Fit and transform data container with the given execution context. Will do:

        .. code-block:: python

            data_container, context = self._fit_data_container(data_container, context)
            data_container = self._transform_data_container(data_container, context)
            return self, data_container


        :param data_container: data container
        :param context: execution context
        :return: transformed data container
        """
        data_container, context = self._fit_data_container(data_container, context)
        data_container = self._transform_data_container(data_container, context)
        return self, data_container

    def transform(self, data_inputs) -> 'HandleOnlyMixin':
        raise Exception(
            'Transform method is not supported for {0}, because it inherits from HandleOnlyMixin. Please use handle_transform instead.'.format(
                self.name))

    def fit(self, data_inputs: ARG_X_INPUTTED, expected_outputs: ARG_Y_EXPECTED = None) -> 'HandleOnlyMixin':
        raise Exception(
            'Fit method is not supported for {0}, because it inherits from HandleOnlyMixin. Please use handle_fit instead.'.format(
                self.name))

    def fit_transform(self, data_inputs: ARG_X_INPUTTED, expected_outputs: ARG_Y_EXPECTED = None) -> Tuple['HandleOnlyMixin', ARG_Y_PREDICTD]:
        raise Exception(
            'Fit transform method is not supported for {0}, because it inherits from HandleOnlyMixin. Please use handle_fit_transform instead.'.format(
                self.name))


class ForceHandleMixin(MixinForBaseTransformer):
    """
    A step that automatically calls handle methods in the transform, fit, and fit_transform methods.
    A class which inherits from ForceHandleMixin can't use BaseStep's
    _fit_data_container, _fit_transform_data_container and _transform_data_container.
    They must be redefined; failure to do so will trigger an Exception on
    initialisation (and would create infinite loop if these checks were not there).

    .. seealso::
        :class:`BaseStep`,
        :class:'MixinForBaseTransformer'
        :class:`HandleOnlyMixin`,
        :class:`TransformHandlerOnlyMixin`,
        :class:`NonTransformableMixin`,
        :class:`NonFittableMixin`,
        :class:`ForceHandleOnlyMixin`
    """

    def __init__(self, cache_folder=None):
        MixinForBaseTransformer.__init__(self)
        warn_deprecated_arg(self, "cache_folder", None, cache_folder, None)

        if isinstance(self, _FittableStep):
            self._ensure_method_overriden("_fit_data_container", _FittableStep)
            self._ensure_method_overriden("_fit_transform_data_container", _FittableStep)
        self._ensure_method_overriden("_transform_data_container", _TransformerStep)

    def _ensure_method_overriden(self, method_name, original_cls):
        """
        Asserts that a given method of current instance overrides the default one defined in a given class. We assume that current instance inherits from the given class but do not test it (MixinForBaseTransformer test inheritance to BaseTransfromer already).

        :param method_name:
        :param original_cls:
        :return:
        """
        if original_cls.__dict__[method_name] == getattr(self, method_name).__func__:
            raise NotImplementedError(
                f"The ForceHandleMixin class overrides fit, transform and fit_transform to force a call on their handler method counterparts. Failure to redefine basic _fit_data_container, _transform_data_container and _fit_transform_data_container methods can cause an infinite loop. Please define {method_name} in {self.__class__.__name__}.")

    def transform(self, data_inputs: ARG_X_INPUTTED) -> ARG_Y_PREDICTD:
        """
        Using :func:`~neuraxle.base._TransformerStep.handle_transform`, transform data inputs.

        :param data_inputs: data inputs
        :return: outputs
        """
        context, data_container = self._encapsulate_data(data_inputs, None, ExecutionMode.TRANSFORM)

        data_container = self.handle_transform(data_container, context)

        return data_container.data_inputs

    def fit(self, data_inputs: ARG_X_INPUTTED, expected_outputs: ARG_Y_EXPECTED = None) -> 'ForceHandleMixin':
        """
        Using :func:`~neuraxle.base._FittableStep.handle_fit`, fit step with the
        given data inputs, and expected outputs.

        :param data_inputs: data inputs
        :return: fitted self
        """
        context, data_container = self._encapsulate_data(data_inputs, expected_outputs, ExecutionMode.FIT)
        new_self = self.handle_fit(data_container, context)

        return new_self

    def fit_transform(
        self, data_inputs: ARG_X_INPUTTED, expected_outputs: ARG_Y_EXPECTED = None
    ) -> Tuple['ForceHandleMixin', ARG_Y_PREDICTD]:
        """
        Using :func:`~neuraxle.base._FittableStep.handle_fit_transform`,
        fit and transform step with the given data inputs, and expected outputs.

        :param data_inputs: data inputs
        :return: fitted self, outputs
        """
        context, data_container = self._encapsulate_data(data_inputs, expected_outputs, ExecutionMode.FIT_TRANSFORM)
        new_self, data_container = self.handle_fit_transform(data_container, context)

        return new_self, data_container.data_inputs

    def _encapsulate_data(
        self, data_inputs: ARG_X_INPUTTED, expected_outputs: ARG_Y_EXPECTED, execution_mode: ExecutionMode
    ) -> Tuple[CX, TrainDACT]:
        """
        Encapsulate data with :class:`~neuraxle.data_container.DataContainer`.

        :param data_inputs: data inputs
        :param expected_outputs: expected outputs
        :param execution_mode: execution mode
        :return: execution context, data container
        """
        data_container = TrainDACT(data_inputs=data_inputs, expected_outputs=expected_outputs)
        context = CX(execution_mode=execution_mode)
        return context, data_container


class ForceHandleIdentity(ForceHandleMixin, Identity):
    """
    An identity step which forces usage of handler methods. Mostly used in unit tests.
    Useful when you only want to define few methods such as _will_process or _did_process.
    """

    def __init__(self):
        Identity.__init__(self)
        ForceHandleMixin.__init__(self)


class ForceHandleOnlyMixin(ForceHandleMixin, HandleOnlyMixin):
    """
    A step that automatically calls handle methods in the transform, fit, and fit_transform methods.

    It also requires the implementation of handler methods :
        - _transform_data_container
        - _fit_transform_data_container
        - _fit_data_container

    .. seealso::
        :class:`BaseStep`,
        :class:`HandleOnlyMixin`,
        :class:`TransformHandlerOnlyMixin`,
        :class:`NonTransformableMixin`,
        :class:`NonFittableMixin`,
        :class:`ForceHandleMixin`
    """

    def __init__(self, cache_folder=None):
        HandleOnlyMixin.__init__(self)
        ForceHandleMixin.__init__(self, cache_folder)


class IdentityHandlerMethodsMixin(ForceHandleOnlyMixin):
    """
    A step that has a default implementation for all handler methods.

    It is useful for steps that only the following methods :
        - :func:`~neuraxle.base._FittableStep.will_fit`
        - :func:`~neuraxle.base._TransformerStep.will_transform`
        - :func:`~neuraxle.base._FittableStep.will_fit_transform`
        - :func:`~neuraxle.base._TransformerStep.will_process`
        - :func:`~neuraxle.base._TransformerStep.did_process`
        - :func:`~neuraxle.base._FittableStep.did_fit`
        - :func:`~neuraxle.base._TransformerStep.did_transform`
        - :func:`~neuraxle.base._FittableStep.did_fit_transform`

    .. seealso::
        :class:`BaseStep`,
        :class:`HandleOnlyMixin`,
        :class:`TransformHandlerOnlyMixin`,
        :class:`NonTransformableMixin`,
        :class:`NonFittableMixin`,
        :class:`ForceHandleMixin`,
        :class:`HandleOnlyMixin`,
        :class:`ForceHandleOnlyMixin`
    """

    def _fit_data_container(
        self, data_container: TrainDACT, context: CX
    ) -> Tuple['BaseTransformer', DACT]:
        return self

    def _transform_data_container(
        self, data_container: TrainDACT, context: CX
    ) -> PredsDACT:
        return data_container

    def _fit_transform_data_container(
        self, data_container: TrainDACT, context: CX
    ) -> Tuple['BaseTransformer', PredsDACT]:
        return self, data_container


class EvaluableStepMixin(MixinForBaseTransformer):
    """
    A step that can be evaluated with the scoring functions.

    .. seealso::
        :class:`BaseStep`
        :class:'MixinForBaseTransformer'
    """

    @abstractmethod
    def get_score(self):
        raise NotImplementedError()


class FullDumpLoader(Identity):
    """
    Identity step that can load the full dump of a pipeline step.
    Used by :func:`~neuraxle.base._HasSavers.load`.

    Usage example:

    .. code-block:: python

        saved_step = FullDumpLoader(
            name=path,
            stripped_saver=self.stripped_saver
        ).load(context_for_loading, True)


    .. seealso::
        :class:`ExecutionContext`
        :class:`BaseStep`,
        :class:`Identity`
    """

    def __init__(self, name, stripped_saver=None):
        if stripped_saver is None:
            stripped_saver = JoblibStepSaver()
        Identity.__init__(self, name=name, savers=[stripped_saver])

    def load(self, context: CX, full_dump=True) -> BaseStep:
        """
        Load the full dump of a pipeline step.

        :param context: execution context
        :param full_dump: load full dump or not (always true, inherited from :class:`BaseStep`
        :return: loaded step
        """
        if not context.stripped_saver.can_load(self, context):
            raise Exception('Cannot Load Full Dump For Step {}'.format(os.path.join(context.get_path(), self.name)))

        loaded_self = context.stripped_saver.load_step(self, context)

        context.pop()
        return loaded_self.load(context, full_dump)


class AssertionMixin(ForceHandleMixin):
    def __init__(self):
        ForceHandleMixin.__init__(self)

    @abstractmethod
    def _assert_at_lifecycle(self, data_container: DACT, context: CX):
        pass


class WillProcessAssertionMixin(AssertionMixin):
    def _will_process(
        self, data_container: TrainDACT, context: CX
    ) -> Tuple[TrainDACT, CX]:
        """
        Calls self._assert_at_lifecycle(data_container, context).
        """
        data_container, context = super()._will_process(data_container, context)
        self._assert_at_lifecycle(data_container, context)

        return data_container, context


class DidProcessAssertionMixin(AssertionMixin):
    def _did_process(
        self, data_container: PredsDACT, context: CX
    ) -> Tuple[PredsDACT, CX]:
        """
        Calls self._assert_at_lifecycle(data_container,context)
        """
        data_container, context = super()._did_process(data_container, context)
        self._assert_at_lifecycle(data_container, context)

        return data_container, context


class AssertExpectedOutputIsNoneMixin(WillProcessAssertionMixin):
    def _assert_at_lifecycle(self, data_container: DACT, context: CX):
        eo_empty = (data_container.expected_outputs is None) or all(v is None for v in data_container.expected_outputs)
        self._assert(
            eo_empty,
            f"Expected datacontainer.expected_outputs to be a `None` or a list of `None`. Received {data_container.expected_outputs}.",
            context
        )


class AssertExpectedOutputIsNotNoneMixin(WillProcessAssertionMixin):
    def _assert_at_lifecycle(self, data_container: DACT, context: CX):
        eo_empty = (data_container.expected_outputs is None) or all(v is None for v in data_container.expected_outputs)
        self._assert(
            not eo_empty,
            f"Expected datacontainer.expected_outputs to not be a `None` nor a list of `None`. Received {data_container.expected_outputs}.",
            context
        )


class AssertExpectedOutputIsNoneStep(AssertExpectedOutputIsNoneMixin, Identity):
    def __init__(self):
        Identity.__init__(self)
        AssertExpectedOutputIsNoneMixin.__init__(self)


class AssertExpectedOutputIsNone(AssertExpectedOutputIsNoneMixin, Identity):
    def __init__(self):
        Identity.__init__(self)
        AssertExpectedOutputIsNoneMixin.__init__(self)


class LocalServiceAssertionWrapper(WillProcessAssertionMixin, MetaStep):
    """
    Is used to assert the presence of service at execution time for a given step
    """

    def __init__(
        self, wrapped: BaseTransformer = None, service_assertions: List[Type['BaseService']] = None
    ):
        MetaStep.__init__(self, wrapped=wrapped)
        WillProcessAssertionMixin.__init__(self)

        if service_assertions is None:
            service_assertions = []
        self.service_assertions: List[Type['BaseService']] = service_assertions

    def _assert_at_lifecycle(self, data_container: DACT, context: CX):
        """
        Assert self.local_service_assertions are present in the context.
        """
        self.do_assert_has_services(context)

    def do_assert_has_services(self, context: CX):
        """
        Assert that all the necessary services are provided in the execution context.

        :param context: The ExecutionContext for which we test the presence of service.
        """
        for service_type in self.service_assertions:
            err_message = (
                f"Expected context to have service of type {service_type.__name__} but it did not. "
                f"Please register the service {service_type.__name__} inside the ExecutionContext. "
            )
            self._assert(context.has_service(service_type=service_type),
                         err_message,
                         context)


class GlobalyRetrievableServiceAssertionWrapper(LocalServiceAssertionWrapper):
    """
    Is used to assert the presence of service at the start of the pipeline AND at execution time for a given step.
    """

    def _global_assert_has_services(self, context: CX) -> RecursiveDict:
        """
        Intended to be used in a .apply('_global_assert_has_services') call from the outside.
        Is used to test the presence of services at the root of the pipeline.

        See also GlobalServiceAssertionExecutorMixin._apply_service_assertions
        :params context : the execution context
        """
        self.do_assert_has_services(context)
        return RecursiveDict()


class GlobalServiceAssertionExecutorMixin(WillProcessAssertionMixin):
    """
    Any step which inherit of this class will test globaly retrievable service
    assertion of itself and all its children on a will_process call.
    """

    def _assert_at_lifecycle(self, data_container: DACT, context: CX):
        """
        Calls _global_assert_has_services on GlobalyRetrievableServiceAssertionWrapper
        instances that are (recursively) children of this node.

        :param data_container: The DataContainer, probably unused.
        :param context: The ExecutionContext for which we test the presence of service.
        """
        self.apply('_global_assert_has_services', context=context)


class _WithContextStepSaver(BaseSaver):
    """
    Custom saver for steps that have an :class:`ExecutionContext`.
    Loading will inject the saved dependencies inside the :class`ExecutionContext`.

    .. seealso::
        :class:`_HasContext`,
        :class:`BaseSaver`,
        :class:`ExecutionContext`
    """

    def load_step(self, step: 'StepWithContext', context: CX) -> 'StepWithContext':
        """
        Load a step with a context by setting the context as the loading context.

        :param step: step with context
        :param context: execution context to load from
        :return: loaded step with context
        """
        step.context = context
        warnings.warn(
            "Warning! the loading of a StepWithContext instance overrides the context attribute with the one provided at loading.")
        return step

    def save_step(self, step: 'StepWithContext', context: CX) -> 'StepWithContext':
        """
        If needed, remove parents of a step with context before saving.

        :param step: step with context
        :param context: execution context to load from
        :return: saved step with context
        """
        del step.context
        return step

    def can_load(self, step: 'StepWithContext', context: 'CX'):
        return True


class StepWithContext(GlobalServiceAssertionExecutorMixin, MetaStep):
    """
    A step with context is a step that has an :class:`ExecutionContext` as a pre-registered dependency.
    This way, it's possible to call the vanilla "fit", "transform", "predict", and other vanilla methods
    on a step with context, and the handle will be made automatically with the provided context, also
    wrapping the data into a data container.
    """

    def __init__(self, wrapped: 'BaseTransformer', context: CX, raise_if_not_root: bool = True):
        MetaStep.__init__(self, wrapped=wrapped, savers=[_WithContextStepSaver()])
        GlobalServiceAssertionExecutorMixin.__init__(self)
        self.context = context
        self.raise_if_not_root = raise_if_not_root

    def _will_process(
        self, data_container: TrainDACT, context: CX
    ) -> Tuple[TrainDACT, CX]:
        """
        Inject the given context and test service assertions (if any are appliable) before processing the wrapped step.

        :param data_container: data container to process
        :return: data container, execution context
        """
        if self.raise_if_not_root:
            self._assert(len(context) == 0, "StepWithContext should be at the root of the pipeline.", context)

        data_container, context = GlobalServiceAssertionExecutorMixin._will_process(self, data_container, self.context)

        return data_container, context

    def save(self, context: CX = None, full_dump=True) -> 'BaseTransformer':
        """
        Save the wrapped step with its context with the provided context location.

        :param context: Context to save that would override the registred one in self.context. Optional.
        :return: saved step with context
        """
        if context is None:
            context = self.context
        return self.wrapped.save(context, full_dump=full_dump)
