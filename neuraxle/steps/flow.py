"""
Neuraxle's Flow Steps
====================================
Pipeline wrapper steps that only implement the handle methods, and don't apply any transformation to the data.

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
from abc import abstractmethod
from typing import Union

from neuraxle.base import BaseStep, MetaStepMixin, DataContainer, ExecutionContext
from neuraxle.data_container import ExpandedDataContainer
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.union import FeatureUnion


class ForceMustHandleMixin:
    """
    A pipeline step that only requires the implementation of handler methods :
        - handle_transform
        - handle_fit_transform
        - handle_fit

    If forbids only implementing fit or transform or fit_transform without the handles. So it forces the handles.

    .. seealso::
        :class:`BaseStep`
    """

    @abstractmethod
    def handle_fit(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        raise NotImplementedError('Must implement handle_fit in {0}'.format(self.name))

    @abstractmethod
    def handle_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        raise NotImplementedError('Must implement handle_transform in {0}'.format(self.name))

    @abstractmethod
    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> (
            'BaseStep', DataContainer):
        raise NotImplementedError('Must implement handle_fit_transform in {0}'.format(self.name))

    def transform(self, data_inputs) -> 'ForceMustHandleMixin':
        raise Exception(
            'Transform method is not supported for {0}, because it inherits from ForceHandleMixin. Please use handle_transform instead.'.format(
                self.name))

    def fit(self, data_inputs, expected_outputs=None) -> 'ForceMustHandleMixin':
        raise Exception(
            'Fit method is not supported for {0}, because it inherits from ForceHandleMixin. Please use handle_fit instead.'.format(
                self.name))

    def fit_transform(self, data_inputs, expected_outputs=None) -> 'ForceMustHandleMixin':
        raise Exception(
            'Fit transform method is not supported for {0}, because it inherits from ForceHandleMixin. Please use handle_fit_transform instead.'.format(
                self.name))


OPTIONAL_ENABLED_HYPERPARAM = 'enabled'


class TrainOnlyWrapper(ForceMustHandleMixin, MetaStepMixin, BaseStep):
    """
    A wrapper to run wrapped step only in test mode

    Execute only in test mode:

    .. code-block:: python

        p = Pipeline([
            TrainOnlyWrapper(Identity())
        ])

    Execute only in train mode:

    .. code-block:: python

        p = Pipeline([
            TrainOnlyWrapper(Identity(), test_only=False)
        ])

    """

    def __init__(self, wrapped: BaseStep, train_only=True):
        ForceMustHandleMixin.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        BaseStep.__init__(self)
        self.test_only = train_only

    def handle_fit(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        """
        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: step, data_container
        :type: (BaseStep, DataContainer)
        """
        if self._should_execute_wrapped_step():
            self.wrapped, data_container = self.wrapped.handle_fit(data_container, context)
            return self, data_container
        return data_container

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> (
            'BaseStep', DataContainer):
        """
        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: step, data_container
        :type: (BaseStep, DataContainer)
        """
        if self._should_execute_wrapped_step():
            self.wrapped, data_container = self.wrapped.handle_fit_transform(data_container, context)
            return self, data_container
        return data_container

    def handle_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: step, data_container
        :type: DataContainer
        """
        if self._should_execute_wrapped_step():
            return self.wrapped.handle_transform(data_container, context)
        return data_container

    def _should_execute_wrapped_step(self):
        return (not self.is_train and self.test_only) or (self.is_train and not self.test_only)


class Optional(ForceMustHandleMixin, MetaStepMixin, BaseStep):
    """
    A wrapper to nullify a step : nullify its hyperparams, and also nullify all of his behavior.

    Example usage :

    .. code-block:: python

        p = Pipeline([
            Optional(Identity(), enabled=True)
        ])

    """

    def __init__(self, wrapped: BaseStep, enabled: bool = True, nullified_return_value=None):
        ForceMustHandleMixin.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        BaseStep.__init__(
            self,
            hyperparams=HyperparameterSamples({
                OPTIONAL_ENABLED_HYPERPARAM: enabled
            })
        )

        if nullified_return_value is None:
            nullified_return_value = []
        self.nullified_return_value = nullified_return_value

    def handle_fit(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        """
        Nullify wrapped step hyperparams, and don't fit the wrapped step.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: step, data_container
        :type: (BaseStep, DataContainer)
        """
        if self.hyperparams[OPTIONAL_ENABLED_HYPERPARAM]:
            self.wrapped, data_container = self.wrapped.handle_fit(data_container, context)
            return self, data_container

        self._nullify_hyperparams()

        return self, DataContainer(
            current_ids=data_container.current_ids,
            data_inputs=self.nullified_return_value,
            expected_outputs=self.nullified_return_value
        )

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> (
            'BaseStep', DataContainer):
        """
        Nullify wrapped step hyperparams, and don't fit_transform the wrapped step.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: step, data_container
        :type: (BaseStep, DataContainer)
        """
        if self.hyperparams[OPTIONAL_ENABLED_HYPERPARAM]:
            self.wrapped, data_container = self.wrapped.handle_fit_transform(data_container, context)
            return self, data_container

        self._nullify_hyperparams()

        return self, DataContainer(
            current_ids=data_container.current_ids,
            data_inputs=self.nullified_return_value,
            expected_outputs=self.nullified_return_value
        )

    def handle_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Nullify wrapped step hyperparams, and don't transform the wrapped step.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: step, data_container
        :type: DataContainer
        """
        if self.hyperparams[OPTIONAL_ENABLED_HYPERPARAM]:
            return self.wrapped.handle_transform(data_container, context)

        self._nullify_hyperparams()
        data_container.set_data_inputs(self.nullified_return_value)

        return DataContainer(
            current_ids=data_container.current_ids,
            data_inputs=self.nullified_return_value,
            expected_outputs=self.nullified_return_value
        )

    def _nullify_hyperparams(self):
        """
        Nullify wrapped step hyperparams using hyperparams_space.nullify().
        """
        hyperparams_space = self.wrapped.get_hyperparams_space()
        self.wrapped.set_hyperparams(hyperparams_space.nullify())


class ChooseOneOrManyStepsOf(FeatureUnion):
    """
    A pipeline to allow choosing many steps using an hyperparameter.

    Example usage :

    .. code-block:: python

        p = Pipeline([
            ChooseOneOrManyStepsOf([
                ('a', Identity()),
                ('b', Identity())
            ])
        ])
        p.set_hyperparams({
            'ChooseOneOrManyStepsOf__a__enabled': True,
            'ChooseOneOrManyStepsOf__b__enabled': False
        })
        # or
        p.set_hyperparams({
            'ChooseOneOrManyStepsOf': {
                'a': { 'enabled': True },
                'b': { 'enabled': False }
            }
        })

    .. seealso::
        :class:`Pipeline`
        :class:`Optional`
    """

    def __init__(self, steps, hyperparams=None):
        FeatureUnion.__init__(self, steps)

        if hyperparams is None:
            self.set_hyperparams(HyperparameterSamples({}))
        else:
            self.set_hyperparams(hyperparams)

        self._make_all_steps_optional()

    def _make_all_steps_optional(self):
        """
        Wrap all steps with :class:`Optional` wrapper.
        """
        step_names = list(self.keys())
        for step_name in step_names:
            self[step_name] = Optional(self[step_name])
        self._refresh_steps()


CHOICE_HYPERPARAM = 'choice'


class ChooseOneStepOf(FeatureUnion):
    """
    A pipeline to allow choosing one step using an hyperparameter.

    Example usage :

    .. code-block:: python

        p = Pipeline([
            ChooseOneStepOf([
                ('a', Identity()),
                ('b', Identity())
            ])
        ])
        p.set_hyperparams({
            'ChooseOneOrManyStepsOf__choice': 'a',
        })
        # or
        p.set_hyperparams({
            'ChooseOneStepOf': {
                'a': { 'enabled': True },
                'b': { 'enabled': False }
            }
        })

    .. seealso::
        :class:`Pipeline`
        :class:`Optional`
    """

    def __init__(self, steps, hyperparams=None):
        FeatureUnion.__init__(self, steps)

        self._make_all_steps_optional()

        if hyperparams is None:
            self.set_hyperparams(HyperparameterSamples({
                CHOICE_HYPERPARAM: list(self.keys())[0]
            }))

    def set_hyperparams(self, hyperparams: Union[HyperparameterSamples, dict]):
        """
        Set chosen step hyperparams.

        :param hyperparams: hyperparams
        :type hyperparams: HyperparameterSamples
        :return:
        """
        super().set_hyperparams(hyperparams)

        step_names = list(self.keys())
        chosen_step_name = self.hyperparams[CHOICE_HYPERPARAM]
        if chosen_step_name not in step_names:
            raise ValueError('Invalid Chosen Step in {0}'.format(self.name))

        for step_name in step_names:
            if step_name == chosen_step_name:
                self[chosen_step_name].set_hyperparams({
                    OPTIONAL_ENABLED_HYPERPARAM: True
                })
            else:
                self[step_name].set_hyperparams({
                    OPTIONAL_ENABLED_HYPERPARAM: False
                })

        return self

    def _make_all_steps_optional(self):
        """
        Wrap all steps with :class:`Optional` wrapper.
        """
        step_names = list(self.keys())
        for step_name in step_names:
            self[step_name] = Optional(self[step_name])

        self._refresh_steps()


class ExpandDim(
    ForceMustHandleMixin,
    MetaStepMixin,
    BaseStep
):
    """
    Similar to numpys expand_dim function, ExpandDim step expands the dimension of all the data inside the data container.
    ExpandDim sends the expanded data container to the wrapped step.
    ExpandDim returns the transformed expanded dim reduced to its original shape (see :func:`~neuraxle.steps.loop.ExpandedDataContainer.reduce_dim`).

    The wrapped step will receive a single current_id, data_input, and expected output:
        - The current_id is a list of one element that contains a single summary hash for all of the current ids.
        - The data_inputs is a list of one element that contains the original expected outputs list.
        - The expected_outputs is a list of one element that contains the original expected outputs list.

    .. seealso::
        :class:`ForceHandleMixin`,
        :class:`MetaStepMixin`,
        :class:`BaseStep`
        :class:`BaseHasher`
        :class:`ExpandedDataContainer`
    """

    def __init__(self, wrapped: BaseStep):
        MetaStepMixin.__init__(self, wrapped)
        BaseStep.__init__(self)

    def handle_transform(self, data_container: DataContainer, context: ExecutionContext):
        """
        Send expanded data container to the wrapped handle_transform method, and returned the reduced transformed data container (back to it's orginal shape).

        :param data_container: data container to transform
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: data container
        :rtype: DataContainer
        """
        expanded_data_container = self._create_expanded_data_container(data_container)

        expanded_data_container = self.wrapped.handle_transform(
            expanded_data_container,
            context.push(self.wrapped)
        )

        return expanded_data_container.reduce_dim()

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext):
        """
        Send expanded data container to the wrapped handle_fit_transform method,
        and returned the reduced transformed data container (back to it's orginal shape).

        :param data_container: data container to fit_transform
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: data container
        :rtype: DataContainer
        """
        expanded_data_container = self._create_expanded_data_container(data_container)

        self.wrapped, expanded_data_container = self.wrapped.handle_fit_transform(
            expanded_data_container,
            context.push(self.wrapped)
        )

        return self, expanded_data_container.reduce_dim()

    def handle_fit(self, data_container: DataContainer, context: ExecutionContext):
        """
        Send expanded data container to the wrapped handle_fit method,
        and returned the reduced transformed data container (back to it's orginal shape).

        :param data_container: data container to fit_transform
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: data container
        :rtype: DataContainer
        """
        expanded_data_container = self._create_expanded_data_container(data_container)

        self.wrapped, expanded_data_container = self.wrapped.handle_fit(
            expanded_data_container,
            context.push(self.wrapped)
        )

        return self, expanded_data_container.reduce_dim()

    def _create_expanded_data_container(self, data_container: DataContainer) -> ExpandedDataContainer:
        """
        Create expanded data container.

        :param data_container: data container to expand
        :type data_container: DataContainer
        :return: expanded data container
        :rtype: ExpandedDataContainer
        """
        current_ids = self.hash(data_container)
        data_container.set_current_ids(current_ids)

        expanded_data_container = ExpandedDataContainer.create_from(data_container)

        return expanded_data_container
