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
from typing import Union

from neuraxle.base import BaseStep, MetaStepMixin, DataContainer, ExecutionContext, TruncableSteps, ResumableStepMixin, \
    HandleOnlyMixin, TransformHandlerOnlyMixin, ForceHandleOnlyMixin, NonFittableMixin
from neuraxle.data_container import ExpandedDataContainer
from neuraxle.hyperparams.distributions import Boolean, Choice
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace
from neuraxle.union import FeatureUnion
import numpy as np

OPTIONAL_ENABLED_HYPERPARAM = 'enabled'


class TrainOrTestOnlyWrapper(ForceHandleOnlyMixin, MetaStepMixin, BaseStep):
    """
    A wrapper to run wrapped step only in test mode, or only in train mode.

    Execute only in test mode:

    .. code-block:: python

        p = TrainOrTestOnlyWrapper(Identity(), is_train_only=True)

    Execute only in train mode:

    .. code-block:: python

        p = TrainOnlyWrapper(Identity(), test_only=False)

    .. seealso::
        :class:`~neuraxle.steps.flow.TrainOnlyWrapper`,
        :class:`~neuraxle.steps.flow.TestOnlyWrapper`,
        :class:`~neuraxle.base.ForceHandleMixin`,
        :class:`~neuraxle.base.MetaStepMixin`,
        :class:`~neuraxle.base.BaseStep`
    """

    def __init__(self, wrapped: BaseStep, is_train_only=True, cache_folder_when_no_handle=None):
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        ForceHandleOnlyMixin.__init__(self, cache_folder=cache_folder_when_no_handle)

        self.is_train_only = is_train_only

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> BaseStep:
        """
        :param data_container: data container
        :param context: execution context
        :return: step, data_container
        """
        if self._should_execute_wrapped_step():
            self.wrapped = self.wrapped.handle_fit(data_container, context)
            return self
        return self

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        """
        :param data_container: data container
        :param context: execution context
        :return: step, data_container
        """
        if self._should_execute_wrapped_step():
            self.wrapped, data_container = self.wrapped.handle_fit_transform(data_container, context)
            return self, data_container
        return self, data_container

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        :param data_container: data container
        :param context: execution context
        :return: step, data_container
        """
        if self._should_execute_wrapped_step():
            return self.wrapped.handle_transform(data_container, context)
        return data_container

    def should_resume(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        context = context.push(self)
        return self._should_execute_wrapped_step() and self.wrapped.should_resume(data_container, context)

    def _should_execute_wrapped_step(self):
        return (self.wrapped.is_train and self.is_train_only) or (not self.wrapped.is_train and not self.is_train_only)


class TrainOnlyWrapper(TrainOrTestOnlyWrapper):
    """
    A wrapper to run wrapped step only in train mode

    Execute only in train mode:

    .. code-block:: python

        p = Pipeline([
            TrainOnlyWrapper(Identity())
        ])

    .. seealso::
        :class:`TrainOrTestOnlyWrapper`,
        :class:`TestOnlyWrapper`
    """

    def __init__(self, wrapped: BaseStep):
        TrainOrTestOnlyWrapper.__init__(self, wrapped=wrapped, is_train_only=True)


class TestOnlyWrapper(TrainOrTestOnlyWrapper):
    """
    A wrapper to run wrapped step only in test mode

    Execute only in train mode:

    .. code-block:: python

        p = Pipeline([
            TestOnlyWrapper(Identity())
        ])

    .. seealso::
        :class:`TrainOrTestOnlyWrapper`,
        :class:`TrainOnlyWrapper`
    """

    def __init__(self, wrapped: BaseStep):
        TrainOrTestOnlyWrapper.__init__(self, wrapped=wrapped, is_train_only=False)


class Optional(ForceHandleOnlyMixin, MetaStepMixin, BaseStep):
    """
    A wrapper to nullify a step : nullify its hyperparams, and also nullify all of his behavior.

    Example usage :

    .. code-block:: python

        p = Pipeline([
            Optional(Identity(), enabled=True)
        ])

    .. seealso::
        :class:`TrainOrTestOnlyWrapper`,
        :class:`TrainOnlyWrapper`
        :class:`~neuraxle.base.MetaStepMixin`,
        :class:`~neuraxle.base.BaseStep`
    """

    def __init__(self, wrapped: BaseStep, enabled: bool = True, nullified_return_value=None,
                 cache_folder_when_no_handle=None, use_hyperparameter_space=True, nullify_hyperparams=True):
        hyperparameter_space = HyperparameterSpace({
            OPTIONAL_ENABLED_HYPERPARAM: Boolean()
        }) if use_hyperparameter_space else {}

        BaseStep.__init__(
            self,
            hyperparams=HyperparameterSamples({
                OPTIONAL_ENABLED_HYPERPARAM: enabled
            }),
            hyperparams_space=hyperparameter_space
        )
        MetaStepMixin.__init__(self, wrapped)
        ForceHandleOnlyMixin.__init__(self, cache_folder_when_no_handle)

        if nullified_return_value is None:
            nullified_return_value = []
        self.nullified_return_value = nullified_return_value
        self.nullify_hyperparams = nullify_hyperparams

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        """
        Nullify wrapped step hyperparams, and don't fit the wrapped step.

        :param data_container: data container
        :param context: execution context
        :return: step, data_container
        """
        if self.hyperparams[OPTIONAL_ENABLED_HYPERPARAM]:
            self.wrapped = self.wrapped.handle_fit(data_container, context)
            return self

        self._nullify_hyperparams()

        return self

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        """
        Nullify wrapped step hyperparams, and don't fit_transform the wrapped step.

        :param data_container: data container
        :param context: execution context
        :return: step, data_container
        """
        if self.hyperparams[OPTIONAL_ENABLED_HYPERPARAM]:
            self.wrapped, data_container = self.wrapped.handle_fit_transform(data_container, context)
            return self, data_container

        self._nullify_hyperparams()

        return self, DataContainer(
            data_inputs=self.nullified_return_value,
            current_ids=data_container.current_ids,
            expected_outputs=self.nullified_return_value
        )

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Nullify wrapped step hyperparams, and don't transform the wrapped step.

        :param data_container: data container
        :param context: execution context
        :return: step, data_container
        """
        if self.hyperparams[OPTIONAL_ENABLED_HYPERPARAM]:
            return self.wrapped.handle_transform(data_container, context)

        self._nullify_hyperparams()
        data_container.set_data_inputs(self.nullified_return_value)

        return DataContainer(
            data_inputs=self.nullified_return_value,
            current_ids=data_container.current_ids,
            expected_outputs=self.nullified_return_value
        )

    def _nullify_hyperparams(self):
        """
        Nullify wrapped step hyperparams using hyperparams_space.nullify().
        """
        if not self.nullify_hyperparams:
            return
        hyperparams_space = self.wrapped.get_hyperparams_space()
        self.wrapped.set_hyperparams(hyperparams_space.nullify())


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
        :class:`~neuraxle.pipeline.Pipeline`
        :class:`Optional`
    """

    def __init__(self, steps, hyperparams=None):
        FeatureUnion.__init__(self, steps, joiner=SelectNonEmptyDataInputs())

        self._make_all_steps_optional()

        if hyperparams is None:
            choices = list(self.keys())[:-1]
            self.set_hyperparams(HyperparameterSamples({
                CHOICE_HYPERPARAM: choices[0]
            }))
            self.set_hyperparams_space(HyperparameterSpace({
                CHOICE_HYPERPARAM: Choice(choices)
            }))

    def set_hyperparams(self, hyperparams: Union[HyperparameterSamples, dict]):
        """
        Set chosen step hyperparams.

        :param hyperparams: hyperparams
        :type hyperparams: HyperparameterSamples
        :return:
        """
        super().set_hyperparams(hyperparams)
        self._update_optional_hyperparams()

        return self

    def update_hyperparams(self, hyperparams: Union[HyperparameterSamples, dict]):
        """
        Set chosen step hyperparams.

        :param hyperparams: hyperparams
        :type hyperparams: HyperparameterSamples
        :return:
        """
        super().update_hyperparams(hyperparams)
        self._update_optional_hyperparams()

        return self

    def _update_optional_hyperparams(self):
        step_names = list(self.keys())
        chosen_step_name = self.hyperparams[CHOICE_HYPERPARAM]
        if chosen_step_name not in step_names:
            raise ValueError('Invalid Chosen Step in {0}'.format(self.name))
        for step_name in step_names[:-1]:
            if step_name == chosen_step_name:
                self[chosen_step_name].set_hyperparams({
                    OPTIONAL_ENABLED_HYPERPARAM: True
                })
            else:
                self[step_name].set_hyperparams({
                    OPTIONAL_ENABLED_HYPERPARAM: False
                })

    def _make_all_steps_optional(self):
        """
        Wrap all steps with :class:`Optional` wrapper.
        """
        step_names = list(self.keys())
        for step_name in step_names[:-1]:
            self[step_name] = Optional(self[step_name].set_name('Optional({})'.format(step_name)),
                                       use_hyperparameter_space=False, nullify_hyperparams=False)

        self._refresh_steps()


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
        :class:`~neuraxle.pipeline.Pipeline`
        :class:`Optional`
    """

    def __init__(self, steps):
        FeatureUnion.__init__(self, steps, joiner=NumpyConcatenateOnCustomAxisIfNotEmpty(axis=-1))
        self.set_hyperparams(HyperparameterSamples({}))
        self._make_all_steps_optional()

    def _make_all_steps_optional(self):
        """
        Wrap all steps with :class:`Optional` wrapper.
        """
        step_names = list(self.keys())
        for step_name in step_names[:-1]:
            self[step_name] = Optional(self[step_name])
        self._refresh_steps()


class NumpyConcatenateOnCustomAxisIfNotEmpty(NonFittableMixin, BaseStep):
    """
    Numpy concetenation step where the concatenation is performed along the specified custom axis.
    """

    def __init__(self, axis):
        """
        Create a numpy concatenate on custom axis object.
        :param axis: the axis where the concatenation is performed.
        :return: NumpyConcatenateOnCustomAxis instance.
        """
        self.axis = axis
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)

    def _transform_data_container(self, data_container, context):
        """
        Handle transform.

        :param data_container: the data container to join
        :param context: execution context
        :return: transformed data container
        """
        data_inputs = self.transform([dc.data_inputs for dc in data_container.data_inputs if len(dc.data_inputs) > 0])
        data_container = DataContainer(data_inputs=data_inputs, current_ids=data_container.current_ids,
                                       expected_outputs=data_container.expected_outputs)
        data_container.set_data_inputs(data_inputs)

        return data_container

    def transform(self, data_inputs):
        """
        Apply the concatenation transformation along the specified axis.
        :param data_inputs:
        :return: Numpy array
        """
        return self._concat(data_inputs)

    def _concat(self, data_inputs):
        return np.concatenate(data_inputs, axis=self.axis)


class SelectNonEmptyDataInputs(TransformHandlerOnlyMixin, BaseStep):
    """
    A step that selects non empty data inputs.

    .. seealso::
        :class:`~neuraxle.base.TransformHandlerOnlyMixin`,
        :class:`~neuraxle.base.BaseStep`
    """

    def __init__(self):
        BaseStep.__init__(self)
        TransformHandlerOnlyMixin.__init__(self)

    def _transform_data_container(self, data_container, context):
        """
        Handle transform.

        :param data_container: the data container to join
        :param context: execution context
        :return: transformed data container
        """
        data_inputs = [dc.data_inputs for dc in data_container.data_inputs if len(dc.data_inputs) > 0]
        if len(data_inputs) == 1:
            data_inputs = data_inputs[0]

        data_container = DataContainer(data_inputs=data_inputs, current_ids=data_container.current_ids,
                                       expected_outputs=data_container.expected_outputs)
        data_container.set_data_inputs(data_inputs)

        return data_container


class ExpandDim(
    ResumableStepMixin,
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
        :class:`~neuraxle.base.ForceAlwaysHandleMixin`,
        :class:`~neuraxle.base.MetaStepMixin`,
        :class:`~neuraxle.base.BaseStep`
        :class:`~neuraxle.base.BaseHasher`
        :class:`~neuraxle.data_container.ExpandedDataContainer`
    """

    def __init__(self, wrapped: BaseStep):
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        ResumableStepMixin.__init__(self)

    def _will_process(self, data_container, context):
        data_container, context = BaseStep._will_process(self, data_container, context)
        return ExpandedDataContainer.create_from(data_container), context

    def _did_process(self, data_container, context):
        data_container = BaseStep._did_process(self, data_container, context)
        return data_container.reduce_dim()

    def resume(self, data_container: DataContainer, context: ExecutionContext):
        context = context.push(self)
        if not isinstance(self.wrapped, ResumableStepMixin):
            raise Exception('cannot resume steps that don\' inherit from ResumableStepMixin')

        old_current_ids = data_container.current_ids

        data_container = self.wrapped.resume(data_container, context)

        expanded_data_container = ExpandedDataContainer(
            data_inputs=data_container.data_inputs,
            expected_outputs=data_container.expected_outputs,
            current_ids=data_container.current_ids,
            summary_id=data_container.summary_id,
            old_current_ids=old_current_ids
        )

        data_container = self._did_process(expanded_data_container, context)

        return data_container

    def should_resume(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        context = context.push(self)
        expanded_data_container = ExpandedDataContainer.create_from(data_container)

        if isinstance(self.wrapped, ResumableStepMixin) and \
                self.wrapped.should_resume(expanded_data_container, context):
            return True

        return False


class ReversiblePreprocessingWrapper(HandleOnlyMixin, TruncableSteps):
    """
    TruncableSteps with a preprocessing step(1), and a postprocessing step(2)
    that inverse transforms with the preprocessing step at the end (1, 2, reversed(1)).

    Example usage :

    .. code-block:: python

        step = ReversiblePreprocessingWrapper(
            preprocessing_step=MultiplyBy2(),
            postprocessing_step=Add10()
        )

        outputs = step.transform(np.array(range(5)))

        assert np.array_equal(outputs, np.array([5, 6, 7, 8, 9]))

    """

    def __init__(self, preprocessing_step, postprocessing_step):
        HandleOnlyMixin.__init__(self)
        TruncableSteps.__init__(self, [
            ("preprocessing_step", preprocessing_step),
            ("postprocessing_step", postprocessing_step)
        ])

    def _fit_data_container(self, data_container: DataContainer,
                            context: ExecutionContext) -> 'ReversiblePreprocessingWrapper':
        """
        Handle fit by fitting preprocessing step, and postprocessing step.

        :param data_container: data container to fit on
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: self, data_container
        :rtype: (ReversiblePreprocessingWrapper, DataContainer)
        """
        self["preprocessing_step"], data_container = \
            self["preprocessing_step"].handle_fit_transform(data_container, context.push(self["preprocessing_step"]))
        self["postprocessing_step"] = \
            self["postprocessing_step"].handle_fit(data_container, context.push(self["postprocessing_step"]))

        current_ids = self.hash(data_container)
        data_container.set_current_ids(current_ids)

        return self

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        According to the idiom of `(1, 2, reversed(1))`, we do this, in order:

            - `1`. Transform preprocessing step
            - `2`. Transform postprocessing step
            - `reversed(1)`. Inverse transform preprocessing step

        :param data_container: data container to transform
        :param context: execution context
        :return: data_container
        """
        data_container = self["preprocessing_step"].handle_transform(data_container,
                                                                     context.push(self["preprocessing_step"]))
        data_container = self["postprocessing_step"].handle_transform(data_container,
                                                                      context.push(self["postprocessing_step"]))

        data_container = self["preprocessing_step"].handle_inverse_transform(data_container,
                                                                             context.push(self["preprocessing_step"]))

        current_ids = self.hash(data_container)
        data_container.set_current_ids(current_ids)

        return data_container

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        """
        According to the idiom of `(1, 2, reversed(1))`, we do this, in order:

            - `1`. Fit Transform preprocessing step
            - `2`. Fit Transform postprocessing step
            - `reversed(1)`. Inverse transform preprocessing step

        :param data_container: data container to transform
        :param context: execution context
        :return: (self, data_container)
        """
        self["preprocessing_step"], data_container = self["preprocessing_step"].handle_fit_transform(
            data_container,
            context.push(self["preprocessing_step"])
        )
        self["postprocessing_step"], data_container = self["postprocessing_step"].handle_fit_transform(
            data_container,
            context.push(self["postprocessing_step"])
        )

        data_container = self["preprocessing_step"].handle_inverse_transform(
            data_container,
            context.push(self["preprocessing_step"])
        )

        current_ids = self.hash(data_container)
        data_container.set_current_ids(current_ids)

        return self, data_container
