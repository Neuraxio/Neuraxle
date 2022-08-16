"""
Neuraxle's Flow Steps
====================================
Wrapper steps that apply some effect to the flow of data in the pipeline.

For instance, the following steps are wrappers of other steps and apply some
control logic to the flow of data into the steps:

- While,
- ForEach,
- BreakIf,
- ContinueIf,
- Try,
- Catch (Except),
- RaiseIf (Throw),
- If & ElseIf & Else,
- Switch

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
from operator import attrgetter
from typing import Callable, Dict, Optional, Tuple, Union

from neuraxle.base import BaseStep, BaseTransformer
from neuraxle.base import ExecutionContext as CX
from neuraxle.base import (ExecutionPhase, ForceHandleOnlyMixin, HandleOnlyMixin, MetaStep, NonFittableMixin,
                           TransformHandlerOnlyMixin, TruncableSteps)
from neuraxle.data_container import DataContainer as DACT
from neuraxle.data_container import ExpandedDataContainer
from neuraxle.hyperparams.distributions import Boolean, Choice
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace
from neuraxle.steps.numpy import NumpyConcatenateOnAxisIfNotEmpty
from neuraxle.union import FeatureUnion

OPTIONAL_ENABLED_HYPERPARAM = 'enabled'


class TrainOrTestOnlyWrapper(ForceHandleOnlyMixin, MetaStep):
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
        MetaStep.__init__(self, wrapped=wrapped)
        ForceHandleOnlyMixin.__init__(self, cache_folder=cache_folder_when_no_handle)

        self.is_train_only = is_train_only

    def _fit_data_container(self, data_container: DACT, context: CX) -> BaseStep:
        """
        :param data_container: data container
        :param context: execution context
        :return: step, data_container
        """
        if self._should_execute_wrapped_step():
            self.wrapped = self.wrapped.handle_fit(data_container, context)
            return self
        return self

    def _fit_transform_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple[BaseStep, DACT]:
        """
        :param data_container: data container
        :param context: execution context
        :return: step, data_container
        """
        if self._should_execute_wrapped_step():
            self.wrapped, data_container = self.wrapped.handle_fit_transform(data_container, context)
            return self, data_container
        return self, data_container

    def _transform_data_container(self, data_container: DACT, context: CX) -> DACT:
        """
        :param data_container: data container
        :param context: execution context
        :return: step, data_container
        """
        if self._should_execute_wrapped_step():
            return self.wrapped.handle_transform(data_container, context)
        return data_container

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
    __test__ = False  # to avoid pytest to run this class as a test class.

    def __init__(self, wrapped: BaseStep):
        TrainOrTestOnlyWrapper.__init__(self, wrapped=wrapped, is_train_only=False)


class ExecuteIf(HandleOnlyMixin, MetaStep):
    def __init__(self, condition_function: Callable, wrapped: BaseStep):
        MetaStep.__init__(self, wrapped)
        HandleOnlyMixin.__init__(self)
        self.condition_function: Callable = condition_function

    def _fit_data_container(self, data_container: DACT, context: CX):
        if self.condition_function(self, data_container, context):
            return MetaStep._fit_data_container(self, data_container, context)
        return self

    def _fit_transform_data_container(self, data_container: DACT, context: CX):
        if self.condition_function(self, data_container, context):
            return MetaStep._fit_transform_data_container(self, data_container, context)
        return self, data_container

    def _transform_data_container(self, data_container: DACT, context: CX):
        if self.condition_function(self, data_container, context):
            return MetaStep._transform_data_container(self, data_container, context)
        return data_container


class IfExecutionPhaseIsThen(ExecuteIf):
    """
    If, at runtime, the execution phase is the same as the one given to the constructor, then execute wrapped step.

    By default, will raise an error if the execution phase is not specified in the context.
    Steps which implement ForceHandleMixin create context with unspecified phase on fit, fit_transform and transform call.
    """

    def __init__(self, phase: ExecutionPhase, wrapped: BaseTransformer, raise_if_phase_unspecified: bool = True):
        ExecuteIf.__init__(self, self.check_context, wrapped)
        self.phase = phase
        self.raise_if_phase_unspecified = raise_if_phase_unspecified

    def check_context(self, step, data_container, context: CX):
        if context.execution_phase == self.phase:
            return True
        elif self.raise_if_phase_unspecified and context.execution_phase == ExecutionPhase.UNSPECIFIED:
            raise ValueError("Execution phase is unspecified while a step requires it to be specified.")
        return False


class ExecutionPhaseSwitch(HandleOnlyMixin, TruncableSteps):
    def __init__(self, phase_to_callable: Dict[ExecutionPhase, BaseTransformer],
                 default: Optional[BaseTransformer] = None):
        phase, steps = zip(*phase_to_callable.items())
        if default:
            steps.append(default)
        TruncableSteps.__init__(self, steps_as_tuple=steps)
        self.phase_to_step_index = {p: i for i, p in enumerate(phase)}
        self.default = default

    def _get_step(self, context):
        if context.execution_phase not in self.phase_to_step_index.keys():
            if self.default is None:
                raise KeyError(f"No behaviour defined for {context.execution_phase}.")
            ind = -1
        else:
            ind = self.phase_to_step_index[context.execution_phase]
        return self.steps_as_tuple[ind][1]

    def _set_step(self, context, step):
        if context.execution_phase not in self.phase_to_step_index.keys():
            if self.default is None:
                raise KeyError(f"No behaviour defined for {context.execution_phase}.")
            ind = -1
        else:
            ind = self.phase_to_step_index[context.execution_phase]
        self.steps_as_tuple[ind] = (self.steps_as_tuple[ind][0], step)
        return self

    def _fit_data_container(self, data_container: DACT, context: CX) -> 'BaseStep':
        step = self._get_step(context).handle_fit(data_container, context)
        return self._set_step(context, step)

    def _fit_transform_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple['BaseTransformer', DACT]:
        step, data_container = self._get_step(context).handle_fit_transform(data_container, context)
        return self._set_step(context, step), data_container

    def _transform_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple['BaseTransformer', DACT]:
        return self._get_step(context).handle_transform(data_container, context)


class OptionalStep(ForceHandleOnlyMixin, MetaStep):
    """
    A wrapper to nullify a step: nullify its hyperparams, and also nullify all of his behavior.
    Example usage:

    .. code-block:: python

        p = Pipeline([
            OptionalStep(Identity(), enabled=True)
        ])

    .. seealso::
        :class:`~neuraxle.base.MetaStep`,
        :class:`~neuraxle.metaopt.AutoML`,
    """

    def __init__(self, wrapped: BaseTransformer, enabled: bool = True, nullified_return_value=None,
                 cache_folder_when_no_handle=None, use_hyperparameter_space=True, nullify_hyperparams=True):
        hyperparameter_space = HyperparameterSpace({
            OPTIONAL_ENABLED_HYPERPARAM: Boolean()
        }) if use_hyperparameter_space else {}

        MetaStep.__init__(
            self,
            hyperparams=HyperparameterSamples({
                OPTIONAL_ENABLED_HYPERPARAM: enabled
            }),
            hyperparams_space=hyperparameter_space,
            wrapped=wrapped
        )
        ForceHandleOnlyMixin.__init__(self, cache_folder_when_no_handle)

        if nullified_return_value is None:
            nullified_return_value = []
        self.nullified_return_value = nullified_return_value
        self.nullify_hyperparams = nullify_hyperparams

    def _fit_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple['BaseTransformer', DACT]:
        """
        Nullify wrapped step hyperparams, and don't fit the wrapped step.

        :param data_container: data container
        :param context: execution context
        :return: step, data_container
        """
        if self.hyperparams[OPTIONAL_ENABLED_HYPERPARAM]:
            self.wrapped = self.wrapped.handle_fit(data_container, context)
            return self
        else:
            self._nullify_hyperparams()
            return self

    def _fit_transform_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple['BaseTransformer', DACT]:
        """
        Nullify wrapped step hyperparams, and don't fit_transform the wrapped step.

        :param data_container: data container
        :param context: execution context
        :return: step, data_container
        """
        if self.hyperparams[OPTIONAL_ENABLED_HYPERPARAM]:
            self.wrapped, data_container = self.wrapped.handle_fit_transform(data_container, context)
            return self, data_container
        else:
            self._nullify_hyperparams()
            return self, self._passtrough_dact(data_container)

    def _transform_data_container(self, data_container: DACT, context: CX) -> DACT:
        """
        Nullify wrapped step hyperparams, and don't transform the wrapped step.

        :param data_container: data container
        :param context: execution context
        :return: step, data_container
        """
        if self.hyperparams[OPTIONAL_ENABLED_HYPERPARAM]:
            return self.wrapped.handle_transform(data_container, context)
        else:
            self._nullify_hyperparams()
            return self._passtrough_dact(data_container)

    def _nullify_hyperparams(self):
        """
        Nullify wrapped step hyperparams using hyperparams_space.nullify().
        """
        if not self.nullify_hyperparams:
            return
        hyperparams_space = self.wrapped.get_hyperparams_space()
        self.wrapped.set_hyperparams(hyperparams_space.nullify())

    def _passtrough_dact(self, data_container):
        return DACT(
            data_inputs=self.nullified_return_value,
            ids=data_container.ids,
            expected_outputs=self.nullified_return_value
        )


class ChooseStepElseIdentity(OptionalStep):
    def __init__(self, wrapped: BaseTransformer, enabled: bool = True, nullify_hyperparams=True):
        OptionalStep.__init__(self, wrapped, enabled, None, None, True, nullify_hyperparams)

    def _passtrough_dact(self, data_container):
        return data_container


class ChooseOneStepOf(FeatureUnion):
    CHOICE_HYPERPARAM = 'choice'
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

    def __init__(self, steps, default_choice=None):
        FeatureUnion.__init__(self, steps, joiner=SelectNonEmptyDataContainer())

        self._make_all_steps_optional()

        choices = list(self.keys())[:-1]

        if default_choice is None:
            self.update_hyperparams({
                ChooseOneStepOf.CHOICE_HYPERPARAM: choices[0]
            })
        else:
            self.update_hyperparams({
                ChooseOneStepOf.CHOICE_HYPERPARAM: default_choice
            })
        self.update_hyperparams_space({
            ChooseOneStepOf.CHOICE_HYPERPARAM: Choice(choices)
        })

    def set_hyperparams(self, hyperparams: Union[HyperparameterSamples, dict]):
        super().set_hyperparams(hyperparams)
        self._update_optional_hyperparams()
        return self

    def _set_hyperparams(self, hyperparams: HyperparameterSamples) -> HyperparameterSamples:
        ret = super()._set_hyperparams(hyperparams)
        self._update_optional_hyperparams()
        return ret

    def update_hyperparams(self, hyperparams: Union[Dict, HyperparameterSamples]) -> 'BaseTransformer':
        super().update_hyperparams(hyperparams)
        self._update_optional_hyperparams()
        return self

    def _update_hyperparams(self, hyperparams: Union[HyperparameterSamples, dict]):
        """
        Set chosen step hyperparams.

        :param hyperparams: hyperparams
        :type hyperparams: HyperparameterSamples
        :return:
        """
        ret = super()._update_hyperparams(hyperparams)
        self._update_optional_hyperparams()
        return ret

    def _update_optional_hyperparams(self):
        step_names = list(self.keys())
        chosen_step_name = self.hyperparams[self.CHOICE_HYPERPARAM] if self.CHOICE_HYPERPARAM in self.hyperparams \
            else step_names[0]

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
            self[step_name] = OptionalStep(
                self[step_name].set_name('OptionalStep({})'.format(step_name)),
                use_hyperparameter_space=False,
                nullify_hyperparams=False
            )

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

    def __init__(self, steps, joiner: NonFittableMixin = None):
        if joiner is None:
            joiner = NumpyConcatenateOnAxisIfNotEmpty(axis=-1)
        FeatureUnion.__init__(self, steps_as_tuple=steps, joiner=joiner)
        self.set_hyperparams(HyperparameterSamples({}))
        self._make_all_steps_optional()

    def _make_all_steps_optional(self):
        """
        Wrap all steps with :class:`Optional` wrapper.
        """
        step_names = list(self.keys())
        for step_name in step_names[:-1]:
            self[step_name] = OptionalStep(self[step_name])
        self._refresh_steps()


class SelectNonEmptyDataInputs(TransformHandlerOnlyMixin, BaseTransformer):
    """
    A step that selects non empty data inputs.

    .. seealso::
        :class:`~neuraxle.base.TransformHandlerOnlyMixin`,
        :class:`~neuraxle.base.BaseStep`
    """

    def __init__(self):
        BaseTransformer.__init__(self)
        TransformHandlerOnlyMixin.__init__(self)

    def _transform_data_container(self, data_container: DACT, context: CX):
        """
        Handle transform.

        :param data_container: the data container to join
        :param context: execution context
        :return: transformed data container
        """
        data_inputs = [dc.data_inputs for dc in data_container.data_inputs if len(dc.data_inputs) > 0]
        if len(data_inputs) == 1:
            data_inputs = data_inputs[0]

        data_container = DACT(data_inputs=data_inputs, ids=data_container.ids,
                              expected_outputs=data_container.expected_outputs)

        return data_container


class SelectNonEmptyDataContainer(TransformHandlerOnlyMixin, BaseTransformer):
    """
    A step that selects non empty data containers.
    Assumes that the given DataContainer contains a list of DataContainer as data_inputs.

    .. seealso::
        :class:`~neuraxle.base.TransformHandlerOnlyMixin`,
        :class:`~neuraxle.base.BaseStep`
    """

    def __init__(self):
        BaseTransformer.__init__(self)
        TransformHandlerOnlyMixin.__init__(self)

    def _transform_data_container(self, data_container: DACT, context: CX):

        filtered_data_containers = list(filter(
            lambda dc: (len(dc.data_inputs) > 0 or len(dc.expected_outputs) > 0),
            data_container.data_inputs
        ))
        if len(filtered_data_containers) == 1:
            return filtered_data_containers[0]
        else:
            return DACT(
                ids=data_container._ids,
                di=list(map(attrgetter("data_inputs"))),
                eo=list(map(attrgetter("expected_outputs"))),
            )


class ExpandDim(MetaStep):
    """
    Similar to numpys expand_dim function, ExpandDim step expands the dimension of all the data inside the data container.
    ExpandDim sends the expanded data container to the wrapped step.

    This is akin from passing the dact data from `shape` to `[1, *shape]` within the wrapped step,
    to then by default go back to the original shape (optional). The ids will now contain a summary id temporarily.

    ExpandDim returns the transformed expanded dim reduced to its original shape (see :func:`~neuraxle.steps.loop.ExpandedDataContainer.reduce_dim`).

    The wrapped step will receive a single current_id, data_input, and expected output:
        - The ids is a list of one element that contains a single summary id created from all of the current ids.
        - The data_inputs is a list of one element that contains the original expected outputs list.
        - The expected_outputs is a list of one element that contains the original expected outputs list.

    .. seealso::
        :class:`~neuraxle.data_container.ExpandedDataContainer`
    """

    def __init__(self, wrapped: BaseTransformer, then_unexpand: bool = True):
        MetaStep.__init__(self, wrapped)

        self.then_unexpand: bool = then_unexpand

    def _will_process(self, data_container: DACT, context: CX) -> Tuple[ExpandedDataContainer, CX]:
        data_container, context = BaseStep._will_process(self, data_container, context)
        return ExpandedDataContainer.create_from(data_container), context

    def _did_process(self, data_container: ExpandedDataContainer, context: CX) -> DACT:
        data_container: ExpandedDataContainer = super()._did_process(data_container, context)
        if self.then_unexpand:
            data_container = data_container.reduce_dim()
        return data_container


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
        TruncableSteps.__init__(self, [
            ("preprocessing_step", preprocessing_step),
            ("postprocessing_step", postprocessing_step)
        ])
        HandleOnlyMixin.__init__(self)

    def _fit_data_container(self, data_container: DACT,
                            context: CX) -> 'ReversiblePreprocessingWrapper':
        """
        Handle fit by fitting preprocessing step, and postprocessing step.

        :param data_container: data container to fit on
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: self, data_container
        :rtype: (ReversiblePreprocessingWrapper, DataContainer)
        """
        self["preprocessing_step"], data_container = self["preprocessing_step"].handle_fit_transform(
            data_container, context)
        self["postprocessing_step"] = self["postprocessing_step"].handle_fit(
            data_container, context)

        return self

    def _transform_data_container(self, data_container: DACT, context: CX) -> DACT:
        """
        According to the idiom of `(1, 2, reversed(1))`, we do this, in order:

            - `1`. Transform preprocessing step
            - `2`. Transform postprocessing step
            - `reversed(1)`. Inverse transform preprocessing step

        :param data_container: data container to transform
        :param context: execution context
        :return: data_container
        """
        data_container = self["preprocessing_step"].handle_transform(
            data_container, context.push(self["preprocessing_step"]))
        data_container = self["postprocessing_step"].handle_transform(
            data_container, context.push(self["postprocessing_step"]))
        data_container = self["preprocessing_step"].handle_inverse_transform(
            data_container, context.push(self["preprocessing_step"]))

        return data_container

    def _fit_transform_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple[BaseStep, DACT]:
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
            data_container, context.push(self["preprocessing_step"]))
        self["postprocessing_step"], data_container = self["postprocessing_step"].handle_fit_transform(
            data_container, context.push(self["postprocessing_step"]))
        data_container = self["preprocessing_step"].handle_inverse_transform(
            data_container, context.push(self["preprocessing_step"]))

        return self, data_container
