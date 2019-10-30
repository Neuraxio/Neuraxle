from abc import abstractmethod
from collections import OrderedDict
from typing import Union

from neuraxle.base import BaseStep, MetaStepMixin, DataContainer, ExecutionContext
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline


class ForceHandleMixin:
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

    def transform(self, data_inputs) -> 'ForceHandleMixin':
        raise Exception(
            'Transform method is not supported for {0}, because it inherits from ForceHandleMixin. Please use handle_transform instead.'.format(
                self.name))

    def fit(self, data_inputs, expected_outputs=None) -> 'ForceHandleMixin':
        raise Exception(
            'Fit method is not supported for {0}, because it inherits from ForceHandleMixin. Please use handle_fit instead.'.format(
                self.name))

    def fit_transform(self, data_inputs, expected_outputs=None) -> 'ForceHandleMixin':
        raise Exception(
            'Fit transform method is not supported for {0}, because it inherits from ForceHandleMixin. Please use handle_fit_transform instead.'.format(
                self.name))


OPTIONAL_ENABLED_HYPERPARAM = 'enabled'


class Optional(ForceHandleMixin, MetaStepMixin, BaseStep):
    """
    A wrapper to nullify a step : nullify its hyperparams, and also nullify all of his behavior.

    Example usage :

    .. code-block:: python

        p = Pipeline([
            Optional(Identity(), enabled=True)
        ])

    """

    def __init__(self, wrapped: BaseStep, enabled: bool = True):
        ForceHandleMixin.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        BaseStep.__init__(
            self,
            hyperparams=HyperparameterSamples({
                OPTIONAL_ENABLED_HYPERPARAM: enabled
            })
        )

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
        return self, data_container

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
        return self, data_container

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
        return data_container

    def _nullify_hyperparams(self):
        """
        Nullify wrapped step hyperparams using hyperparams_space.nullify().
        """
        hyperparams_space = self.wrapped.get_hyperparams_space()
        self.wrapped.set_hyperparams(hyperparams_space.nullify())


class ChooseOneOrManyStepsOf(Pipeline):
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
        Pipeline.__init__(self, steps)

        if hyperparams is None:
            self.set_hyperparams(HyperparameterSamples({}))
        else:
            self.set_hyperparams(hyperparams)

        self._make_all_steps_optional()

    def set_hyperparams(self, hyperparams: Union[HyperparameterSamples, OrderedDict, dict]) -> BaseStep:
        """
        Set hyperparams for step selection, and nullify the steps that are not chosen.

        :param hyperparams: hyperparams
        :type hyperparams: HyperparameterSamples
        :return: self
        :rtype: BaseStep
        """
        super().set_hyperparams(hyperparams=hyperparams)

        self._validate_choice_hyperparams()
        self._set_all_hyperparams_steps_enabled_by_default()

        return self

    def _validate_choice_hyperparams(self):
        for key in self.hyperparams.keys():
            if key not in self.keys():
                raise ValueError('Invalid Choosen Step {0} in {1}'.format(key, self.name))

    def _set_all_hyperparams_steps_enabled_by_default(self):
        for step_name in self.keys():
            if step_name not in self.hyperparams.keys():
                self.hyperparams[step_name] = {
                    OPTIONAL_ENABLED_HYPERPARAM: True
                }

    def _make_all_steps_optional(self):
        """
        Wrap all steps with :class:`Optional` wrapper.
        """
        step_names = list(self.keys())
        for step_name in step_names:
            self[step_name] = Optional(self[step_name])
        self._refresh_steps()
