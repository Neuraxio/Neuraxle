from abc import abstractmethod
from collections import OrderedDict
from typing import Union

from py._log import warning

from neuraxle.base import BaseStep, MetaStepMixin, DataContainer, ExecutionContext
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline


class ForceHandleMixin:
    """
    A pipeline step that only requires the implementation of handler methods :
        - handle_transform
        - handle_fit_transform
        - handle_fit

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


class Nullify(ForceHandleMixin, MetaStepMixin, BaseStep):
    """
    A wrapper to nullify a step : nullify its hyperparams, and also nullify all of his behavior.

    Example usage :

    .. code-block:: python

        p = Pipeline([
            Nullify(Identity())
        ])

    """

    def __init__(self, wrapped: BaseStep):
        ForceHandleMixin.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        BaseStep.__init__(self)

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
        self._nullify_hyperparams()
        return data_container

    def _nullify_hyperparams(self):
        """
        Nullify wrapped step hyperparams using hyperparams_space.nullify().
        """
        hyperparams_space = self.wrapped.get_hyperparams_space()
        self.wrapped.set_hyperparams(hyperparams_space.nullify())


CHOOSE_ONE_OR_MANY_STEPS_OF_CHOICE_HYPERPARAM = 'choice'


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
            'ChooseOneOrManyStepsOf': {
                'choice': {
                    'a': True
                    'b': False
                }
            }
        })

    .. seealso::
        :class:`Pipeline`
        :class:`Nullify`
    """

    def __init__(self, steps, hyperparams=None):
        Pipeline.__init__(self, steps)

        if hyperparams is None:
            hyperparams = HyperparameterSamples({
                'choice': {}
            })

        self.set_hyperparams(hyperparams)

    def set_hyperparams(self, hyperparams: Union[HyperparameterSamples, OrderedDict, dict]) -> BaseStep:
        """
        Set hyperparams for step selection, and nullify the steps that are not chosen.

        :param hyperparams: hyperparams
        :type hyperparams: HyperparameterSamples
        :return: self
        :rtype: BaseStep
        """
        super().set_hyperparams(hyperparams=hyperparams)

        if CHOOSE_ONE_OR_MANY_STEPS_OF_CHOICE_HYPERPARAM not in self.hyperparams:
            raise ValueError('\'choice\' hyperparam not set in {0} hyperparams'.format(self.name))

        for key in self.hyperparams[CHOOSE_ONE_OR_MANY_STEPS_OF_CHOICE_HYPERPARAM].keys():
            if key not in self.keys():
                raise ValueError('Invalid Choosen Step {0} in {1}'.format(key, self.name))

        for key in self.keys():
            if key not in self.hyperparams[CHOOSE_ONE_OR_MANY_STEPS_OF_CHOICE_HYPERPARAM].keys():
                self.hyperparams[CHOOSE_ONE_OR_MANY_STEPS_OF_CHOICE_HYPERPARAM][key] = False
                warning.warn('You have not set any choice hyperparameter for {0} in {1}. Setting choice to False by default.'.format(key, self.name))

        self.nullify_steps_that_are_not_chosen()

        return self

    def nullify_steps_that_are_not_chosen(self):
        """
        Nullify steps that are not chosen using :class:`Nullify` step wrapper.
        """
        for step_name, is_chosen in self.hyperparams[CHOOSE_ONE_OR_MANY_STEPS_OF_CHOICE_HYPERPARAM].items():
            if not is_chosen:
                self[step_name] = Nullify(self[step_name])
            elif isinstance(self[step_name], Nullify):
                self[step_name] = self[step_name].get_step()
        self._refresh_steps()
