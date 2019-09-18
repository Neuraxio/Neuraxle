from abc import abstractmethod
from typing import Any, Tuple

from neuraxle.base import MetaStepMixin, BaseStep, DataContainer


class OutputTransformerMixin:
    """
    Mixin to be able to modify expected_outputs inside a step
    """

    @abstractmethod
    def transform_input_output(self, data_inputs, expected_outputs=None) -> Tuple[Any, Any]:
        """
        Transform data inputs, and expected outputs at the same time

        :param data_inputs:
        :param expected_outputs:
        :return: tuple(data_inputs, expected_outputs)
        """
        raise NotImplementedError()


class OutputTransformerWrapper(MetaStepMixin, BaseStep):
    """
    Output transformer wrapper wraps a step that inherits OutputTransformerMixin,
    and updates the data inputs, and expected outputs for each transform.
    """
    def __init__(self, wrapped: BaseStep):
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)

    def handle_transform(self, data_container: DataContainer) -> DataContainer:
        """
        Handle transform by updating the data inputs, and expected outputs inside the data container.

        :param data_container:
        :return:
        """
        new_data_inputs, new_expected_outputs = self.wrapped.transform_input_output(
            data_inputs=data_container.data_inputs, expected_outputs=data_container.expected_outputs
        )
        data_container.set_data_inputs(new_data_inputs)
        data_container.set_expected_outputs(new_expected_outputs)

        current_ids = self.hasher.hash(data_container.current_ids, self.hyperparams, data_container.data_inputs)
        data_container.set_current_ids(current_ids)

        return data_container

    def handle_fit_transform(self, data_container: DataContainer) -> ('BaseStep', DataContainer):
        """
        Handle transform by fitting the step,
        and updating the data inputs, and expected outputs inside the data container.

        :param data_container:
        :return:
        """
        new_self = self.wrapped.fit(data_container.data_inputs, data_container.expected_outputs)

        data_container = self.handle_transform(data_container)

        return new_self, data_container
