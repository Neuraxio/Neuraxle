import numpy as np

from neuraxle.base import NonFittableMixin, BaseStep


# TODO: check order of "NonFittableMixin, BaseStep"


class NumpyFlattenDatum(NonFittableMixin, BaseStep):
    def __init__(self):
        super().__init__()

    def transform(self, data_inputs):
        return data_inputs.reshape(data_inputs.shape[0], -1)

    def transform_one(self, data_inputs):
        return data_inputs.flatten()


class NumpyConcatenateInnerFeatures(NonFittableMixin, BaseStep):
    def __init__(self):
        super().__init__()

    def transform(self, data_inputs):
        return np.concatenate(data_inputs, axis=-1)

    def transform_one(self, data_inputs):
        return np.concatenate(data_inputs, axis=-1)


class NumpyTranspose(NonFittableMixin, BaseStep):
    def transform(self, data_inputs):
        return np.array(data_inputs).transpose()

    def transform_one(self, data_input):
        raise BrokenPipeError("Cannot simply transform_one here.")


class NumpyShapePrinter(NonFittableMixin, BaseStep):
    def transform(self, data_inputs):
        print(self.__class__.__name__ + ":", data_inputs.shape)
        return data_inputs

    def transform_one(self, data_input):
        raise BrokenPipeError("You should use `.transform` instead for a shape printer.")
