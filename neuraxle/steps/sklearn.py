from sklearn.linear_model import Ridge

from neuraxle.base import BaseStep
from neuraxle.steps.numpy import NumpyTranspose
from neuraxle.union import ModelStacking


class SKLearnWrapper(BaseStep):
    def __init__(self, wrapped_sklearn_predictor, return_all_sklearn_default_params_on_get=False):
        # TODO: throw if not initialized (e.g.: class type passed)
        self.wrapped_sklearn_predictor = wrapped_sklearn_predictor
        params = wrapped_sklearn_predictor.get_params()
        super().__init__(params, params)
        self.return_all_sklearn_default_params_on_get = return_all_sklearn_default_params_on_get

    def fit(self, data_inputs, expected_outputs=None):
        self.wrapped_sklearn_predictor.fit(data_inputs, expected_outputs)
        return self

    def transform(self, data_inputs):
        if hasattr(self.wrapped_sklearn_predictor, 'predict'):
            return self.wrapped_sklearn_predictor.predict(data_inputs)
        return self.wrapped_sklearn_predictor.transform(data_inputs)

    def set_hyperparams(self, flat_hyperparams: dict):
        super().set_hyperparams(flat_hyperparams)
        self.wrapped_sklearn_predictor.set_params(**flat_hyperparams)

    def get_hyperparams(self):
        if self.return_all_sklearn_default_params_on_get:
            return self.wrapped_sklearn_predictor.get_params()
        else:
            return super(SKLearnWrapper, self).get_hyperparams()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        type_ = self.__class__
        module = type_.__module__
        qualname = type_.__qualname__
        wrappedname = str(self.wrapped_sklearn_predictor.__class__.__name__)
        return "<{}.{}({}(...)) object {}>".format(module, qualname, wrappedname, hex(id(self)))


class RidgeModelStacking(ModelStacking):
    def __init__(self, brothers):
        super().__init__(brothers, SKLearnWrapper(Ridge()), NumpyTranspose())

