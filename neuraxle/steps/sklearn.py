"""
Pipeline Steps Based on Scikit-Learn
=====================================
Those steps works with scikit-learn (sklearn) transformers and estimators.

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

from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge

from neuraxle.base import BaseStep
from neuraxle.hyperparams.distributions import LogUniform, Boolean
from neuraxle.hyperparams.space import HyperparameterSpace, HyperparameterSamples
from neuraxle.steps.numpy import NumpyTranspose
from neuraxle.union import ModelStacking


class SKLearnWrapper(BaseStep):
    def __init__(
            self,
            wrapped_sklearn_predictor,
            hyperparams_space: HyperparameterSpace = None,
            return_all_sklearn_default_params_on_get=False
    ):
        if not isinstance(wrapped_sklearn_predictor, BaseEstimator):
            raise ValueError("The wrapped_sklearn_predictor must be an instance of scikit-learn's BaseEstimator.")
        self.wrapped_sklearn_predictor = wrapped_sklearn_predictor
        params: HyperparameterSamples = wrapped_sklearn_predictor.get_params()
        super().__init__(hyperparams=params, hyperparams_space=hyperparams_space)
        self.return_all_sklearn_default_params_on_get = return_all_sklearn_default_params_on_get
        self.name += "_" + wrapped_sklearn_predictor.__class__.__name__

    def fit(self, data_inputs, expected_outputs=None) -> 'SKLearnWrapper':
        self.wrapped_sklearn_predictor = self.wrapped_sklearn_predictor.fit(data_inputs, expected_outputs)
        return self

    def transform(self, data_inputs):
        if hasattr(self.wrapped_sklearn_predictor, 'predict'):
            return self.wrapped_sklearn_predictor.predict(data_inputs)
        return self.wrapped_sklearn_predictor.transform(data_inputs)

    def set_hyperparams(self, flat_hyperparams: dict) -> BaseStep:
        super().set_hyperparams(flat_hyperparams)
        self.wrapped_sklearn_predictor.set_params(**flat_hyperparams)
        return self

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
        super().__init__(
            brothers,
            SKLearnWrapper(
                Ridge(),
                HyperparameterSpace({"alpha": LogUniform(0.1, 10.0), "fit_intercept": Boolean()})
            ),
            joiner=NumpyTranspose()
        )
