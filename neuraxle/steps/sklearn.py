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

..
    Thanks to Umaneo Technologies Inc. for their contributions to this Machine Learning
    project, visit https://www.umaneo.com/ for more information on Umaneo Technologies Inc.

"""
from typing import Any

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
        params: HyperparameterSamples = HyperparameterSamples(wrapped_sklearn_predictor.get_params())
        BaseStep.__init__(self, hyperparams=params, hyperparams_space=hyperparams_space)
        self.return_all_sklearn_default_params_on_get = return_all_sklearn_default_params_on_get
        self.name += "_" + wrapped_sklearn_predictor.__class__.__name__

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BaseStep', Any):

        if hasattr(self.wrapped_sklearn_predictor, 'fit_transform'):
            out = self.wrapped_sklearn_predictor.fit_transform(data_inputs, expected_outputs)
            return self, out

        self.wrapped_sklearn_predictor = self.wrapped_sklearn_predictor.fit(data_inputs, expected_outputs)
        if hasattr(self.wrapped_sklearn_predictor, 'predict'):
            return self, self.wrapped_sklearn_predictor.predict(data_inputs)

        return self, self.wrapped_sklearn_predictor.transform(data_inputs)

    def fit(self, data_inputs, expected_outputs=None) -> 'SKLearnWrapper':
        self.wrapped_sklearn_predictor = self.wrapped_sklearn_predictor.fit(data_inputs, expected_outputs)
        return self

    def transform(self, data_inputs):
        if hasattr(self.wrapped_sklearn_predictor, 'predict'):
            return self.wrapped_sklearn_predictor.predict(data_inputs)
        return self.wrapped_sklearn_predictor.transform(data_inputs)

    def set_hyperparams(self, flat_hyperparams: HyperparameterSamples) -> BaseStep:
        BaseStep.set_hyperparams(self, flat_hyperparams)
        self.wrapped_sklearn_predictor.set_params(**HyperparameterSamples(flat_hyperparams).to_flat_as_dict_primitive())
        return self

    def get_hyperparams(self):
        if self.return_all_sklearn_default_params_on_get:
            return HyperparameterSamples(self.wrapped_sklearn_predictor.get_params()).to_flat()
        else:
            return BaseStep.get_hyperparams(self)

    def get_wrapped_sklearn_predictor(self):
        return self.wrapped_sklearn_predictor

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
        ModelStacking.__init__(
            self,
            brothers,
            SKLearnWrapper(
                Ridge(),
                HyperparameterSpace({"alpha": LogUniform(0.1, 10.0), "fit_intercept": Boolean()})
            ),
            joiner=NumpyTranspose()
        )
