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
import functools
import inspect
from typing import Any, Tuple

from neuraxle.base import BaseStep
from neuraxle.base import ExecutionContext as CX
from neuraxle.hyperparams.distributions import Boolean, LogUniform
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace, RecursiveDict)
from neuraxle.steps.numpy import NumpyTranspose
from neuraxle.union import ModelStacking

from sklearn.base import BaseEstimator
from sklearn.ensemble import BaseEnsemble
from sklearn.linear_model import Ridge


class SKLearnWrapper(BaseStep):
    def __init__(
            self,
            wrapped_sklearn_predictor,
            hyperparams_space: HyperparameterSpace = None,
            return_all_sklearn_default_params_on_get: bool = False,
            use_partial_fit: bool = False,
            use_predict_proba: bool = False,
            partial_fit_kwargs: dict = None
    ):
        if not isinstance(wrapped_sklearn_predictor, BaseEstimator):
            raise ValueError("The wrapped_sklearn_predictor must be an instance of scikit-learn's BaseEstimator.")
        self.wrapped_sklearn_predictor = wrapped_sklearn_predictor
        self.is_ensemble = isinstance(wrapped_sklearn_predictor, BaseEnsemble)
        params: dict = wrapped_sklearn_predictor.get_params()
        self._delete_base_estimator_from_dict(params)
        BaseStep.__init__(self, hyperparams=params, hyperparams_space=hyperparams_space)
        self.return_all_sklearn_default_params_on_get = return_all_sklearn_default_params_on_get
        self.name += "_" + wrapped_sklearn_predictor.__class__.__name__
        self.use_partial_fit: bool = use_partial_fit
        if self.use_partial_fit:
            if partial_fit_kwargs is None:
                partial_fit_kwargs = {}
            self.partial_fit_kwargs = partial_fit_kwargs
        self.use_predict_proba: bool = use_predict_proba

    def _setup(self, context: CX = None) -> 'SKLearnWrapper':
        BaseStep._setup(self, context)
        if self.use_partial_fit:
            self.wrapped_sklearn_predictor.fit = functools.partial(self.wrapped_sklearn_predictor.partial_fit,
                                                                   **self.partial_fit_kwargs)
        return self

    def _delete_base_estimator_from_dict(self, params):
        """
        Sklearn BaseEnsemble models contain other models as parameter; those can't be json encoded. We retrieve the parameters of theses sub-models on a .get_params(deep=True) call, we simply need to delete them from the parameter dictionary to avoid errors when saving/loading hyperparameters.
        """
        for name in list(params.keys()):
            if isinstance(params[name], BaseEstimator):
                del params[name]

    def fit_transform(self, data_inputs, expected_outputs=None) -> Tuple['BaseStep', Any]:
        if hasattr(self.wrapped_sklearn_predictor, 'fit_transform'):
            if expected_outputs is None or len(inspect.getfullargspec(self.wrapped_sklearn_predictor.fit).args) < 3:
                out = self._sklearn_fit_transform_without_expected_outputs(data_inputs)
            else:
                out = self._sklearn_fit_transform_with_expected_outputs(data_inputs, expected_outputs)
            return self, out

        self.fit(data_inputs, expected_outputs)

        return self, self.transform(data_inputs)

    def _sklearn_fit_transform_with_expected_outputs(self, data_inputs, expected_outputs):
        self.wrapped_sklearn_predictor = self.wrapped_sklearn_predictor.fit(data_inputs, expected_outputs)
        return self.transform(data_inputs)

    def _sklearn_fit_transform_without_expected_outputs(self, data_inputs):
        if self.use_partial_fit:
            self.wrapped_sklearn_predictor = self.wrapped_sklearn_predictor.fit(data_inputs)
            out = self.transform(data_inputs)
        else:
            out = self.wrapped_sklearn_predictor.fit_transform(data_inputs)
        return out

    def fit(self, data_inputs, expected_outputs=None) -> 'SKLearnWrapper':
        if expected_outputs is None or len(inspect.getfullargspec(self.wrapped_sklearn_predictor.fit).args) < 3:
            self._sklearn_fit_without_expected_outputs(data_inputs)
        else:
            self._sklearn_fit_with_expected_outputs(data_inputs, expected_outputs)
        return self

    def _sklearn_fit_with_expected_outputs(self, data_inputs, expected_outputs):
        self.wrapped_sklearn_predictor = self.wrapped_sklearn_predictor.fit(data_inputs, expected_outputs)

    def _sklearn_fit_without_expected_outputs(self, data_inputs):
        self.wrapped_sklearn_predictor = self.wrapped_sklearn_predictor.fit(data_inputs)

    def transform(self, data_inputs):
        if self.use_predict_proba and hasattr(self.wrapped_sklearn_predictor, 'predict_proba'):
            return self.wrapped_sklearn_predictor.predict_proba(data_inputs)
        elif hasattr(self.wrapped_sklearn_predictor, 'predict'):
            return self.wrapped_sklearn_predictor.predict(data_inputs)
        return self.wrapped_sklearn_predictor.transform(data_inputs)

    def _set_hyperparams(self, hyperparams: HyperparameterSamples) -> BaseStep:
        """
        Set hyperparams for base step, and the wrapped sklearn_predictor.

        :param hyperparams:
        :return: self
        """
        # set the step hyperparams, and set the wrapped sklearn predictor params
        BaseStep._set_hyperparams(self, hyperparams)
        self.wrapped_sklearn_predictor.set_params(
            **hyperparams.with_separator(RecursiveDict.DEFAULT_SEPARATOR).to_flat_dict()
        )

        return self.hyperparams

    def _update_hyperparams(self, hyperparams: HyperparameterSamples) -> BaseStep:
        """
        Update hyperparams for base step, and the wrapped sklearn_predictor.

        :param hyperparams:
        :return: self
        """
        # update the step hyperparams, and update the wrapped sklearn predictor params
        BaseStep._update_hyperparams(self, hyperparams)
        self.wrapped_sklearn_predictor.set_params(
            **self.hyperparams.with_separator(RecursiveDict.DEFAULT_SEPARATOR).to_flat_dict()
        )

        return self.hyperparams

    def _get_hyperparams(self):
        if self.return_all_sklearn_default_params_on_get:
            hp = self.wrapped_sklearn_predictor.get_params()
            self._delete_base_estimator_from_dict(hp)
            return HyperparameterSamples(hp)
        else:
            return BaseStep._get_hyperparams(self)

    def get_wrapped_sklearn_predictor(self):
        return self.wrapped_sklearn_predictor

    def _repr(self, level=0, verbose=False) -> str:
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
