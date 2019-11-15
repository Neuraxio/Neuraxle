"""
Scikit-learn metaoptimizers
=====================================
E.g.: for use with RandomizedSearchCV.

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

from neuraxle.base import MetaStepMixin, BaseStep


class MetaSKLearnWrapper(MetaStepMixin, BaseStep):

    def __init__(self, wrapped):
        """
        Wrap a scikit-learn MetaEstimatorMixin for usage in Neuraxle. 
        This class is similar to the SKLearnWrapper class of Neuraxle that can wrap a scikit-learn BaseEstimator. 
        
        :param wrapped: a scikit-learn object of type "MetaEstimatorMixin". 
        """
        MetaStepMixin.__init__(self)
        BaseStep.__init__(self)
        self.wrapped_sklearn_metaestimator = wrapped  # TODO: use self.set_step of the MetaStepMixin instead?
        # sklearn.model_selection.RandomizedSearchCV

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        self.wrapped_sklearn_metaestimator = self.wrapped_sklearn_metaestimator.fit(data_inputs, expected_outputs)
        return self

    def transform(self, data_inputs):
        return self.wrapped_sklearn_metaestimator.transform(data_inputs)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        type_ = self.__class__
        module = type_.__module__
        qualname = type_.__qualname__
        wrappedname = str(self.wrapped_sklearn_metaestimator.__class__.__name__)
        return "<{}.{}({}(...)) object {}>".format(module, qualname, wrappedname, hex(id(self)))
