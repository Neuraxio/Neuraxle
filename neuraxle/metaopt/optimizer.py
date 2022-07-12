"""
Neuraxle's Hyperparameter Optimizer Base Classes
====================================================

Not all hyperparameter optimizers are there, but the base can be found here.


.. seealso::
    :class:`~neuraxle.metaopt.hyperopt.tpe.TreeParzenEstimator`,


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


"""

from abc import ABC, abstractmethod

from neuraxle.hyperparams.space import HyperparameterSamples


class BaseHyperparameterOptimizer(ABC):

    @abstractmethod
    def find_next_best_hyperparams(self, round_scope) -> HyperparameterSamples:
        """
        Find the next best hyperparams using previous trials, that is the
        whole :class:`neuraxle.metaopt.data.aggregate.Round`.

        :param round: a :class:`neuraxle.metaopt.data.aggregate.Round`
        :return: next hyperparameter samples to train on
        """
        raise NotImplementedError()


class HyperparameterSamplerStub(BaseHyperparameterOptimizer):

    def __init__(self, preconfigured_hp_samples: HyperparameterSamples):
        self.preconfigured_hp_samples = preconfigured_hp_samples

    def find_next_best_hyperparams(self, round_scope) -> HyperparameterSamples:
        return self.preconfigured_hp_samples
