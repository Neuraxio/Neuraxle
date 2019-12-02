"""
Data Steps
====================================
You can find here steps that take action on data.

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
import random
from typing import Iterable

from neuraxle.base import BaseStep, MetaStepMixin, NonFittableMixin
from neuraxle.steps.output_handlers import InputAndOutputTransformerMixin


class DataShuffler(NonFittableMixin, InputAndOutputTransformerMixin, BaseStep):
    """
    Data Shuffling step that shuffles data inputs, and expected_outputs at the same time.

    .. code-block:: python

        p = Pipeline([
            TrainOnlyWrapper(DataShuffler(seed=42, increment_seed_after_each_fit=True)),
            TrainOnlyWrapper(EpochRepeater(ForecastingPipeline(), epochs=EPOCHS))
            TestOnlyWrapper(ForecastingPipeline())
        ])

    .. warning::
        You probably always want to wrap this step by a :class:`TrainOnlyWrapper`

    .. seealso::
        :class:`EpochRepeater`,
        :class:`TrainOnlyWrapper`,
        :class:`InputAndOutputTransformerMixin`,
        :class:`BaseStep`
    """

    def __init__(self, seed, increment_seed_after_each_fit=True):
        InputAndOutputTransformerMixin.__init__(self)
        BaseStep.__init__(self)
        self.seed = seed
        self.increment_seed_after_each_fit = increment_seed_after_each_fit

    def transform(self, data_inputs):
        """
        Shuffle data inputs, and expected outputs.

        :param data_inputs: (data inputs, expected outputs) tuple to shuffle
        :return:
        """
        if self.increment_seed_after_each_fit:
            self.seed += 1

        di, eo = data_inputs
        data = list(zip(di, eo))
        random.Random(self.seed).shuffle(data)

        data_inputs_shuffled, expected_outputs_shuffled = list(zip(*data))

        return list(data_inputs_shuffled), list(expected_outputs_shuffled)


class EpochRepeater(MetaStepMixin, BaseStep):
    """
    Repeat wrapped step fit, or transform for the number of epochs passed in the constructor.

    .. code-block:: python

        p = Pipeline([
            TrainOnlyWrapper(DataShuffler(seed=42, increment_seed_after_each_fit=True)),
            TrainOnlyWrapper(EpochRepeater(ForecastingPipeline(), epochs=EPOCHS))
            TestOnlyWrapper(ForecastingPipeline())
        ])

    .. seealso::
        :class:`DataShuffler`,
        :class:`MetaStepMixin`,
        :class:`TrainOnlyWrapper`,
        :class:`TestOnlyWrapper`,
        :class:`BaseStep`
    """

    def __init__(self, wrapped, epochs, fit_only=True):
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        self.fit_only = fit_only
        self.epochs = epochs

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BaseStep', Iterable):
        """
        Fit transform wrapped step self.epochs times.

        :param data_inputs: data inputs to fit on
        :param expected_outputs: expected_outputs to fit on
        :return: fitted self
        """
        for _ in range(self.epochs -1):
            self.wrapped, data_inputs = self.wrapped.fit(data_inputs, expected_outputs)
        return self, self.wrapped.fit_transform(data_inputs, expected_outputs)

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        """
        Fit wrapped step self.epochs times.

        :param data_inputs: data inputs to fit on
        :param expected_outputs: expected_outputs to fit on
        :return: fitted self
        """
        for _ in range(self.epochs):
            self.wrapped = self.wrapped.fit(data_inputs, expected_outputs)
        return self

    def transform(self, data_inputs):
        """
        Transform wrapped step self.epochs times if self.fit_only is False otherwise transform wrapped step only one time.

        :param data_inputs: data inputs to transform
        :return: transformed data inputs
        """
        if not self.fit_only:
            for _ in range(self.epochs):
                data_inputs = self.wrapped.transform(data_inputs)
        else:
            data_inputs = self.wrapped.transform(data_inputs)
        return data_inputs
