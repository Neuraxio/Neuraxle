"""
Tests for features steps
========================================

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
import pytest

from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.steps.features import Cheap3DTo2DTransformer, FFTPeakBinWithValue
import numpy as np


def test_fft_peak_bin_with_values():
    data_inputs = np.random.random((4, 5, 2))
    step = FFTPeakBinWithValue()

    outputs = step.transform(data_inputs)

    assert outputs.shape == (4, 4)


@pytest.mark.parametrize("hyperparams, expected_feature_count", [
    (HyperparameterSamples({
        'FFT__enabled': True,
        'NumpyMean__enabled': True,
        'NumpyMedian__enabled': True,
        'NumpyMin__enabled': True,
        'NumpyMax__enabled': True
    }), 18),
    (HyperparameterSamples({
        'FFT__enabled': False,
        'NumpyMean__enabled': True,
        'NumpyMedian__enabled': True,
        'NumpyMin__enabled': True,
        'NumpyMax__enabled': True
    }), 8),
    (HyperparameterSamples({
        'FFT__enabled': True,
        'NumpyMean__enabled': False,
        'NumpyMedian__enabled': True,
        'NumpyMin__enabled': True,
        'NumpyMax__enabled': True
    }), 16),
    (HyperparameterSamples({
        'FFT__enabled': True,
        'NumpyMean__enabled': True,
        'NumpyMedian__enabled': False,
        'NumpyMin__enabled': True,
        'NumpyMax__enabled': True
    }), 16),
    (HyperparameterSamples({
        'FFT__enabled': True,
        'NumpyMean__enabled': True,
        'NumpyMedian__enabled': True,
        'NumpyMin__enabled': False,
        'NumpyMax__enabled': True
    }), 16),
    (HyperparameterSamples({
        'FFT__enabled': True,
        'NumpyMean__enabled': True,
        'NumpyMedian__enabled': True,
        'NumpyMin__enabled': True,
        'NumpyMax__enabled': False
    }), 16)
])
def test_cheap_3D_to_2D_transformer(hyperparams: HyperparameterSamples, expected_feature_count: int):
    step = Cheap3DTo2DTransformer()
    step.set_hyperparams(hyperparams=hyperparams)
    data_inputs = np.random.random((7, 5, 2))

    outputs = step.transform(data_inputs)

    assert outputs.shape == (4, expected_feature_count)
