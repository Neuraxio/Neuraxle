"""
Feature Steps
==========================================================
You can find here steps that featurize your data.

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
from neuraxle.base import NonFittableMixin, BaseStep
from neuraxle.pipeline import Pipeline
from neuraxle.steps.flow import ChooseOneOrManyStepsOf
from neuraxle.steps.numpy import NumpyFFT, NumpyAbs, NumpyFlattenDatum, NumpyConcatenateInnerFeatures, NumpyMean, \
    NumpyMedian, NumpyMin, NumpyMax
from neuraxle.union import FeatureUnion
import numpy as np


class FFTPeakBinWithValue(NonFittableMixin, BaseStep):
    """
    Compute peak fft bins (int), and their magnitudes' value (float), to concatenate them.

    .. seealso::
        :class:`~neuraxle.base.BaseStep`,
        :class:`~neuraxle.base.NonFittableMixin`
    """

    def transform(self, data_inputs):
        """
        Will compute peak fft bins (int), and their magnitudes' value (float), to concatenate them.

        :param data_inputs: real magnitudes of an fft. It could be of shape [batch_size, bins, features].
        :return: Two arrays without bins concatenated on feature axis. Shape: [batch_size, 2 * features]
        """
        time_bins_axis = -2
        peak_bin = np.argmax(data_inputs, axis=time_bins_axis)
        peak_bin_val = np.max(data_inputs, axis=time_bins_axis)

        # Notice that here another FeatureUnion could have been used with a joiner:
        transformed = np.concatenate([peak_bin, peak_bin_val], axis=-1)

        return transformed


class Cheap3DTo2DTransformer(ChooseOneOrManyStepsOf):
    """
    Prebuild class to featurize 3D data for classification.

    You can enable, or disable features using hyperparams :

    .. code-block:: python

        step = Cheap3DTo2DTransformer().set_hyperparams(hyperparams={
            'FFT__enabled': True,
            'NumpyMean__enabled': True,
            'NumpyMedian__enabled': True,
            'NumpyMin__enabled': True,
            'NumpyMax__enabled': True
        })

    .. seealso::
        :class:`~neuraxle.steps.flow.ChooseOneOrManyStepsOf`,
        :class:`NumpyFFT`,
        :class:`NumpyAbs`,
        :class:`NumpyFlattenDatum`,
        :class:`FFTPeakBinWithValue`,
        :class:`NumpyConcatenateInnerFeatures`,
        :class:`NumpyMean`,
        :class:`NumpyMedian`,
        :class:`NumpyMin`,
        :class:`NumpyMax`
    """

    def __init__(self):
        super().__init__([
            Pipeline([
                NumpyFFT(),
                NumpyAbs(),
                FeatureUnion([
                    NumpyFlattenDatum(),  # Reshape from 3D to flat 2D: flattening data except on batch size
                    FFTPeakBinWithValue()  # Extract 2D features from the 3D FFT bins
                ], joiner=NumpyConcatenateInnerFeatures())
            ]).set_name('FFT'),
            NumpyMean(),
            NumpyMedian(),
            NumpyMin(),
            NumpyMax()
        ])
