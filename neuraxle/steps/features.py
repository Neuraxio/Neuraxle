"""
Featurization Steps
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
from neuraxle.pipeline import Pipeline
from neuraxle.steps.flow import ChooseOneOrManyStepsOf
from neuraxle.steps.numpy import NumpyFFT, NumpyAbs, NumpyFlattenDatum, NumpyConcatenateInnerFeatures, NumpyMean, \
    NumpyMedian, NumpyMin, NumpyMax, NumpyArgMax
from neuraxle.union import FeatureUnion


class FFTPeakBinWithValue(FeatureUnion):
    """
    Compute peak fft bins (int), and their magnitudes' value (float), to concatenate them.
    This is intended to be used only after a NumpyFFT absolute step.

    .. seealso::
        :class:`~neuraxle.base.BaseStep`,
        :class:`~neuraxle.base.NonFittableMixin`,
        :class:`~neuraxle.steps.numpy.NumpyFFT`,
        :class:`Cheap3DTo2DTransformer`
    """
    def __init__(self):
        super().__init__([
            NumpyArgMax(axis=-2),
            NumpyMax(axis=-2)
        ], joiner=NumpyConcatenateInnerFeatures())


class Cheap3DTo2DTransformer(ChooseOneOrManyStepsOf):
    """
    Prebuild class to featurize 3D data into 2D data for simple classification or regression, for instance.

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
        :class:`~neuraxle.steps.numpy.NumpyFFT`,
        :class:`~neuraxle.steps.numpy.NumpyAbs`,
        :class:`~neuraxle.steps.numpy.NumpyFlattenDatum`,
        :class:`FFTPeakBinWithValue`,
        :class:`~neuraxle.steps.numpy.NumpyConcatenateInnerFeatures`,
        :class:`~neuraxle.steps.numpy.NumpyMean`,
        :class:`~neuraxle.steps.numpy.NumpyMedian`,
        :class:`~neuraxle.steps.numpy.NumpyMin`,
        :class:`~neuraxle.steps.numpy.NumpyMax`
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
