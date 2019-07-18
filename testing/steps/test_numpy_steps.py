"""
Tests for NumPy Steps
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

from neuraxle.steps.numpy import *


def test_flatten_datum():
    flat = NumpyFlattenDatum()
    data = np.random.random((10, 4, 5, 2))  # 4D array (could be ND with N>=2).
    expected_data = np.copy(data).reshape(10, 4 * 5 * 2)  # 2D array.

    flat, received_data = flat.fit_transform(data)

    assert (received_data == expected_data).all()


def test_concat_features():
    concat = NumpyConcatenateInnerFeatures()
    # ND arrays
    data1 = np.random.random((10, 4, 5, 2))
    data2 = np.random.random((10, 4, 5, 10))
    expected_all_data = np.concatenate([data1, data2], axis=-1)

    concat, received_all_data = concat.fit_transform([data1, data2])

    assert tuple(received_all_data.shape) == tuple(expected_all_data.shape)
    assert (received_all_data == expected_all_data).all()


def test_numpy_transpose():
    tr = NumpyTranspose()
    data = np.random.random((10, 7))
    expected_data = np.copy(data).transpose()

    tr, received_data = tr.fit_transform(data)

    assert (received_data == expected_data).all()


def test_numpy_shape_printer():
    pr = NumpyShapePrinter()
    pr.fit_transform(np.ones((10, 11)))
