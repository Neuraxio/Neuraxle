"""
Inverse Transforms in Neuraxle: How to Reverse a Prediction
============================================================

This demonstrates how to make a prediction, and then to undo the prediction to get back the original inputs or an
estimate of the original inputs. Not every pipeline steps have an inverse transform method, because not every operation
is reversible.

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

import numpy as np

from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import MultiplyByN


def main():
    p = Pipeline([MultiplyByN(multiply_by=2)])

    data_inputs = np.array([1, 2])
    generated_outputs = p.transform(data_inputs)
    regenerated_inputs = reversed(p).transform(generated_outputs)

    assert np.array_equal(regenerated_inputs, data_inputs)
    assert np.array_equal(generated_outputs, 2 * data_inputs)


if __name__ == "__main__":
    main()
