"""
Create Nested Pipelines in Neuraxle
================================================

You can create pipelines within pipelines using the composition design pattern.

This demonstrates how to create pipelines within pipelines, and how to access the steps and their
attributes in the nested pipelines.

For more info, see the `thread here <https://stackoverflow.com/questions/28822756/getting-model-attributes-from-scikit-learn-pipeline/58359509#58359509>`__.

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from neuraxle.base import Identity
from neuraxle.pipeline import Pipeline


def main():
    np.random.seed(42)
    X = np.random.randint(5, size=(100, 5))

    # Create and fit the pipeline:
    pipeline = Pipeline([
        StandardScaler(),
        Identity(),
        Pipeline([
            Identity(),
            Identity(),  # Note: an Identity step is a step that does nothing.
            Identity(),  # We use it here for demonstration purposes.
            Pipeline([
                Identity(),
                PCA(n_components=2)
            ])
        ])
    ])
    pipeline, X_t = pipeline.fit_transform(X)

    # Get the components:
    pca_components = pipeline["Pipeline"]["Pipeline"][-1].get_wrapped_sklearn_predictor().components_
    assert pca_components.shape == (2, 5)

    # Discussion:
    # https://stackoverflow.com/questions/28822756/getting-model-attributes-from-scikit-learn-pipeline/58359509#58359509


if __name__ == "__main__":
    main()
