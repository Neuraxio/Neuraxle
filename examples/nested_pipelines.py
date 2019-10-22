import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from neuraxle.base import Identity
from neuraxle.pipeline import Pipeline
from neuraxle.steps.sklearn import SKLearnWrapper


def main():
    # Create and fit the pipeline:
    pipeline = Pipeline([
        SKLearnWrapper(StandardScaler()),
        Identity(),
        Pipeline([
            Identity(),  # Note: an Identity step is a step that does nothing.
            Identity(),  # We use it here for demonstration purposes.
            Pipeline([
                Identity(),
                Identity(),
                SKLearnWrapper(PCA(n_components=2))
            ])
        ])
    ])
    X = np.random.randint(5, size=(2, 4))

    pipeline, X_t = pipeline.fit_transform(X)

    # Get the components:
    pca_components = pipeline["Pipeline"]["Pipeline"][-1].get_wrapped_sklearn_predictor().components_

    return pca_components
