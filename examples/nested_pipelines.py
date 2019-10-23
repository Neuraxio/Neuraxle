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
    np.random.seed(42)
    X = np.random.randint(5, size=(100, 5))

    pipeline, X_t = pipeline.fit_transform(X)

    # Get the components:
    pca_components = pipeline["Pipeline"]["Pipeline"][-1].get_wrapped_sklearn_predictor().components_

    assert pca_components.shape == (2, 5)

    # https://stackoverflow.com/questions/28822756/getting-model-attributes-from-scikit-learn-pipeline/58359509#58359509

if __name__ == "__main__":
    main()
