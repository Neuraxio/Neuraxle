import numpy as np
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from neuraxle.distributed.clustering import ClusteringWrapper
from neuraxle.hyperparams.distributions import RandInt
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import NumpyConcatenateOuterBatch
from neuraxle.steps.sklearn import SKLearnWrapper


def test_clustering_wrapper(tmpdir):
    boston = load_boston()
    X, y = shuffle(boston.data, boston.target, random_state=13)
    X = X.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    sklearn_pca = Pipeline([SKLearnWrapper(PCA(n_components=2), HyperparameterSpace({"n_components": RandInt(1, 3)}))]).fit(X_train, y_train)
    sklearn_pca = sklearn_pca.fit(X_train, y_train)

    p = Pipeline([
        ClusteringWrapper(
            sklearn_pca,
            hosts=['http://127.0.0.1:5000/pipeline'],
            joiner=NumpyConcatenateOuterBatch(),
            n_jobs=10,
            batch_size=10,
            n_workers_per_step=1
        )
    ], cache_folder=tmpdir)

    outputs = p.transform(X_train)

