import numpy as np
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from neuraxle.hyperparams.distributions import RandInt
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.parallel import SaverParallelTransform
from neuraxle.pipeline import Pipeline
from neuraxle.steps.sklearn import SKLearnWrapper


def test_saver_parallel_transform(tmpdir):
    boston = load_boston()
    X, y = shuffle(boston.data, boston.target, random_state=13)
    X = X.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    sk_learn_pca = SKLearnWrapper(PCA(n_components=2), HyperparameterSpace({"n_components": RandInt(1, 3)}))
    sk_learn_pca = sk_learn_pca.fit(X_train, y_train)
    expected_outputs = sk_learn_pca.transform(X_train)

    p = Pipeline([
        SaverParallelTransform(
            sk_learn_pca,
            n_jobs=10,
            batch_size=10
        )
    ], cache_folder=tmpdir)

    actual_outputs = p.transform(X_train)

    assert np.array_equal(expected_outputs, actual_outputs)
