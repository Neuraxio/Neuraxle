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


def test_saver_parallel_transform_should_parallelize():
    boston = load_boston()
    X, y = shuffle(boston.data, boston.target, random_state=13)
    X = X.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)


    p = Pipeline([
        SaverParallelTransform(
            SKLearnWrapper(PCA(n_components=2), HyperparameterSpace({"n_components": RandInt(1, 3)}))
        )
    ])

    outputs_train = p.transform(X_train)
