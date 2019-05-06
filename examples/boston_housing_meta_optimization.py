"""
Boston Housing Regression with Meta Optimization
==========================
This is an automatic machine learning example. It is more sophisticated than the other simple regression example.
Not only a pipeline is defined, but also an hyperparameter space is defined for the pipeline. Then, a random search is
performed to find the best possible combination of hyperparameters by sampling randomly in the hyperparameter space.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import NumpyTranspose, NumpyShapePrinter
from neuraxle.steps.sklearn import SKLearnWrapper
from neuraxle.union import AddFeatures, FeatureUnion, Identity

boston = load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

p = Pipeline([
    NumpyShapePrinter(),
    AddFeatures([
        SKLearnWrapper(PCA(n_components=2)),
        SKLearnWrapper(FastICA(n_components=2)),
    ]),
    NumpyShapePrinter(),
    FeatureUnion([
        SKLearnWrapper(GradientBoostingRegressor()),
        SKLearnWrapper(GradientBoostingRegressor(n_estimators=500)),
        SKLearnWrapper(GradientBoostingRegressor(max_depth=5)),
        SKLearnWrapper(KMeans()),
    ], joiner=NumpyTranspose()),
    NumpyShapePrinter(),
    SKLearnWrapper(Ridge()),
    NumpyShapePrinter(),
    Identity()  # This step simply does nothing.
])

print("Fitting on train:")
p.fit(X_train, y_train)
# p.meta_fit(X_train, y_train)
print("")

print("Transforming test:")
y_test_predicted = p.transform(X_test)
print("")

print("Evaluating transformed test:")
score = r2_score(y_test_predicted, y_test)
print('R2 regression score:', score)
