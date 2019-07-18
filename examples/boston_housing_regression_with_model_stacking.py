"""
Boston Housing Regression
==========================

This example solves a regression problem using a pipeline with the following steps:

 - Feature augmentation with PCA and Fast ICA,
 - A Pre-regression using an ensemble containing gradient boosted, and a KMeans clustering for even more features in the stacking,
 - The model stacking using a ridge regression.

This example also prints the shapes of the objects between the pipeline elements.

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

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import NumpyShapePrinter
from neuraxle.steps.sklearn import SKLearnWrapper, RidgeModelStacking
from neuraxle.union import AddFeatures

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
    RidgeModelStacking([
        SKLearnWrapper(GradientBoostingRegressor()),
        SKLearnWrapper(GradientBoostingRegressor(n_estimators=500)),
        SKLearnWrapper(GradientBoostingRegressor(max_depth=5)),
        SKLearnWrapper(KMeans()),
    ]),
    NumpyShapePrinter(),
])

print("Fitting on train:")
p = p.fit(X_train, y_train)
print("")

print("Transforming train and test:")
y_train_predicted = p.transform(X_train)
y_test_predicted = p.transform(X_test)
print("")

print("Evaluating transformed train:")
score = r2_score(y_train_predicted, y_train)
print('R2 regression score:', score)
print("")

print("Evaluating transformed test:")
score = r2_score(y_test_predicted, y_test)
print('R2 regression score:', score)
