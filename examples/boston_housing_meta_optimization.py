"""
Boston Housing Regression with Meta Optimization
================================================

This is an automatic machine learning example. It is more sophisticated than the other simple regression example.
Not only a pipeline is defined, but also an hyperparameter space is defined for the pipeline. Then, a random search is
performed to find the best possible combination of hyperparameters by sampling randomly in the hyperparameter space.

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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from neuraxle.hyperparams.distributions import RandInt, LogUniform, Boolean
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.random import RandomSearch, KFoldCrossValidation
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import NumpyTranspose
from neuraxle.steps.sklearn import SKLearnWrapper
from neuraxle.union import AddFeatures, ModelStacking

boston = load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

# Note that the hyperparameter spaces are defined here during the pipeline definition, but it could be already set
# within the classes ar their definition if using custom classes, or also it could be defined after declaring the
# pipeline using a flat dict or a nested dict.
p = Pipeline([
    AddFeatures([
        SKLearnWrapper(
            PCA(n_components=2),
            HyperparameterSpace({"n_components": RandInt(1, 3)})
        ),
        SKLearnWrapper(
            FastICA(n_components=2),
            HyperparameterSpace({"n_components": RandInt(1, 3)})
        ),
    ]),
    ModelStacking([
        SKLearnWrapper(
            GradientBoostingRegressor(),
            HyperparameterSpace({
                "n_estimators": RandInt(50, 600), "max_depth": RandInt(1, 10), "learning_rate": LogUniform(0.07, 0.7)
            })
        ),
        SKLearnWrapper(
            GradientBoostingRegressor(),
            HyperparameterSpace({
                "n_estimators": RandInt(50, 600), "max_depth": RandInt(1, 10), "learning_rate": LogUniform(0.07, 0.7)
            })
        ),
        SKLearnWrapper(
            GradientBoostingRegressor(),
            HyperparameterSpace({
                "n_estimators": RandInt(50, 600), "max_depth": RandInt(1, 10), "learning_rate": LogUniform(0.07, 0.7)
            })
        ),
        SKLearnWrapper(
            KMeans(),
            HyperparameterSpace({"n_clusters": RandInt(5, 10)})
        ),
    ],
        joiner=NumpyTranspose(),
        judge=SKLearnWrapper(
            Ridge(),
            HyperparameterSpace({"alpha": LogUniform(0.7, 1.4), "fit_intercept": Boolean()})
        ),
    )
])

print("Meta-fitting on train:")
p = p.meta_fit(X_train, y_train, metastep=RandomSearch(
    n_iter=10,
    higher_score_is_better=True,
    validation_technique=KFoldCrossValidation(scoring_function=r2_score, k_fold=10)
))
# Here is an alternative way to do it, more "pipeliney":
# p = RandomSearch(
#     n_iter=15,
#     higher_score_is_better=True,
#     validation_technique=KFoldCrossValidation(scoring_function=r2_score, k_fold=3)
# ).set_step(p).fit(X_train, y_train).get_best_model()
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
