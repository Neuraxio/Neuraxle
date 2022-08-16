"""
Usage of AutoML loop, and hyperparams with sklearn models.
=============================================================

This demonstrates how you can build an AutoML loop that finds the best possible sklearn classifier.
It also shows you how to add hyperparams to sklearn steps using SKLearnWrapper.
This example has been derived and simplified from the following repository: https://github.com/Neuraxio/Kata-Clean-Machine-Learning-From-Dirty-Code
Here, 2D data is fitted, whereas in the original example 3D (sequential / time series) data is preprocessed and then fitted with the same models.

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
import shutil

from neuraxle.base import ExecutionContext as CX
from neuraxle.hyperparams.distributions import (Boolean, Choice, LogUniform,
                                                RandInt)
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.auto_ml import (AutoML, RandomSearchSampler,
                                      ValidationSplitter)
from neuraxle.metaopt.callbacks import ScoringCallback
from neuraxle.metaopt.repositories.json import HyperparamsOnDiskRepository
from neuraxle.pipeline import Pipeline
from neuraxle.steps.flow import ChooseOneStepOf
from neuraxle.steps.numpy import NumpyRavel
from neuraxle.steps.output_handlers import OutputTransformerWrapper
from neuraxle.steps.sklearn import SKLearnWrapper
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


def main(tmpdir: str):
    # Define classification models, and hyperparams.

    decision_tree_classifier = SKLearnWrapper(
        DecisionTreeClassifier(),
        HyperparameterSpace({
            'criterion': Choice(['gini', 'entropy']),
            'splitter': Choice(['best', 'random']),
            'min_samples_leaf': RandInt(2, 5),
            'min_samples_split': RandInt(2, 4)
        }))

    extra_tree_classifier = SKLearnWrapper(
        ExtraTreeClassifier(),
        HyperparameterSpace({
            'criterion': Choice(['gini', 'entropy']),
            'splitter': Choice(['best', 'random']),
            'min_samples_leaf': RandInt(2, 5),
            'min_samples_split': RandInt(2, 4)
        }))

    ridge_classifier = Pipeline([OutputTransformerWrapper(NumpyRavel()), SKLearnWrapper(
        RidgeClassifier(),
        HyperparameterSpace({
            'alpha': Choice([0.0, 1.0, 10.0, 100.0]),
            'fit_intercept': Boolean(),
            'normalize': Boolean()
        }))
    ]).set_name('RidgeClassifier')

    logistic_regression = Pipeline([OutputTransformerWrapper(NumpyRavel()), SKLearnWrapper(
        LogisticRegression(),
        HyperparameterSpace({
            'C': LogUniform(0.01, 10.0),
            'fit_intercept': Boolean(),
            'penalty': Choice(['none', 'l2']),
            'max_iter': RandInt(20, 200)
        }))
    ]).set_name('LogisticRegression')

    random_forest_classifier = Pipeline([OutputTransformerWrapper(NumpyRavel()), SKLearnWrapper(
        RandomForestClassifier(),
        HyperparameterSpace({
            'n_estimators': RandInt(50, 600),
            'criterion': Choice(['gini', 'entropy']),
            'min_samples_leaf': RandInt(2, 5),
            'min_samples_split': RandInt(2, 4),
            'bootstrap': Boolean()
        }))
    ]).set_name('RandomForestClassifier')

    # Define a classification pipeline that lets the AutoML loop choose one of the classifier.
    # See also ChooseOneStepOf documentation: https://www.neuraxle.org/stable/api/steps/neuraxle.steps.flow.html#neuraxle.steps.flow.ChooseOneStepOf

    pipeline = Pipeline([
        ChooseOneStepOf([
            decision_tree_classifier,
            extra_tree_classifier,
            ridge_classifier,
            logistic_regression,
            random_forest_classifier
        ])
    ])

    # Create the AutoML loop object.
    # See also AutoML documentation: https://www.neuraxle.org/stable/api/metaopt/neuraxle.metaopt.auto_ml.html#neuraxle.metaopt.auto_ml.AutoML

    auto_ml = AutoML(
        pipeline=pipeline,
        hyperparams_optimizer=RandomSearchSampler(),
        validation_splitter=ValidationSplitter(validation_size=0.20).set_to_force_expected_outputs_for_scoring(),
        scoring_callback=ScoringCallback(accuracy_score, higher_score_is_better=True),
        n_trials=7,
        epochs=1,
        hyperparams_repository=HyperparamsOnDiskRepository(cache_folder=tmpdir),
        refit_best_trial=True,
        continue_loop_on_error=False
    )

    # Load data, and launch AutoML loop !

    X_train, y_train, X_test, y_test = generate_classification_data()
    auto_ml = auto_ml.fit(X_train, y_train)

    # Get the model from the best trial, and make predictions using predict, as per the `refit_best_trial=True` argument to AutoML.
    y_pred = auto_ml.predict(X_test)

    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Test accuracy score:", accuracy)

    shutil.rmtree(tmpdir)


def generate_classification_data():
    data_inputs, expected_outputs = make_classification(
        n_samples=10000,
        n_repeated=0,
        n_classes=3,
        n_features=4,
        n_clusters_per_class=1,
        class_sep=1.5,
        flip_y=0,
        weights=[0.5, 0.5, 0.5]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        data_inputs,
        expected_outputs,
        test_size=0.20
    )

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    main(CX.get_new_cache_folder())
