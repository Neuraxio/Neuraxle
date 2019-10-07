"""
Easy REST API Model Serving with Neuraxle
================================================

This demonstrates an easy way to deploy your Neuraxle model or pipeline to a REST API.

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

from neuraxle.api.flask import FlaskRestApiWrapper, JSONDataBodyDecoder, JSONDataResponseEncoder
from neuraxle.pipeline import Pipeline
from neuraxle.steps.sklearn import SKLearnWrapper, RidgeModelStacking
from neuraxle.union import AddFeatures

boston = load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

pipeline = Pipeline([
    AddFeatures([
        SKLearnWrapper(PCA(n_components=2)),
        SKLearnWrapper(FastICA(n_components=2)),
    ]),
    RidgeModelStacking([
        SKLearnWrapper(GradientBoostingRegressor()),
        SKLearnWrapper(KMeans()),
    ]),
])

print("Fitting on train:")
pipeline = pipeline.fit(X_train, y_train)
print("")

print("Transforming train and test:")
y_train_predicted = pipeline.transform(X_train)
y_test_predicted = pipeline.transform(X_test)
print("")

print("Evaluating transformed train:")
score = r2_score(y_train_predicted, y_train)
print('R2 regression score:', score)
print("")

print("Evaluating transformed test:")
score = r2_score(y_test_predicted, y_test)
print('R2 regression score:', score)

print("Deploying the application by routing data to the transform method:")


class CustomJSONDecoderFor2DArray(JSONDataBodyDecoder):
    """This is a custom JSON decoder class that precedes the pipeline's transformation."""

    def decode(self, data_inputs):
        """
        Transform a JSON list object into an np.array object.

        :param data_inputs: json object
        :return: np array for data inputs
        """
        return np.array(data_inputs)


class CustomJSONEncoderOfOutputs(JSONDataResponseEncoder):
    """This is a custom JSON response encoder class for converting the pipeline's transformation outputs."""

    def encode(self, data_inputs) -> dict:
        """
        Convert predictions to a dict for creating a JSON Response object.

        :param data_inputs:
        :return:
        """
        return {
            'predictions': list(data_inputs)
        }


app = FlaskRestApiWrapper(
    json_decoder=CustomJSONDecoderFor2DArray(),
    wrapped=pipeline,
    json_encoder=CustomJSONEncoderOfOutputs()
).get_app()

print("Finally, run the app by uncommenting this next line of code:")
# app.run(debug=False, port=5000)

print("You can now call your pipeline over HTTP with a (JSON) REST API.")
# test_predictictions = requests.post(
#     url='http://127.0.0.1:5000/',
#     json=X_test.tolist()
# )
# print(test_predictictions)
# print(test_predictictions.content)
