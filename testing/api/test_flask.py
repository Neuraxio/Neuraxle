"""
Tests for Flask.
============================================

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

from neuraxle.api.flask import JSONDataResponseEncoder, JSONDataBodyDecoder, FlaskRestApiWrapper
from neuraxle.base import BaseStep, NonFittableMixin


def setup_api():
    class Decoder(JSONDataBodyDecoder):
        """This is a custom JSON decoder class that precedes the pipeline's transformation."""

        def decode(self, data_inputs):
            """
            Transform a JSON list object into an np.array object.

            :param data_inputs: json object
            :return: np array for data inputs
            """
            print(data_inputs)
            return np.array(data_inputs)

    class Encoder(JSONDataResponseEncoder):
        """This is a custom JSON response encoder class for converting the pipeline's transformation outputs."""

        def encode(self, data_inputs) -> dict:
            """
            Convert predictions to a dict for creating a JSON Response object.

            :param data_inputs:
            :return:
            """
            return {
                'predictions': data_inputs.tolist()
            }

    class Multiplier(NonFittableMixin, BaseStep):
        def transform(self, data_inputs):
            return 2 * data_inputs

    app = FlaskRestApiWrapper(
        json_decoder=Decoder(),
        wrapped=Multiplier(),
        json_encoder=Encoder()
    ).get_app()

    app.testing = True

    test_client = app.test_client()

    return test_client


def test_api_wrapper_works():
    test_client = setup_api()
    data_inputs = [
        [0, 1, 2],
        [3, 4, 5],
    ]

    json_response = test_client.get('/', json=data_inputs)

    predictions_np_arr = np.array(json_response.json["predictions"])
    expected_outputs = 2 * np.array(data_inputs)
    assert np.array_equal(predictions_np_arr, expected_outputs)
