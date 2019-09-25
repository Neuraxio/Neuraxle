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

from typing import List

import numpy as np

from neuraxle.api.flask import FlaskRestApiWrapper, JSONDataResponseEncoder, JSONDataBodyDecoder
from neuraxle.base import BaseStep, NonFittableMixin


def setup_api():
    # TODO: here notice that the two custom classes below are duplicate code from the documentation example I wrote.
    #   However, I want the custom class to be defined manually in the documentation as a full implementation example.
    #   I suggest to delete this todo without care for now.

    class CustomJSONDecoderFor2DArray(JSONDataBodyDecoder):
        """This is a custom JSON decoder class that precedes the pipeline's transformation."""

        def decode(self, data_inputs: dict):
            values_in_json_2d_arr: List[List[int]] = data_inputs["values"]
            return np.array(values_in_json_2d_arr)

    class Multiplier(NonFittableMixin, BaseStep):
        # TODO: extract and clean this step to somewhere else where appropriate.
        #  And even possibly make the multiplier a float hyperparam.
        def transform(self, data_inputs):
            assert data_inputs.shape == (2, 3)
            return 2 * data_inputs

    class CustomJSONEncoderOfOutputs(JSONDataResponseEncoder):
        """This is a custom JSON response encoder class for converting the pipeline's transformation outputs."""

        def encode(self, data_inputs) -> dict:
            return {
                'predictions': list(data_inputs)
            }

    app = FlaskRestApiWrapper(
        json_decoder=CustomJSONDecoderFor2DArray(),
        wrapped=Multiplier(),
        json_encoder=CustomJSONEncoderOfOutputs(),
    ).get_app()

    app.testing = True

    test_client = app.test_client()

    return test_client


def test_api_wrapper_works():
    # TODO: make the test pass as it is designed.
    test_client = setup_api()
    data_inputs = [
        [0, 1, 2],
        [3, 4, 5],
    ]

    json_response = test_client.post('/', json={data_inputs})

    predictions_np_arr = np.array(json_response["predictions"])
    expected_outputs = 2 * np.array(data_inputs)
    assert predictions_np_arr == expected_outputs
