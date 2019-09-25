"""
TODO: module name docstring
==============================================

TODO: module description docstring
"""
from abc import ABC

from neuraxle.base import BaseStep, NonFittableMixin
from neuraxle.pipeline import Pipeline


class JSONDataBodyDecoder(NonFittableMixin, BaseStep, ABC):  # TODO: is order of ABC good here
    """
    Class to be used within a FlaskRESTApiWrapper to convert input json to actual data (e.g.: arrays)
    """

    # TODO: perhaps have some handle_transform functionnality in the future to pass
    #  hashes and/or IDs as inputs (ADD ISSUE)

    def transform(self, data_inputs):
        # TODO: you might need to edit this transform method if using `Flask-RESTful`.
        #  But please make such that `decode` receive json.
        return self.decode(data_inputs)

    def decode(self, data_inputs: dict):
        """
        Will convert data_inputs to a dict or a compatible data structure for jsonification

        :param encoded data_inputs (dict parsed from json):
        :return: data_inputs (as a data structure compatible with pipeline's data inputs)
        """
        raise NotImplementedError("TODO: inherit from the `JSONDataBodyDecoder` class and implement this method.")


class JSONDataResponseEncoder(NonFittableMixin, BaseStep, ABC):  # TODO: is order of ABC good here?
    """
    Base class to be used within a FlaskRESTApiWrapper to convert prediction output to json response.
    """

    # TODO: perhaps have some handle_transform functionnality in the future to allow
    #  returning hashes and/or IDs as inputs (ADD ISSUE)

    def transform(self, data_inputs):
        from flask import jsonify
        return jsonify(self.encode(data_inputs))

    def encode(self, data_inputs) -> dict:
        """
        Will convert data_inputs to a dict or a compatible data structure for jsonification.

        :param data_inputs (a data structure outputted by the pipeline after a transform):
        :return: encoded data_inputs (jsonifiable dict)
        """
        raise NotImplementedError("TODO: inherit from the `JSONDataResponseEncoder` class and implement this method.")


class FlaskRestApiWrapper(Pipeline):
    """
    Wrap a pipeline to easily deploy it to a REST API. Just provide a json encoder and a json decoder.

    Usage example:

    ```
    class CustomJSONDecoderFor2DArray(JSONDataBodyDecoder):
        '''This is a custom JSON decoder class that precedes the pipeline's transformation.'''

        def decode(self, data_inputs: dict):
            values_in_json_2d_arr: List[List[int]] = data_inputs["values"]
            return np.array(values_in_json_2d_arr)

    class CustomJSONEncoderOfOutputs(JSONDataResponseEncoder):
        '''This is a custom JSON response encoder class for converting the pipeline's transformation outputs.'''

        def encode(self, data_inputs) -> dict:
            return {
                'predictions': list(data_inputs)
            }

    app = FlaskRestApiWrapper(
        json_decoder=CustomJSONDecoderFor2DArray(),
        wrapped=Pipeline(...),
        json_encoder=CustomJSONEncoderOfOutputs(),
    ).get_app()

    app.run(debug=False, port=5000)
    ```
    """

    def __init__(
            self,
            json_decoder: JSONDataBodyDecoder,
            wrapped: BaseStep,
            json_encoder: JSONDataResponseEncoder,
            route='/',
    ):  # TODO: auth and https. Might open issue and do this later. I'd see that as optional arguments here in init.
        super().__init__([
            json_decoder,
            wrapped,
            json_encoder
        ])
        self.route: str = route

    def get_app(self):
        """
        This methods returns a REST API wrapping the pipeline.

        :return: a Flask app (as given by `app = Flask(__name__)` and then configured).
        """
        from flask import Flask, request
        from flask_restful import Api, Resource

        app = Flask(__name__)
        api = Api(app)

        class RESTfulRes(Resource):
            def get(self):
                json_data_inputs = request.get_json()
                json_data_outputs = self.transform(json_data_inputs)
                return json_data_outputs

        api.add_resource(
            RESTfulRes,
            self.route
        )

        return app

# TODO: all the incomplete docstrings here.
