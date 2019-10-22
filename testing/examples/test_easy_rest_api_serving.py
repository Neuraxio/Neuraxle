from flask import Flask

from examples.easy_rest_api_serving import main


def test_easy_rest_api_serving():
    app = main()

    assert isinstance(app, Flask)