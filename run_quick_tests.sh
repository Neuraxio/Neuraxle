#!/usr/bin/env bash
py.test  testing/ -n 10 --ignore=testing/metaopt/test_tpe.py --ignore=testing/examples/test_examples.py --disable-pytest-warnings

