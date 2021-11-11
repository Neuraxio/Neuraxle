#!/usr/bin/env bash
py.test -n 10 testing/metaopt/test_tpe.py testing/examples/test_examples.py --disable-pytest-warnings --durations=10

