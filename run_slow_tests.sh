#!/usr/bin/env bash
py.test -n 10 testing_neuraxle/metaopt/test_tpe.py testing_neuraxle/examples/test_examples.py --disable-pytest-warnings --durations=10 --timeout=60

