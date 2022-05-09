#!/usr/bin/env bash
py.test testing_neuraxle/ -n 10 --ignore=testing_neuraxle/metaopt/test_tpe.py --ignore=testing_neuraxle/examples/test_examples.py --disable-pytest-warnings --durations=10 $1 $2 $3 $4

