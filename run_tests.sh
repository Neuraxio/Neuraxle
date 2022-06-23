#!/usr/bin/env bash
py.test -n 10 testing_neuraxle/ --disable-pytest-warnings --durations=10 --timeout=60
