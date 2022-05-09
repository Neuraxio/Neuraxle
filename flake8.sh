#!/usr/bin/env bash
flake8 neuraxle testing_neuraxle --count --max-line-length=120 --select=E9,F63,F7,F82 --statistics --show-source

