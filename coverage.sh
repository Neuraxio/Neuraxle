#!/usr/bin/env bash
./flake8.sh
pytest --cov-report html --cov-report xml:cov.xml --cov=neuraxle testing

