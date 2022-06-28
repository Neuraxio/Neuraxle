#!/usr/bin/env bash
./flake8.sh
pytest -n 7 --cov-report html --cov-report xml:cov.xml --cov-config=.coveragerc --cov=neuraxle testing_neuraxle
# pytest --cov-report html --cov=neuraxle testing_neuraxle; open htmlcov/index.html

