#!/usr/bin/env bash
./flake8.sh
pytest -n 7 --cov-report html --cov-report xml:cov.xml --cov=neuraxle testing
# pytest --cov-report html --cov=neuraxle testing; open htmlcov/index.html

