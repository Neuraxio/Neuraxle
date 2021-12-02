"""
Neuraxle's SQLAlchemy Hyperparameter Repository Classes
=================================================
Data objects and related repositories used by AutoML, SQL version.

Classes are splitted like this for the AutoML:
- Projects
- Clients
- Rounds (runs)
- Trials
- TrialSplits
- MetricResults

..
    Copyright 2021, Neuraxio Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.


"""
from typing import List

from neuraxle.base import TrialStatus
from neuraxle.data_container import DataContainer as DACT
from neuraxle.metaopt.data.aggregates import Round, Trial
from neuraxle.metaopt.data.vanilla import HyperparamsRepository
from sqlalchemy import (Boolean, DateTime, MetaData, PickleType, String, Table,
                        create_engine)
from sqlalchemy.testing.schema import Column


def get_database_path(user: str, password: str, host: str, dialect: str, driver: str = ''):
    """

    :param user:
    :param password:
    :param dialect: e.g. mysql
    :param driver: (Optional) e.g. "pymysql"
    :return:
    """
    return f"dialect[+driver]://user:password@host/dbname"


def get_database_path_sqlite(path):
    return f"sqlite:///{path}"


meta = MetaData()

trial_register = Table(
    'trials', meta,
    Column('id', String, primary_key=True),
    Column('status', String),
    Column('hyperparams', PickleType),
    Column('main_metric_name', String),
    Column('start_time', DateTime),
    Column('end_time', DateTime),
    Column('error', String),
    Column('error_traceback', String)
)

metrics_register = Table(
    'metrics', meta,
    Column('name', String),
    Column('is_higher_score_better', Boolean)
)

metrics_measurements = Table(
    'measurements', meta,
    Column('trial_id', String, foreign_key=True),
    Column('is_validation', Boolean),
    Column('value', PickleType)  # Should contain a list of measurements
)


class InDatabaseHyperparamRepository(HyperparamsRepository):
    def __init__(self, db_path):
        self.engine = create_engine(db_path, echo=True)

    def create_db(self):
        meta.create_all(self.engine)

    def _execute(self, *expressions: List[str]):
        """
        :param expressions:
        :return:
        :rtype Union[ResultProxy, List[ResultProxy]]
        """
        if len(expressions) == 0:
            raise ValueError("No expressions provided for execution!")

        result = []
        with self.engine.connect() as conn:
            for expr in expressions:
                result.append(conn.execute(expr))

        return result[0] if len(result) == 1 else result

    def load_trials(self, status: 'TrialStatus') -> 'Round':
        """
        Load all hyperparameter trials with their corresponding score.
        Sorted by creation date.

        :return: Trials (hyperparams, scores)
        """
        select = trial_register.select()
        if status is not None:
            select = select.where(status=TrialStatus.value)
        res = self._execute(select)

        trials = Round()

        for row in res:
            raise NotImplementedError()
            trial = Trial()
            trials.append(trial)

    def _save_trial(self, trial: 'Trial'):
        """
        save trial.

        :param trial: trial to save.
        :return:
        """
        raise NotImplementedError()

    def _insert_new_trial(self, trial: Trial):
        trial_hash = self.get_trial_id(trial.hyperparams)
        return trial_register.insert().value(id=trial_hash, status=TrialStatus.PLANNED)

    def new_trial(self, auto_ml_container):
        """
        Create a new trial with the best next hyperparams.

        :param context:
        :param auto_ml_container: auto ml data container
        :return: trial
        """
        trial = HyperparamsRepository.new_trial(self, auto_ml_container)
        query = self._insert_new_trial(trial)
        self._execute(query)
        return trial
