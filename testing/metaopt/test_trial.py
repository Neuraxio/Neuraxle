import copy
import datetime

import pytest
from neuraxle.base import Identity, TrialStatus
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.logging.logging import LOGGING_DATETIME_STR_FORMAT
from neuraxle.metaopt.data.aggregates import (MetricResults, Round, Trial,
                                              TrialSplit)
from neuraxle.metaopt.data.vanilla import (AutoMLContext,
                                           HyperparamsRepository,
                                           RoundDataclass, ScopedLocation,
                                           VanillaHyperparamsRepository, to_json)
from testing.metaopt.test_repo_dataclasses import (SOME_FULL_SCOPED_LOCATION,
                                                   SOME_ROOT_DATACLASS,
                                                   SOME_TRIAL_DATACLASS)

SOME_OTHER_METRIC_NAME = 'MSE'

EXPECTED_ERROR_TRACEBACK = 'NoneType: None\n'

EXPECTED_METRIC_RESULTS = {
    SOME_OTHER_METRIC_NAME: {
        'train_values': [0.45, 0.6, 0.3],
        'validation_values': [0.5, 0.7, 0.4],
        'higher_score_is_better': False
    }
}


class TestTrials:
    def setup(self):
        self.hp: HyperparameterSamples = HyperparameterSamples({'a': 2})
        round_loc = SOME_FULL_SCOPED_LOCATION[:RoundDataclass]
        self.cx: AutoMLContext = AutoMLContext.from_context().with_loc(round_loc)
        self.trial: Trial = Trial(
            _dataclass=copy.deepcopy(SOME_TRIAL_DATACLASS),
            context=self.cx,
            is_deep=True,
        )
        self.cx.repo.save(
            copy.deepcopy(SOME_ROOT_DATACLASS), ScopedLocation(), deep=True)

    def test_trial_should_have_end_time_later_than_start_time(self):
        with self.trial.new_validation_split(False) as trial_split:
            # trial_split.set_success()
            pass

        assert isinstance(trial_split._dataclass.created_time, datetime.datetime)
        assert isinstance(trial_split._dataclass.start_time, datetime.datetime)
        assert isinstance(trial_split._dataclass.end_time, datetime.datetime)
        assert trial_split._dataclass.created_time <= trial_split.start_time
        assert trial_split._dataclass.start_time < trial_split.end_time

    def test_trial_should_create_new_split(self):
        with self.trial.new_validation_split(False) as trial_split:
            # trial_split.set_success()
            pass

        assert self.trial._validation_splits[-1] == trial_split

    def test_trial_split_is_new_best_score_should_return_true_with_one_score(self):
        with self.trial.new_validation_split(False) as trial_split:
            trial_split: TrialSplit = trial_split.with_n_epochs(1)

            with trial_split.managed_metric(
                SOME_OTHER_METRIC_NAME, higher_score_is_better=False
            ) as metric:
                metric: MetricResults = metric

                metric.add_train_result(0.45)
                metric.add_valid_result(0.5)

                assert metric.is_new_best_score()

    def test_trial_split_is_new_best_score_should_return_false_with_not_a_new_best_score(self):
        with self.trial.new_validation_split(False) as trial_split:
            trial_split: TrialSplit = trial_split
            trial_split.with_n_epochs(2)

            trial_split.next_epoch()
            with trial_split.managed_metric(
                SOME_OTHER_METRIC_NAME, higher_score_is_better=False
            ) as metric:
                metric: MetricResults = metric

                metric.add_train_result(0.45)
                metric.add_valid_result(0.5)

            trial_split.next_epoch()
            with trial_split.managed_metric(
                SOME_OTHER_METRIC_NAME, higher_score_is_better=False
            ) as metric:
                metric: MetricResults = metric

                metric.add_train_result(0.6)
                metric.add_valid_result(0.7)

                assert not metric.is_new_best_score()

    def test_success_trial_split_to_json(self):
        trial_split = self._given_success_trial_validation_split(self.trial)
        trial_json = to_json(trial_split._dataclass)

        self._then_success_trial_split_json_is_valid(trial_json)

    def _then_success_trial_split_json_is_valid(self, trial_json):
        assert trial_json['status'] == TrialStatus.SUCCESS.value
        assert trial_json['error'] is None
        assert trial_json['error_traceback'] is None
        assert trial_json['metric_results'] == EXPECTED_METRIC_RESULTS
        assert trial_json['main_metric_name'] == SOME_OTHER_METRIC_NAME
        start_time = datetime.datetime.strptime(trial_json['start_time'], LOGGING_DATETIME_STR_FORMAT)
        end_time = datetime.datetime.strptime(trial_json['end_time'], LOGGING_DATETIME_STR_FORMAT) + datetime.timedelta(
            hours=1)
        assert start_time < end_time

        return True

    def test_success_trial_to_json(self):
        with self.trial:
            self._given_success_trial_validation_split(self.trial)

        trial_json = self.trial.to_json()

        assert trial_json['status'] == TrialStatus.SUCCESS.value
        assert trial_json['error'] is None
        assert trial_json['error_traceback'] is None
        assert trial_json['main_metric_name'] == self.trial.main_metric_name
        assert self._then_success_trial_split_json_is_valid(trial_json['validation_splits'][0])

        start_time = datetime.datetime.strptime(
            trial_json['start_time'], LOGGING_DATETIME_STR_FORMAT)
        end_time = datetime.datetime.strptime(
            trial_json['end_time'], LOGGING_DATETIME_STR_FORMAT) + datetime.timedelta(hours=1)

        assert start_time < end_time

    def test_success_trial_get_validation_score(self):
        with self.trial:
            self._given_success_trial_validation_split(self.trial, best_score=0.3)

        validation_score = self.trial.get_avg_validation_score()

        assert validation_score == 0.3

    def test_success_trial_multiple_splits_should_average_the_scores(self):
        with self.trial:
            self._given_success_trial_validation_split(self.trial, best_score=0.3)
            self._given_success_trial_validation_split(self.trial, best_score=0.1)

        validation_score = self.trial.get_avg_validation_score()

        assert validation_score == 0.2

    def test_trial_with_failed_split_should_only_average_successful_splits(self):

        with self.trial:
            self._given_success_trial_validation_split(self.trial, best_score=0.03)
            self._given_success_trial_validation_split(self.trial, best_score=0.01)
            self._given_failed_trial_split(self.trial)

        validation_score = self.trial.get_avg_validation_score()

        assert validation_score == 0.02

    def _given_success_trial_validation_split(self, trial: Trial, best_score=0.4):
        with trial.new_validation_split(False) as trial_split:
            trial_split: TrialSplit = trial_split.with_n_epochs(3)

            trial_split.next_epoch()
            with trial_split.managed_metric(
                SOME_OTHER_METRIC_NAME, higher_score_is_better=False
            ) as metric:
                metric: MetricResults = metric

                metric.add_train_result(0.45)
                metric.add_valid_result(0.5)

            trial_split.next_epoch()
            with trial_split.managed_metric(
                SOME_OTHER_METRIC_NAME, higher_score_is_better=False
            ) as metric:
                metric: MetricResults = metric

                metric.add_train_result(0.6)
                metric.add_valid_result(0.7)

            trial_split.next_epoch()
            with trial_split.managed_metric(
                SOME_OTHER_METRIC_NAME, higher_score_is_better=False
            ) as metric:
                metric: MetricResults = metric

                metric.add_train_result(best_score)
                metric.add_valid_result(best_score)

            return trial_split

    def test_failure_trial_split_to_json(self):
        with self.trial:
            trial_split = self._given_failed_trial_split(self.trial)

        trial_json = trial_split.to_json()

        self._then_failed_validation_split_json_is_valid(trial_json, trial_split)

    def _then_failed_validation_split_json_is_valid(self, trial_json, trial_split):
        assert trial_json['status'] == TrialStatus.FAILED.value
        assert trial_json['error'] == str(trial_split.error)
        assert trial_json['error_traceback'] == EXPECTED_ERROR_TRACEBACK
        assert trial_json['metric_results'] == EXPECTED_METRIC_RESULTS
        assert trial_json['main_metric_name'] == trial_split.main_metric_name

        start_time = datetime.datetime.strptime(
            trial_json['start_time'], LOGGING_DATETIME_STR_FORMAT)
        end_time = datetime.datetime.strptime(
            trial_json['end_time'], LOGGING_DATETIME_STR_FORMAT) + datetime.timedelta(hours=1)
        assert start_time < end_time
        return True

    def test_failure_trial_to_json(self):
        with self.trial:
            trial_split = self._given_failed_trial_split(self.trial)

        trial_json = self.trial.to_json()

        assert trial_json['status'] == TrialStatus.FAILED.value
        assert trial_json['error'] == str(trial_split.error)
        assert trial_json['error_traceback'] == EXPECTED_ERROR_TRACEBACK
        assert trial_json['main_metric_name'] == self.trial.main_metric_name
        assert self._then_failed_validation_split_json_is_valid(
            trial_json['validation_splits'][0], trial_split=trial_split)

        start_time = datetime.datetime.strptime(
            trial_json['start_time'], LOGGING_DATETIME_STR_FORMAT)
        end_time = datetime.datetime.strptime(
            trial_json['end_time'], LOGGING_DATETIME_STR_FORMAT) + datetime.timedelta(hours=1)

        assert start_time < end_time

    def _given_failed_trial_split(self, trial: Trial):
        with trial.new_validation_split(True) as trial_split:
            trial_split: TrialSplit = trial_split.with_n_epochs(3)

            trial_split.next_epoch()
            with trial_split.managed_metric(
                SOME_OTHER_METRIC_NAME, higher_score_is_better=False
            ) as metric:
                metric: MetricResults = metric

                metric.add_train_result(0.45)
                metric.add_valid_result(0.5)

            trial_split.next_epoch()
            with trial_split.managed_metric(
                SOME_OTHER_METRIC_NAME, higher_score_is_better=False
            ) as metric:
                metric: MetricResults = metric

                metric.add_train_result(0.6)
                metric.add_valid_result(0.7)

            trial_split.next_epoch()
            with trial_split.managed_metric(
                SOME_OTHER_METRIC_NAME, higher_score_is_better=False
            ) as metric:
                metric: MetricResults = metric

                raise IndexError('index error')

    def test_trials_get_best_hyperparams_should_return_hyperparams_of_best_trial(self):
        # Given
        trial_1 = self.trial
        with trial_1:
            self._given_success_trial_validation_split(trial_1, best_score=0.2)

        hp_trial_2 = HyperparameterSamples({'b': 3})
        trial_2 = Trial(
            trial_number=1, save_trial_function=self.repo.save_trial,
            hyperparams=hp_trial_2, main_metric_name=SOME_OTHER_METRIC_NAME)
        with trial_2:
            self._given_success_trial_validation_split(trial_2, best_score=0.1)

        trials = Round(trials=[trial_1, trial_2])

        # When
        best_hyperparams = trials.get_best_hyperparams()

        # Then
        assert best_hyperparams == hp_trial_2
