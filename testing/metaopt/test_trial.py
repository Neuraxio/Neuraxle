import copy
import datetime
import json

import pytest
from neuraxle.base import Identity, TrialStatus
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.logging.logging import LOGGING_DATETIME_STR_FORMAT
from neuraxle.metaopt.data.aggregates import (MetricResults, Round, Trial,
                                              TrialSplit)
from neuraxle.metaopt.data.vanilla import (AutoMLContext, ClientDataclass,
                                           HyperparamsRepository,
                                           MetricResultsDataclass,
                                           RoundDataclass, ScopedLocation,
                                           TrialDataclass,
                                           VanillaHyperparamsRepository,
                                           from_json, to_json)
from testing.metaopt.test_automl_dataclasses import (SOME_FULL_SCOPED_LOCATION,
                                                   SOME_ROOT_DATACLASS,
                                                   SOME_TRIAL_DATACLASS)

SOME_OTHER_METRIC_NAME = 'MSE'

EXPECTED_ERROR_TRACEBACK = 'NoneType: None\n'

EXPECTED_METRIC_RESULTS = {
    SOME_OTHER_METRIC_NAME: {
        '__type__': MetricResultsDataclass.__name__,
        'metric_name': SOME_OTHER_METRIC_NAME,
        'validation_values': [0.5, 0.7, 0.4],
        'train_values': [0.45, 0.6, 0.4],
        'higher_score_is_better': False,
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
            trial_split._set_success()
            pass

        assert isinstance(trial_split._dataclass.created_time, datetime.datetime)
        assert isinstance(trial_split._dataclass.start_time, datetime.datetime)
        assert isinstance(trial_split._dataclass.end_time, datetime.datetime)
        assert trial_split._dataclass.created_time <= trial_split._dataclass.start_time
        assert trial_split._dataclass.start_time < trial_split._dataclass.end_time

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

        self._then_success_trial_split_json_is_valid(json.loads(trial_json))

    def _then_success_trial_split_json_is_valid(self, trial_json):
        assert trial_json['status'] == TrialStatus.SUCCESS.value

        assert MetricResultsDataclass.from_dict(
            trial_json['metric_results'][SOME_OTHER_METRIC_NAME]
        ) == MetricResultsDataclass.from_dict(
            EXPECTED_METRIC_RESULTS[SOME_OTHER_METRIC_NAME])

        start_time = datetime.datetime.strptime(
            trial_json['start_time'], LOGGING_DATETIME_STR_FORMAT)
        created_time = datetime.datetime.strptime(
            trial_json['created_time'], LOGGING_DATETIME_STR_FORMAT)
        end_time = datetime.datetime.strptime(
            trial_json['end_time'], LOGGING_DATETIME_STR_FORMAT)
        assert created_time <= start_time
        assert created_time < end_time
        assert start_time < end_time

        return True

    def test_success_trial_to_json(self):
        self._given_success_trial_validation_split(self.trial)

        trial_json = json.loads(to_json(self.trial._dataclass))

        assert trial_json['status'] == TrialStatus.SUCCESS.value

        assert self._then_success_trial_split_json_is_valid(trial_json['validation_splits'][-1])

        start_time = datetime.datetime.strptime(
            trial_json['start_time'], LOGGING_DATETIME_STR_FORMAT)
        end_time = datetime.datetime.strptime(
            trial_json['end_time'], LOGGING_DATETIME_STR_FORMAT)

        assert start_time < end_time

    def test_success_trial_get_validation_score(self):
        self._given_success_trial_validation_split(self.trial, best_score=0.3)

        validation_score = self.trial.get_avg_validation_score(SOME_OTHER_METRIC_NAME)

        assert validation_score == 0.3

    def test_success_trial_multiple_splits_should_average_the_scores(self):
        self._given_success_trial_validation_split(self.trial, best_score=0.3)
        self._given_success_trial_validation_split(self.trial, best_score=0.1)

        validation_score = self.trial.get_avg_validation_score(SOME_OTHER_METRIC_NAME)

        assert validation_score == 0.2

    def test_trial_with_failed_split_should_only_average_successful_splits(self):
        self._given_success_trial_validation_split(self.trial, best_score=0.03)
        self._given_success_trial_validation_split(self.trial, best_score=0.01)
        self._given_failed_trial_split_that_continues_on_error(self.trial)

        validation_score = self.trial.get_avg_validation_score(SOME_OTHER_METRIC_NAME)

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
        trial_split = self._given_failed_trial_split_that_continues_on_error(self.trial)

        trial_json = to_json(trial_split._dataclass)

        self._then_failed_validation_split_json_is_valid(json.loads(trial_json), trial_split)

    def _then_failed_validation_split_json_is_valid(self, trial_json, trial_split):
        assert trial_json['status'] == TrialStatus.FAILED.value
        # assert trial_json['error'] == str(trial_split.error)
        # assert trial_json['error_traceback'] == EXPECTED_ERROR_TRACEBACK

        assert MetricResultsDataclass.from_dict(
            trial_json['metric_results'][SOME_OTHER_METRIC_NAME]
        ) == MetricResultsDataclass.from_dict(
            EXPECTED_METRIC_RESULTS[SOME_OTHER_METRIC_NAME])

        start_time = datetime.datetime.strptime(
            trial_json['start_time'], LOGGING_DATETIME_STR_FORMAT)
        end_time = datetime.datetime.strptime(
            trial_json['end_time'], LOGGING_DATETIME_STR_FORMAT)
        assert start_time < end_time
        return True

    def test_failure_trial_to_json(self):
        trial_split = self._given_failed_trial_split_that_continues_on_error(self.trial)

        trial_json = json.loads(to_json(self.trial._dataclass))

        # assert trial_json['error'] == str(trial_split.error)
        # assert trial_json['error_traceback'] == EXPECTED_ERROR_TRACEBACK
        assert self._then_failed_validation_split_json_is_valid(
            trial_json['validation_splits'][-1], trial_split)

        created_time = datetime.datetime.strptime(
            trial_json['created_time'], LOGGING_DATETIME_STR_FORMAT)
        end_time = datetime.datetime.strptime(
            trial_json['end_time'], LOGGING_DATETIME_STR_FORMAT)

        assert created_time < end_time

    def _given_failed_trial_split_that_continues_on_error(self, trial: Trial):
        """
        This method isn't supposed to raise anything.
        """
        try:
            with trial.new_validation_split(continue_loop_on_error=True) as trial_split:
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

                    metric.add_train_result(0.4)
                    metric.add_valid_result(0.4)

                    raise IndexError('index error')
            return trial._validation_splits[-1]
        except IndexError as e:
            assert False, (
                f"This should not happen: IndexError should be handled by the "
                f"`trial.new_validation_split(continue_loop_on_error=True)` "
                f"context manager... Here is the error: `{e}`.")

    def test_trials_get_best_hyperparams_should_return_hyperparams_of_best_trial(self):
        # Given
        trial_1 = self.trial
        trial_split_1 = self._given_success_trial_validation_split(trial_1, best_score=0.2)

        hp_trial_2 = HyperparameterSamples({'b': 3})
        trial_2 = Trial(
            _dataclass=TrialDataclass(
                trial_number=1,
                hyperparams=hp_trial_2,
            ).start(),
            context=self.cx,
            is_deep=True)
        self.cx.repo.save(trial_2._dataclass, SOME_FULL_SCOPED_LOCATION[:RoundDataclass].with_id(1))
        trial__split_2 = self._given_success_trial_validation_split(trial_2, best_score=0.1)
        self.cx.repo.save(trial_2._set_success()._dataclass, SOME_FULL_SCOPED_LOCATION[:RoundDataclass].with_id(1))

        trials = Round(
            RoundDataclass(trials=[trial_1._dataclass, trial_2._dataclass]),
            context=self.cx.with_loc(SOME_FULL_SCOPED_LOCATION[:ClientDataclass]),
            is_deep=True)

        # When
        best_hyperparams = trials.get_best_hyperparams(SOME_OTHER_METRIC_NAME)

        # Then
        assert best_hyperparams == hp_trial_2
        assert trial_2.is_success()
