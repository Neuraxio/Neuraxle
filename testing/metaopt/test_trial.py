import datetime

import pytest
from neuraxle.base import Identity, TrialStatus
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.logging.logging import LOGGING_DATETIME_STR_FORMAT
from neuraxle.metaopt.data.trial import RoundManager, TrialManager
from neuraxle.metaopt.data.vanilla import InMemoryHyperparamsRepository

EXPECTED_ERROR_TRACEBACK = 'NoneType: None\n'

EXPECTED_METRIC_RESULTS = {
    'mse': {
        'train_values': [0.5, 0.7, 0.4],
        'validation_values': [0.5, 0.7, 0.4],
        'higher_score_is_better': False
    }
}

MAIN_METRIC_NAME = 'mse'


class TestTrials:
    def setup(self):
        self.hp = HyperparameterSamples({'a': 2})
        self.repo = InMemoryHyperparamsRepository()
        self.trial = TrialManager(
            trial_number=0,
            save_trial_function=self.repo.save_trial,
            hyperparams=self.hp,
            main_metric_name=MAIN_METRIC_NAME
        )

    @pytest.mark.skip(reason="TODO: AutoML Refactor")
    def test_trial_should_have_end_time_later_than_start_time(self):
        with self.trial.new_validation_split(Identity()) as trial_split:
            trial_split.set_success()

        assert isinstance(trial_split.start_time, datetime.datetime)
        assert isinstance(trial_split.end_time, datetime.datetime)
        assert trial_split.start_time < trial_split.end_time

    @pytest.mark.skip(reason="TODO: AutoML Refactor")
    def test_trial_should_create_new_split(self):
        with self.trial.new_validation_split(Identity()) as trial_split:
            trial_split.set_success()

        assert self.trial.validation_splits[0] == trial_split

    @pytest.mark.skip(reason="TODO: AutoML Refactor")
    def test_trial_split_is_new_best_score_should_return_true_with_one_score(self):
        with self.trial.new_validation_split(Identity()) as trial_split:
            trial_split.add_metric_results_train(
                name=MAIN_METRIC_NAME, score=0.5, higher_score_is_better=False)
            trial_split.add_metric_results_validation(
                name=MAIN_METRIC_NAME, score=0.5, higher_score_is_better=False)

        assert trial_split.is_new_best_score()

    @pytest.mark.skip(reason="TODO: AutoML Refactor")
    def test_trial_split_is_new_best_score_should_return_false_with_not_a_new_best_score(self):
        with self.trial.new_validation_split(Identity()) as trial_split:
            trial_split.add_metric_results_train(
                name=MAIN_METRIC_NAME, score=0.5, higher_score_is_better=False)
            trial_split.add_metric_results_validation(
                name=MAIN_METRIC_NAME, score=0.5, higher_score_is_better=False)

            trial_split.add_metric_results_train(
                name=MAIN_METRIC_NAME, score=0.7, higher_score_is_better=False)
            trial_split.add_metric_results_validation(
                name=MAIN_METRIC_NAME, score=0.7, higher_score_is_better=False)

        assert not trial_split.is_new_best_score()

    @pytest.mark.skip(reason="TODO: AutoML Refactor")
    def test_trial_split_is_new_best_score_should_return_true_with_a_new_best_score_after_multiple_scores(self):
        with self.trial.new_validation_split(Identity()) as trial_split:
            trial_split.add_metric_results_train(
                name=MAIN_METRIC_NAME, score=0.5, higher_score_is_better=False)
            trial_split.add_metric_results_validation(
                name=MAIN_METRIC_NAME, score=0.5, higher_score_is_better=False)

            trial_split.add_metric_results_train(
                name=MAIN_METRIC_NAME, score=0.7, higher_score_is_better=False)
            trial_split.add_metric_results_validation(
                name=MAIN_METRIC_NAME, score=0.7, higher_score_is_better=False)

            trial_split.add_metric_results_train(
                name=MAIN_METRIC_NAME, score=0.4, higher_score_is_better=False)
            trial_split.add_metric_results_validation(
                name=MAIN_METRIC_NAME, score=0.4, higher_score_is_better=False)

        assert trial_split.is_new_best_score()

    @pytest.mark.skip(reason="TODO: AutoML Refactor")
    def test_success_trial_split_to_json(self):
        with self.trial:
            trial_split = self._given_success_trial_validation_split(self.trial)
            trial_json = trial_split.to_json()

        self._then_success_trial_split_json_is_valid(trial_json)

    def _then_success_trial_split_json_is_valid(self, trial_json):
        assert trial_json['status'] == TrialStatus.SUCCESS.value
        assert trial_json['error'] is None
        assert trial_json['error_traceback'] is None
        assert trial_json['metric_results'] == EXPECTED_METRIC_RESULTS
        assert trial_json['main_metric_name'] == MAIN_METRIC_NAME
        start_time = datetime.datetime.strptime(trial_json['start_time'], LOGGING_DATETIME_STR_FORMAT)
        end_time = datetime.datetime.strptime(trial_json['end_time'], LOGGING_DATETIME_STR_FORMAT) + datetime.timedelta(
            hours=1)
        assert start_time < end_time

        return True

    @pytest.mark.skip(reason="TODO: AutoML Refactor")
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

    @pytest.mark.skip(reason="TODO: AutoML Refactor")
    def test_success_trial_get_validation_score(self):
        with self.trial:
            self._given_success_trial_validation_split(self.trial, best_score=0.3)

        validation_score = self.trial.get_avg_validation_score()

        assert validation_score == 0.3

    @pytest.mark.skip(reason="TODO: AutoML Refactor")
    def test_success_trial_multiple_splits_should_average_the_scores(self):
        with self.trial:
            self._given_success_trial_validation_split(self.trial, best_score=0.3)
            self._given_success_trial_validation_split(self.trial, best_score=0.1)

        validation_score = self.trial.get_avg_validation_score()

        assert validation_score == 0.2

    @pytest.mark.skip(reason="TODO: AutoML Refactor")
    def test_trial_with_failed_split_should_only_average_successful_splits(self):

        with self.trial:
            self._given_success_trial_validation_split(self.trial, best_score=0.3)
            self._given_success_trial_validation_split(self.trial, best_score=0.1)
            self._given_failed_trial_split(self.trial)

        validation_score = self.trial.get_avg_validation_score()

        assert validation_score == 0.2

    def _given_success_trial_validation_split(self, trial, best_score=0.4):
        with trial.new_validation_split(Identity()) as trial_split:
            trial_split.add_metric_results_train(
                name=MAIN_METRIC_NAME, score=0.5, higher_score_is_better=False)
            trial_split.add_metric_results_validation(
                name=MAIN_METRIC_NAME, score=0.5,
                higher_score_is_better=False)

            trial_split.add_metric_results_train(
                name=MAIN_METRIC_NAME, score=0.7, higher_score_is_better=False)
            trial_split.add_metric_results_validation(
                name=MAIN_METRIC_NAME, score=0.7, higher_score_is_better=False)

            trial_split.add_metric_results_train(
                name=MAIN_METRIC_NAME, score=best_score, higher_score_is_better=False)
            trial_split.add_metric_results_validation(
                name=MAIN_METRIC_NAME, score=best_score, higher_score_is_better=False)

            trial_split.set_success()
            trial.set_success()

        return trial_split

    @pytest.mark.skip(reason="TODO: AutoML Refactor")
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

    @pytest.mark.skip(reason="TODO: AutoML Refactor")
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

    def _given_failed_trial_split(self, trial):
        with trial.new_validation_split(Identity()) as trial_split:
            trial_split.add_metric_results_train(
                name=MAIN_METRIC_NAME, score=0.5, higher_score_is_better=False)
            trial_split.add_metric_results_validation(
                name=MAIN_METRIC_NAME, score=0.5, higher_score_is_better=False)

            trial_split.add_metric_results_train(
                name=MAIN_METRIC_NAME, score=0.7, higher_score_is_better=False)
            trial_split.add_metric_results_validation(
                name=MAIN_METRIC_NAME, score=0.7, higher_score_is_better=False)

            trial_split.add_metric_results_train(
                name=MAIN_METRIC_NAME, score=0.4, higher_score_is_better=False)
            trial_split.add_metric_results_validation(
                name=MAIN_METRIC_NAME, score=0.4, higher_score_is_better=False)
            error = IndexError('index error')
            trial_split.set_failed(error)
            trial.set_failed(error)
        return trial_split

    @pytest.mark.skip(reason="TODO: AutoML Refactor")
    def test_trials_get_best_hyperparams_should_return_hyperparams_of_best_trial(self):
        # Given
        trial_1 = self.trial
        with trial_1:
            self._given_success_trial_validation_split(trial_1, best_score=0.2)

        hp_trial_2 = HyperparameterSamples({'b': 3})
        trial_2 = TrialManager(
            trial_number=1, save_trial_function=self.repo.save_trial,
            hyperparams=hp_trial_2, main_metric_name=MAIN_METRIC_NAME)
        with trial_2:
            self._given_success_trial_validation_split(trial_2, best_score=0.1)

        trials = RoundManager(trials=[trial_1, trial_2])

        # When
        best_hyperparams = trials.get_best_hyperparams()

        # Then
        assert best_hyperparams == hp_trial_2
