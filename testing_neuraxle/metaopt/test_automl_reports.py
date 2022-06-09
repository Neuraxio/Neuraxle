
from typing import List

import pytest
from neuraxle.metaopt.data.reporting import (BaseReport, ClientReport,
                                             MetricResultsReport,
                                             ProjectReport, RoundReport,
                                             TrialReport, TrialSplitReport,
                                             dataclass_2_report)
from neuraxle.metaopt.data.vanilla import BaseDataclass
from testing_neuraxle.metaopt.test_automl_dataclasses import (
    ALL_DATACLASSES, HYPERPARAMS_DIMS, HYPERPARAMS_DIMS_WILDCARDS, SOME_CLIENT_DATACLASS,
    SOME_METRIC_NAME, SOME_PROJECT_DATACLASS, SOME_ROUND_DATACLASS, SOME_TRIAL_DATACLASS)


def test_project_report_to_clients_with_best_scores_df():
    pr = ProjectReport(SOME_PROJECT_DATACLASS)

    df = pr.to_clients_with_best_scores_df()

    assert ClientReport.CLIENT_ID_COLUMN_NAME in df.columns


def test_client_report_to_rounds_with_best_scores_df():
    cr = ClientReport(SOME_CLIENT_DATACLASS)

    df = cr.to_rounds_with_best_scores_df()

    assert RoundReport.ROUND_ID_COLUMN_NAME in df.columns


def test_round_dc_to_scatterplot_df():
    rr = RoundReport(SOME_ROUND_DATACLASS)

    df = rr.to_round_scatterplot_df(SOME_METRIC_NAME, HYPERPARAMS_DIMS_WILDCARDS)

    assert SOME_METRIC_NAME in df.columns
    assert TrialReport.TRIAL_ID_COLUMN_NAME in df.columns
    for d in HYPERPARAMS_DIMS_WILDCARDS:
        assert d in df.columns


def test_round_dc_to_scores_over_time_df():
    rr = RoundReport(SOME_ROUND_DATACLASS)

    df = rr.to_scores_over_time_df(SOME_METRIC_NAME, HYPERPARAMS_DIMS_WILDCARDS)

    assert SOME_METRIC_NAME in df.columns
    assert TrialReport.TRIAL_ID_COLUMN_NAME in df.columns
    assert MetricResultsReport.EPOCH_COLUMN_NAME in df.columns
    for d in HYPERPARAMS_DIMS_WILDCARDS:
        assert d in df.columns


def test_round_metric_names():
    rr = RoundReport(SOME_ROUND_DATACLASS)

    assert rr.get_metric_names() == [SOME_METRIC_NAME]


@pytest.mark.parametrize("discard_singles,expected_hp_dims", ([False, HYPERPARAMS_DIMS_WILDCARDS], [True, []]))
def test_round_hp_wildcards_scenario(discard_singles: bool, expected_hp_dims: List[str]):
    rr = RoundReport(SOME_ROUND_DATACLASS)

    hp_wildcards = rr.list_hyperparameters_wildcards(discard_singles=discard_singles)

    assert hp_wildcards == expected_hp_dims


@pytest.mark.parametrize('dc', ALL_DATACLASSES[1:])
def test_reports_has_sufficient_dc_info(dc: BaseDataclass):
    r: dataclass_2_report[dc.__class__] = BaseReport.from_dc(dc)
    df = r.info_df()

    assert len(dc.to_dict()) - 3 == len(df.index), (
        f"Dataclass dc={dc} should have rows for each attribute that isn't the "
        f"class name, id, or subdataclasses collections. Got df={df.to_string()}."
    )


def test_trial_report_to_scores_over_time_df():
    tr = TrialReport(SOME_TRIAL_DATACLASS)

    df = tr.to_scores_over_time_df(SOME_METRIC_NAME)

    assert TrialSplitReport.TRIAL_SPLIT_ID_COLUMN_NAME in df.columns
    assert MetricResultsReport.EPOCH_COLUMN_NAME in df.columns
    assert MetricResultsReport.TRAIN_VAL_COLUMN_NAME in df.columns
    assert SOME_METRIC_NAME in df.columns
