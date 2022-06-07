
from neuraxle.metaopt.data.reporting import (MetricResultsReport, RoundReport,
                                             TrialReport)
from testing_neuraxle.metaopt.test_automl_dataclasses import (
    HYPERPARAMS_DIMS, SOME_METRIC_NAME, SOME_ROUND_DATACLASS)


def test_round_dc_to_scatterplot_df():
    rr = RoundReport(SOME_ROUND_DATACLASS)

    df = rr.to_round_scatterplot_df(SOME_METRIC_NAME, HYPERPARAMS_DIMS)

    assert SOME_METRIC_NAME in df.columns
    assert TrialReport.TRIAL_ID_COLUMN_NAME in df.columns
    for d in HYPERPARAMS_DIMS:
        assert d in df.columns


def test_round_dc_to_scores_over_time_df():
    rr = RoundReport(SOME_ROUND_DATACLASS)

    df = rr.to_scores_over_time_df(SOME_METRIC_NAME, HYPERPARAMS_DIMS)

    assert SOME_METRIC_NAME in df.columns
    assert TrialReport.TRIAL_ID_COLUMN_NAME in df.columns
    assert MetricResultsReport.EPOCH_COLUMN_NAME in df.columns
    for d in HYPERPARAMS_DIMS:
        assert d in df.columns
