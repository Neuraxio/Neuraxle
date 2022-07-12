"""
Neuraxle's AutoML Metric Reporting classes.
=====================================================

Classes are splitted like this for the metric analysis:

- ProjectReport
- ClientReport
- RoundReport
- TrialReport
- TrialSplitReport

These AutoML reports are used to get information from the nested dataclasses, such as to create visuals.

..
    Copyright 2022, Neuraxio Inc.

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
import json
import pprint
import typing
from collections import OrderedDict, defaultdict
from numbers import Number
from typing import (Any, Dict, Generic, Iterable, List, Optional, Tuple, Type,
                    TypeVar, Union)

import numpy as np
import pandas as pd
from neuraxle.base import TrialStatus
from neuraxle.hyperparams.space import (FlatDict, HyperparameterSamples,
                                        RecursiveDict)
from neuraxle.metaopt.data.vanilla import (BaseDataclass, ClientDataclass,
                                           MetricResultsDataclass,
                                           ProjectDataclass, RootDataclass,
                                           RoundDataclass, ScopedLocationAttr,
                                           ScopedLocationAttrInt,
                                           SubDataclassT, TrialDataclass,
                                           TrialSplitDataclass,
                                           dataclass_2_id_attr, from_json,
                                           to_json)

SubReportT = TypeVar('SubReportT', bound=Optional['BaseReport'])


def _filter_df_hps(wildcarded_hps: FlatDict, wildcards_to_keep: List[str] = None) -> FlatDict:
    """
    Filters the hyperparameters so as to keep only the indicated wildcards to keep.
    Also parses the hyperparameters to strings if not numeric.
    """
    def _to_str_if_not_number(value: Any) -> Union[str, Number]:
        if isinstance(value, Number) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            return repr(value)
        return str(value)
    return {
        k: _to_str_if_not_number(v) for k, v in wildcarded_hps.items()
        if wildcards_to_keep is None or k in wildcards_to_keep
    }


class BaseReport(Generic[SubReportT, SubDataclassT]):
    """
    A report contains a dataclass of the same subclass-level of itself, so as to be able to
    dig into the dataclass so as to observe it, such as to generate statistics and query its information.
    The dataclasses represent the results of an AutoML optimization round, even multiple rounds.
    These AutoML reports are used to get information from the nested dataclasses, such as to create visuals.

    .. seealso::
        :class:`neuraxle.metaopt.data.vanilla.BaseDataclass`
        :class:`neuraxle.metaopt.data.reporting.BaseAggregate`
    """
    def __init__(self, dc: SubDataclassT):
        self._dataclass: SubDataclassT = dc

    @staticmethod
    def from_dc(dc: SubDataclassT) -> SubReportT:
        return dataclass_2_report[dc.__class__](dc)

    @staticmethod
    def from_json(json_dc_data: str) -> SubReportT:
        return BaseReport.from_dc(from_json(json_dc_data))

    def get_id(self) -> ScopedLocationAttr:
        return self._dataclass.get_id()

    def info_df(self) -> pd.DataFrame:
        _json: str = to_json(self._dataclass.empty())

        info: dict = json.loads(_json)
        del info['__type__']
        del info[self._dataclass._sublocation_attr_name]
        del info[self._dataclass._id_attr_name]
        info = {k: pprint.pformat(v) if not isinstance(v, dict) else pprint.pformat(
            dict(HyperparameterSamples(v).to_flat_dict(use_wildcards=True))) for k, v in info.items()}

        column_names = ['Attribute', 'Value']
        if len(info) == 0:
            df = pd.DataFrame(columns=column_names)
        else:
            df = pd.DataFrame.from_records([info], columns=list(info.keys()))
            df = df.transpose()
            df = df.reset_index(level=0)
            df.columns = column_names
        return df

    def __len__(self) -> int:
        return len(self._dataclass.get_sublocation())

    def __iter__(self) -> Iterable[SubReportT]:
        for subdataclass in self._dataclass.get_sublocation_values():
            if subdataclass is not None:
                yield self.subreport(subdataclass)

    def __getitem__(self, item: ScopedLocationAttr) -> SubReportT:
        """
        Get trial at the given index.

        :param item: trial index
        :return:
        """
        subdataclass = self._dataclass.get_sublocation()[item]
        return self.subreport(subdataclass)

    def subreport(self, _dataclass: SubDataclassT) -> SubReportT:
        return report_2_subreport[self.__class__](_dataclass)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        subobjs = str([str(t) for t in self.__iter__()])
        return (
            f"{self.__class__.__name__}("
            f"id={self._dataclass.get_id()}, {subobjs}"
            f")"
        )

    def __eq__(self, other: object) -> bool:
        return self._dataclass == other._dataclass


class RootReport(BaseReport['ProjectReport', RootDataclass]):
    pass


class ProjectReport(BaseReport['ClientReport', ProjectDataclass]):

    def to_clients_with_best_scores_df(self) -> pd.DataFrame:
        """
        Get a dataframe with the info of each last round for each client.
        """
        df_data = []
        for client_report in self:
            client_report: ClientReport = client_report
            last_round_report: RoundReport = client_report[-1]

            last_best_result_summary: Tuple[float, ScopedLocationAttrInt, TrialStatus,
                                            FlatDict] = last_round_report.best_result_summary(use_wildcards=True)
            main_metric_name = last_round_report.main_metric_name
            df_data.append({
                ClientReport.CLIENT_ID_COLUMN_NAME: client_report.get_id(),
                f"most recent {RoundReport.ROUND_ID_COLUMN_NAME}": last_round_report.get_id(),
                'n trials for most recent round': len(last_round_report),
                'main metric name': main_metric_name,
                'best score for main metric': last_best_result_summary[0],
                'best trial status': last_best_result_summary[2].value,
                'best hyperparameters': pprint.pformat(last_best_result_summary[3])
            })

        return pd.DataFrame(df_data)


class ClientReport(BaseReport['RoundReport', ClientDataclass]):
    CLIENT_ID_COLUMN_NAME = dataclass_2_id_attr[ClientDataclass]

    def to_rounds_with_best_scores_df(self) -> pd.DataFrame:
        """
        Get a dataframe with the best scores for each round as well as the metric used for that round.
        """
        df_data = []
        for round_report in self:
            round_report: RoundReport = round_report
            best_result_summary: Tuple[float, ScopedLocationAttrInt,
                                       TrialStatus, FlatDict] = round_report.best_result_summary(use_wildcards=True)
            main_metric_name = round_report.main_metric_name
            df_data.append({
                RoundReport.ROUND_ID_COLUMN_NAME: round_report.get_id(),
                'n trials': len(round_report),
                'main metric name': main_metric_name,
                'best score for main metric': best_result_summary[0],
                'best trial status': best_result_summary[2].value,
                'best hyperparameters': pprint.pformat(best_result_summary[3])
            })

        return pd.DataFrame(df_data)


class RoundReport(BaseReport['TrialReport', RoundDataclass]):
    ROUND_ID_COLUMN_NAME = dataclass_2_id_attr[RoundDataclass]
    SUMMARY_STATUS_COLUMNS_NAME = 'status'

    @property
    def main_metric_name(self) -> str:
        return self._dataclass.main_metric_name

    def get_best_trial(self, metric_name: str = None) -> Optional['TrialReport']:
        """
        Return trial report with best score from all trials, provided that this trial has a score and was successful.
        """
        metric_name = metric_name or self.main_metric_name
        best_trial_id = self.get_best_trial_id(metric_name)
        if best_trial_id is None:
            return None
        return self[best_trial_id]

    def get_best_trial_id(self, metric_name: str = None) -> Optional[ScopedLocationAttr]:
        """
        Get best trial id from all trials. Will return None if there are no successful trial with such score.
        """
        metric_name = metric_name or self.main_metric_name
        best_score, best_trial_id = None, None
        _is_higher_score_better = self.is_higher_score_better(metric_name)

        for trial_report in self:
            trial_report: TrialReport = trial_report

            trial_score = trial_report.get_avg_validation_score(metric_name)

            _has_better_score = best_score is None or (
                trial_score is not None and trial_report.is_success() and (
                    _is_higher_score_better == (trial_score > best_score)
                )
            )

            if _has_better_score:
                best_score = trial_score
                best_trial_id = trial_report.get_id()

        return best_trial_id

    def get_best_hyperparams(self, metric_name: str = None) -> HyperparameterSamples:
        """
        Get best hyperparams from all trials.
        """
        if metric_name is None:
            metric_name = self.main_metric_name
        best_trial_report = self.get_best_trial(metric_name)
        if best_trial_report is None:
            return HyperparameterSamples()
        return best_trial_report.get_hyperparams()

    def is_higher_score_better(self, metric_name: str = None) -> bool:
        """
        Return true if higher score is better. If metric_name is None, the optimizer's
        metric is taken.
        """
        metric_name = metric_name or self.main_metric_name
        if metric_name is None:
            raise ValueError("metric name and main metric name are not defined.")

        if len(self) == 0:
            return ValueError("No trial found, cannot determine if higher score is better.")

        all_metric_names: List[List[str]] = [t.get_metric_names() for t in self]
        if metric_name not in sum(all_metric_names, []):
            raise ValueError(
                f"No trial found with the metric {metric_name}. "
                f"Trials for round {self.get_id()} have the "
                f"respective following metrics: {all_metric_names}")

        for last_with_score in reversed(range(len(self))):
            if metric_name in all_metric_names[last_with_score]:
                break

        last_trial_with_score = self[last_with_score]
        return last_trial_with_score.is_higher_score_better(metric_name)

    def get_n_val_splits(self):
        """
        Finds the number of validation splits on record in this round's first trial.
        """
        if len(self) > 0:
            return len(self[0])
        return 0

    def get_metric_names(self) -> List[str]:
        """
        Get the name of all metrics on record.
        """
        _metrics = []
        for i in self:
            i: TrialReport = i
            _ms = i.get_metric_names()
            for m in _ms:
                if m not in _metrics:
                    _metrics.append(m)
        if self.main_metric_name in _metrics:
            _metrics.remove(self.main_metric_name)
            _metrics.insert(0, self.main_metric_name)
        return _metrics

    def best_result_summary(self, metric_name: str = None, use_wildcards: bool = False) -> Tuple[float, ScopedLocationAttrInt, TrialStatus, FlatDict]:
        """
        Return the best result summary for the given metric, as the `[score, trial_number, hyperparams_flat_dict]`.
        """
        metric_name = metric_name or self.main_metric_name
        return self.summary(metric_name, use_wildcards)[0]

    def summary(
        self, metric_name: str = None, use_wildcards: bool = False
    ) -> List[Tuple[float, ScopedLocationAttrInt, FlatDict]]:
        """
        Get a summary of the round. Best score is first.
        Values in the returned triplet tuples are: (score, trial_number, hyperparams),
        sorted by score such that the best score is first.
        """
        metric_name = metric_name or self.main_metric_name
        results: List[float, ScopedLocationAttrInt, TrialStatus, FlatDict] = list()
        results_not_success: List[float, ScopedLocationAttrInt, TrialStatus, FlatDict] = list()

        for trial in self:
            trial: TrialReport = trial
            score = trial.get_avg_validation_score(metric_name)
            trial_number = trial._dataclass.trial_number
            hp = trial.get_hyperparams().to_flat_dict(use_wildcards=use_wildcards)
            status = trial.get_status()

            _resultslist = results
            if not status == TrialStatus.SUCCESS:
                _resultslist = results_not_success
            _resultslist.append((score, trial_number, status, hp))

        is_reverse: bool = self.is_higher_score_better(metric_name)
        results = list(sorted(results, reverse=is_reverse))
        results_not_success = list(sorted(results_not_success, reverse=is_reverse))
        return results + results_not_success

    def get_all_hyperparams(self, as_flat: bool = True, use_wildcards: bool = False) -> List[FlatDict]:
        """
        Get the list of hyperparams for all trials.
        """
        if use_wildcards and not as_flat:
            raise ValueError("Cannot use wildcards with non-flat hyperparams.")

        hyperparams: List[FlatDict] = list()
        for trial in self:
            hps = trial.get_hyperparams()
            if as_flat:
                hps = hps.to_flat_dict(use_wildcards=use_wildcards)
            hyperparams.append(hps)

        return hyperparams

    def list_hyperparameters_wildcards(self, discard_singles=False) -> List[str]:
        """
        Returns a list of all the hyperparameters wildcards used in the round.
        Discarding singles would prune out the hyperparameters with values that never vary.
        """
        trial_reports: List[TrialReport] = list(self)
        trials_hps = [t.get_hyperparams().to_wildcards() for t in trial_reports]
        trials_hps_keys = set()
        trials_hps_values = defaultdict(set)
        for hps in trials_hps:
            for hp, hp_val in hps.items():
                trials_hps_keys.add(hp)
                trials_hps_values[hp].add(hp_val)
        if discard_singles:
            trials_hps_keys_step_2 = [k for k, v in trials_hps_values.items() if len(v) > 1]
            return trials_hps_keys_step_2
        return list(trials_hps_keys)

    def list_successful_avg_validation_scores(self) -> List[float]:
        """
        Returns a list of all the average validation scores on record, only for succesful trials.
        """
        return [t.get_avg_validation_score(self.main_metric_name) for t in self.successful_trials]

    @property
    def successful_trials(self) -> List['TrialReport']:
        """
        Returns a list of all the succesful trials on record.
        """
        return [t for t in self if t.is_success()]

    def to_round_scatterplot_df(self, metric_name: str = None, wildcards_to_keep: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns a dataframe with trial ids, the selected metric, and the wildcarded hyperparameters to keep.
        """
        metric_name = metric_name or self.main_metric_name
        summary: List[float, ScopedLocationAttrInt, FlatDict] = self.summary(metric_name, use_wildcards=True)

        splom_df = pd.DataFrame([
            {**{TrialReport.TRIAL_ID_COLUMN_NAME: trial_id}, **{self.SUMMARY_STATUS_COLUMNS_NAME: status.value}, **{metric_name: score},
                **_filter_df_hps(hyperparams, wildcards_to_keep)}
            for score, trial_id, status, hyperparams in summary
        ])
        splom_df.set_index(TrialReport.TRIAL_ID_COLUMN_NAME)

        return splom_df

    def to_scores_over_time_df(self, metric_name: str = None, wildcards_to_keep: List[str] = None) -> pd.DataFrame:
        """
        Returns a dataframe with trial ids, epochs, the selected metric, and the wildcarded hyperparameters to keep.
        """
        metric_name = metric_name or self.main_metric_name
        _df_list: List[Dict(str, Any)] = []

        trials: List[TrialReport] = list(self)
        for trial in trials:
            scores_over_time: List[float] = trial.get_avg_validation_score(metric_name, over_time=True)
            if scores_over_time is None:
                continue
            hps = _filter_df_hps(trial.get_hyperparams().to_wildcards(), wildcards_to_keep)
            for epoch, s in enumerate(scores_over_time):
                _df_list.append({
                    TrialReport.TRIAL_ID_COLUMN_NAME: trial._dataclass.get_id(),
                    MetricResultsReport.EPOCH_COLUMN_NAME: epoch,
                    metric_name: s,
                    **hps,
                })

        df = pd.DataFrame(_df_list)
        df.set_index(TrialReport.TRIAL_ID_COLUMN_NAME)

        return df


class TrialReport(BaseReport['TrialSplitReport', TrialDataclass]):
    TRIAL_ID_COLUMN_NAME = dataclass_2_id_attr[TrialDataclass]

    def get_metric_names(self) -> List[str]:
        """
        Get the name of all metrics on record.
        """
        _metrics = []
        for i in self:
            i: TrialSplitReport = i
            _ms = i.get_metric_names()
            for m in _ms:
                if m not in _metrics:
                    _metrics.append(m)
        return _metrics

    def get_hyperparams(self) -> RecursiveDict:
        """
        Get the hyperparameters of the trial.
        """
        return self._dataclass.hyperparams

    def is_success(self):
        """
        Checks if the trial is successful from its dataclass record.
        """
        return self._dataclass.status == TrialStatus.SUCCESS

    def get_status(self) -> TrialStatus:
        """
        Get the status of the trial.
        """
        return self._dataclass.status

    def are_all_splits_successful(self) -> bool:
        """
        Return true if all splits are successful.
        """
        return all(i.is_success() for i in self)

    def are_all_splits_failures(self) -> bool:
        """
        Return true if all splits are failed.
        """
        return all(not i.is_success() for i in self)

    def get_avg_validation_score(self, metric_name: str, over_time=False) -> Optional[Union[float, List[float]]]:
        """
        Returns the average score for all validation splits's
        best validation score for the specified scoring metric.

        : param metric_name: The name of the metric to use.
        : param over_time: If true, return all the avg scores over time instead of the best avg score.
        : return: validation score
        """
        if self.is_success():
            scores = [
                (
                    val_split[metric_name].get_valid_scores()  # List[float]
                    if over_time is True else
                    val_split[metric_name].get_best_validation_score()  # best float
                )
                for val_split in self
                if val_split.is_success() and metric_name in val_split.get_metric_names()
            ]
            scores = [s for s in scores if s is not None]
            return np.mean(scores, axis=0) if len(scores) > 0 else None

    def get_avg_n_epoch_to_best_validation_score(self, metric_name: str) -> Optional[float]:
        if metric_name not in self._dataclass.get_sublocation_keys():
            return None

        n_epochs = [
            val_split[metric_name].get_n_epochs_to_best_validation_score()
            for val_split in self
            if val_split.is_success() and metric_name in val_split.get_metric_names()
        ]

        n_epochs = sum(n_epochs) / len(n_epochs) if len(n_epochs) > 0 else None
        return n_epochs

    def is_higher_score_better(self, metric_name: str) -> bool:
        return self[-1].is_higher_score_better(metric_name)

    def to_scores_over_time_df(self, metric_name: str) -> pd.DataFrame:
        """
        Returns a dataframe with the trial id, the epoch, and the selected metric.
        """
        _df_list: List[Dict(str, Any)] = []
        for trial_split in self:
            trial_split: TrialSplitReport = trial_split

            _split_df_list = trial_split._get_scores_over_time_train_val_df_list(metric_name)

            _df_list.extend(_split_df_list)
        df = pd.DataFrame(_df_list)
        df.set_index(TrialSplitReport.TRIAL_SPLIT_ID_COLUMN_NAME)
        return df


class TrialSplitReport(BaseReport['MetricResultsReport', TrialSplitDataclass]):
    TRIAL_SPLIT_ID_COLUMN_NAME = dataclass_2_id_attr[TrialSplitDataclass]

    def get_hyperparams(self) -> RecursiveDict:
        """
        Get the hyperparameters of the trial.
        """
        return self._dataclass.hyperparams

    def get_metric_names(self) -> List[str]:
        """
        List metric names that are the subdataclass' keys.
        """
        return list(self._dataclass.metric_results.keys())

    def is_success(self):
        """
        Set trial status to success.
        """
        return self._dataclass.status == TrialStatus.SUCCESS

    def is_higher_score_better(self, metric_name: str) -> bool:
        return self[metric_name].is_higher_score_better()

    def to_scores_over_time_df(self, metric_name: str) -> pd.DataFrame:
        """
        Returns a dataframe with the trial id, the epoch, and the selected metric.
        """
        _df_list = self._get_scores_over_time_train_val_df_list(metric_name)
        df = pd.DataFrame(_df_list)
        df.set_index(TrialSplitReport.TRIAL_SPLIT_ID_COLUMN_NAME)
        return df

    def _get_scores_over_time_train_val_df_list(self, metric_name: str) -> List[Dict[str, Any]]:
        _df_list: List[Dict(str, Any)] = []

        train_scores_over_time: List[float] = self[metric_name].get_train_scores()
        if train_scores_over_time is None:
            return []
        for epoch, s in enumerate(train_scores_over_time):
            _df_list.append({
                TrialSplitReport.TRIAL_SPLIT_ID_COLUMN_NAME: self.get_id(),
                MetricResultsReport.EPOCH_COLUMN_NAME: epoch,
                MetricResultsReport.TRAIN_VAL_COLUMN_NAME: 'train',
                metric_name: s,
            })

        valid_scores_over_time: List[float] = self[metric_name].get_valid_scores()
        if valid_scores_over_time is None:
            return _df_list
        for epoch, s in enumerate(valid_scores_over_time):
            _df_list.append({
                TrialSplitReport.TRIAL_SPLIT_ID_COLUMN_NAME: self.get_id(),
                MetricResultsReport.EPOCH_COLUMN_NAME: epoch,
                MetricResultsReport.TRAIN_VAL_COLUMN_NAME: 'validation',
                metric_name: s,
            })

        return _df_list


class MetricResultsReport(BaseReport[None, MetricResultsDataclass]):
    METRIC_COLUMN_NAME = "metric"
    EPOCH_COLUMN_NAME = "epoch"
    TRAIN_VAL_COLUMN_NAME = "phase"

    @property
    def metric_name(self) -> str:
        return self._dataclass.metric_name

    def get_train_scores(self) -> List[float]:
        """
        Return the train scores' values.
        """
        return self._dataclass.train_values

    def get_valid_scores(self) -> List[float]:
        """
        Return the validation scores for the given scoring metric.
        """
        return self._dataclass.validation_values

    def get_final_validation_score(self) -> float:
        """
        Return the latest validation score for the given scoring metric.
        """
        return self.get_valid_scores()[-1]

    def get_best_validation_score(self) -> Optional[float]:
        """
        Return the best validation score for the given scoring metric.
        """
        scores = self.get_valid_scores()
        if len(scores) == 0:
            return None

        if self.is_higher_score_better():
            f = np.max
        else:
            f = np.min

        return f(scores)

    def get_n_epochs_to_best_validation_score(self) -> Optional[int]:
        """
        Return the number of epochs
        """
        scores = self.get_valid_scores()
        if len(scores) == 0:
            return None

        if self.is_higher_score_better():
            f = np.argmax
        else:
            f = np.argmin

        return f(scores)

    def is_higher_score_better(self) -> bool:
        """
        Return True if higher scores are better for the main metric.
        """
        return self._dataclass.higher_score_is_better

    def is_new_best_score(self) -> bool:
        """
        Return True if the latest validation score is the new best score.
        """
        if self.get_best_validation_score() in self.get_valid_scores()[:-1]:
            return False
        return True


report_2_subreport: typing.OrderedDict[Type[BaseReport], Type[BaseReport]] = OrderedDict([
    (RootReport, ProjectReport),
    (ProjectReport, ClientReport),
    (ClientReport, RoundReport),
    (RoundReport, TrialReport),
    (TrialReport, TrialSplitReport),
    (TrialSplitReport, MetricResultsReport),
    (MetricResultsReport, None),
])

report_2_dataclass: typing.OrderedDict[Type[BaseReport], Type[BaseDataclass]] = OrderedDict([
    (RootReport, RootDataclass),
    (ProjectReport, ProjectDataclass),
    (ClientReport, ClientDataclass),
    (RoundReport, RoundDataclass),
    (TrialReport, TrialDataclass),
    (TrialSplitReport, TrialSplitDataclass),
    (MetricResultsReport, MetricResultsDataclass),
])
dataclass_2_report: typing.OrderedDict[Type[BaseDataclass], Type[BaseReport]] = {
    dc: rep
    for rep, dc in report_2_dataclass.items()
}
