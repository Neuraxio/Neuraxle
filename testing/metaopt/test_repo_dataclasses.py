import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pytest
from neuraxle.base import (BaseService, BaseStep, BaseTransformer,
                           ExecutionContext, Flow, HandleOnlyMixin, Identity,
                           MetaStep, TrialStatus, synchroneous_flow_method)
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.distributions import RandInt, Uniform
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace)
from neuraxle.metaopt.auto_ml import AutoML, DefaultLoop, RandomSearch, Trainer
from neuraxle.metaopt.callbacks import (CallbackList, EarlyStoppingCallback,
                                        MetricCallback)
from neuraxle.metaopt.data.aggregates import (Client, Project, Root, Round,
                                              Trial, TrialSplit)
from neuraxle.metaopt.data.vanilla import (DEFAULT_CLIENT, DEFAULT_PROJECT,
                                           AutoMLContext, AutoMLFlow,
                                           BaseDataclass,
                                           BaseHyperparameterOptimizer,
                                           ClientDataclass,
                                           MetricResultsDataclass,
                                           ProjectDataclass, RootDataclass,
                                           RoundDataclass, ScopedLocation,
                                           TrialDataclass, TrialSplitDataclass,
                                           VanillaHyperparamsRepository,
                                           as_named_odict, dataclass_2_id_attr)
from neuraxle.metaopt.validation import (GridExplorationSampler,
                                         ValidationSplitter)
from neuraxle.pipeline import Pipeline
from neuraxle.steps.data import DataShuffler
from neuraxle.steps.flow import TrainOnlyWrapper
from neuraxle.steps.numpy import AddN, MultiplyByN
from sklearn.metrics import median_absolute_error

SOME_METRIC_NAME = 'MAE'


BASE_TRIAL_ARGS = {
    "hyperparams": HyperparameterSamples(),
    "status": TrialStatus.RUNNING,
    "created_time": datetime.datetime.now(),
    "start_time": datetime.datetime.now() + datetime.timedelta(minutes=1),
    "end_time": datetime.datetime.now() + datetime.timedelta(days=1),
    "log": "some_log",
}

SOME_METRIC_RESULTS_DATACLASS = MetricResultsDataclass(
    metric_name=SOME_METRIC_NAME,
    validation_values=[3, 2, 1],
    train_values=[2, 1, 0],
    higher_score_is_better=False,
)
SOME_TRIAL_SPLIT_DATACLASS = TrialSplitDataclass(
    split_number=0,
    metric_results=as_named_odict(SOME_METRIC_RESULTS_DATACLASS),
    **BASE_TRIAL_ARGS,
).end(TrialStatus.SUCCESS)
SOME_TRIAL_DATACLASS = TrialDataclass(
    trial_number=0,
    validation_splits=[SOME_TRIAL_SPLIT_DATACLASS],
    **BASE_TRIAL_ARGS,
).end(TrialStatus.SUCCESS)
SOME_ROUND_DATACLASS = RoundDataclass(
    round_number=0,
    trials=[SOME_TRIAL_DATACLASS],
)
SOME_CLIENT_DATACLASS = ClientDataclass(
    client_name=DEFAULT_CLIENT,
    main_metric_name=SOME_METRIC_NAME,
    rounds=[SOME_ROUND_DATACLASS],
)
SOME_PROJECT_DATACLASS = ProjectDataclass(
    project_name=DEFAULT_PROJECT,
    clients=as_named_odict(SOME_CLIENT_DATACLASS),
)
SOME_ROOT_DATACLASS = RootDataclass(
    projects=as_named_odict(SOME_PROJECT_DATACLASS),
)

SOME_FULL_SCOPED_LOCATION: ScopedLocation = ScopedLocation(
    DEFAULT_PROJECT, DEFAULT_CLIENT, 0, 0, 0, SOME_METRIC_NAME
)


@pytest.mark.parametrize("scope_slice_len, expected_dataclass, dataclass_type", [
    (0, SOME_ROOT_DATACLASS, RootDataclass),
    (1, SOME_PROJECT_DATACLASS, ProjectDataclass),
    (2, SOME_CLIENT_DATACLASS, ClientDataclass),
    (3, SOME_ROUND_DATACLASS, RoundDataclass),
    (4, SOME_TRIAL_DATACLASS, TrialDataclass),
    (5, SOME_TRIAL_SPLIT_DATACLASS, TrialSplitDataclass),
    (6, SOME_METRIC_RESULTS_DATACLASS, MetricResultsDataclass),
])
def test_dataclass_getters(
    scope_slice_len: int,
    expected_dataclass: BaseDataclass,
    dataclass_type: Type[BaseDataclass],
):
    sliced_scope = SOME_FULL_SCOPED_LOCATION[:dataclass_type]
    assert (scope_slice_len == 0 and sliced_scope is None) or len(sliced_scope) == scope_slice_len

    sliced_scope = SOME_FULL_SCOPED_LOCATION[:scope_slice_len]
    assert len(sliced_scope) == scope_slice_len

    dc = SOME_ROOT_DATACLASS[sliced_scope]

    assert isinstance(dc, dataclass_type)
    assert dc.get_id() == expected_dataclass.get_id()
    assert dc.get_id() == SOME_FULL_SCOPED_LOCATION[scope_slice_len - 1]
    assert dc.get_id() == SOME_FULL_SCOPED_LOCATION[dataclass_type]
    assert dc == expected_dataclass


@pytest.mark.parametrize("dataclass_type, scope", [
    (ProjectDataclass, ScopedLocation(DEFAULT_PROJECT)),
    (ClientDataclass, ScopedLocation(DEFAULT_PROJECT, DEFAULT_CLIENT)),
])
def test_base_empty_default_dataclass_getters(
    dataclass_type: Type[BaseDataclass],
    scope: ScopedLocation,
):
    root: RootDataclass = RootDataclass()

    dc = root[scope]

    assert isinstance(dc, dataclass_type)
    assert dc.get_id() == scope.as_list()[-1]
