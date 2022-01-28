import copy
import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pytest
from neuraxle.base import (BaseService, BaseStep, BaseTransformer,
                           CX, Flow, HandleOnlyMixin, Identity,
                           MetaStep, TrialStatus, synchroneous_flow_method)
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.distributions import RandInt, Uniform
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace)
from neuraxle.metaopt.auto_ml import ControlledAutoML, DefaultLoop, RandomSearch, Trainer
from neuraxle.metaopt.callbacks import (CallbackList, EarlyStoppingCallback,
                                        MetricCallback)
from neuraxle.metaopt.data.aggregates import (Client, Project, Root, Round,
                                              Trial, TrialSplit)
from neuraxle.metaopt.data.json_repo import HyperparamsJSONRepository
from neuraxle.metaopt.data.vanilla import (DEFAULT_CLIENT, DEFAULT_PROJECT,
                                           AutoMLContext, BaseDataclass,
                                           BaseHyperparameterOptimizer,
                                           ClientDataclass, HyperparamsRepository,
                                           MetricResultsDataclass,
                                           ProjectDataclass, RootDataclass,
                                           RoundDataclass, ScopedLocation,
                                           TrialDataclass, TrialSplitDataclass,
                                           VanillaHyperparamsRepository,
                                           as_named_odict, dataclass_2_id_attr,
                                           from_json, to_json)
from neuraxle.metaopt.validation import (GridExplorationSampler,
                                         ValidationSplitter)
from neuraxle.pipeline import Pipeline
from neuraxle.steps.data import DataShuffler
from neuraxle.steps.flow import TrainOnlyWrapper
from neuraxle.steps.numpy import AddN, MultiplyByN
from sklearn.metrics import median_absolute_error
from testing.test_step_saving import ROOT

SOME_METRIC_NAME = 'MAE'
HYPERPARAMS = {'learning_rate': 0.01}


SOME_METRIC_RESULTS_DATACLASS = MetricResultsDataclass(
    metric_name=SOME_METRIC_NAME,
    validation_values=[3, 2, 1],
    train_values=[2, 1, 0],
    higher_score_is_better=False,
)
SOME_TRIAL_DATACLASS = TrialDataclass(
    trial_number=0,
    hyperparams=HYPERPARAMS,
).start()
SOME_TRIAL_SPLIT_DATACLASS = TrialSplitDataclass(
    split_number=0,
    metric_results=as_named_odict(SOME_METRIC_RESULTS_DATACLASS),
    hyperparams=HYPERPARAMS,
).start()
SOME_TRIAL_SPLIT_DATACLASS.end(TrialStatus.SUCCESS)
SOME_TRIAL_DATACLASS.store(SOME_TRIAL_SPLIT_DATACLASS)
SOME_TRIAL_DATACLASS.end(TrialStatus.SUCCESS)
SOME_ROUND_DATACLASS = RoundDataclass(
    round_number=0,
    trials=[SOME_TRIAL_DATACLASS],
    main_metric_name=SOME_METRIC_NAME,
)
SOME_CLIENT_DATACLASS = ClientDataclass(
    client_name=DEFAULT_CLIENT,
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

ALL_DATACLASSES = [
    SOME_ROOT_DATACLASS,
    SOME_PROJECT_DATACLASS,
    SOME_CLIENT_DATACLASS,
    SOME_ROUND_DATACLASS,
    SOME_TRIAL_DATACLASS,
    SOME_TRIAL_SPLIT_DATACLASS,
    SOME_METRIC_RESULTS_DATACLASS,
]


@pytest.mark.parametrize("scope_slice_len", list(range(len(ALL_DATACLASSES))))
def test_dataclass_getters(scope_slice_len: int):
    expected_dataclass: BaseDataclass = ALL_DATACLASSES[scope_slice_len]
    dataclass_type: Type[BaseDataclass] = expected_dataclass.__class__
    sliced_scope = SOME_FULL_SCOPED_LOCATION[:dataclass_type]
    assert len(sliced_scope) == scope_slice_len

    sliced_scope = SOME_FULL_SCOPED_LOCATION[:scope_slice_len]
    assert len(sliced_scope) == scope_slice_len

    dc = SOME_ROOT_DATACLASS[sliced_scope]

    assert isinstance(dc, dataclass_type)
    assert dc.get_id() == expected_dataclass.get_id()
    assert dc.get_id() == sliced_scope[scope_slice_len - 1]
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


def test_auto_ml_context_loc_stays_the_same():
    context = CX()
    context = AutoMLContext.from_context(
        context, VanillaHyperparamsRepository(context.get_path()))

    c0 = context.push_attr(RootDataclass())
    c1 = c0.push_attr(ProjectDataclass(project_name=DEFAULT_PROJECT))
    c2 = c1.push_attr(ClientDataclass(client_name=DEFAULT_CLIENT))

    assert c0.loc.as_list() == []
    assert c1.loc.as_list() == [DEFAULT_PROJECT]
    assert c2.loc.as_list() == [DEFAULT_PROJECT, DEFAULT_CLIENT]


def test_context_changes_independently_once_copied():
    cx = CX()
    cx = AutoMLContext.from_context()

    copied_cx: AutoMLContext = cx.copy().push_attr(
        ProjectDataclass(project_name=DEFAULT_PROJECT))

    assert copied_cx.loc.as_list() == [DEFAULT_PROJECT]
    assert cx.loc.as_list() == []


@pytest.mark.parametrize("cp", [copy.copy, copy.deepcopy])
def test_scoped_location_can_copy_and_change(cp):
    sl = ScopedLocation(DEFAULT_PROJECT, DEFAULT_CLIENT, 0, 0, 0, SOME_METRIC_NAME)

    sl_copy = cp(sl)
    sl_copy.pop()
    sl_copy.pop()

    assert sl_copy != sl
    assert len(sl_copy) == len(sl) - 2


def test_logger_level_works_with_multiple_levels():
    c0 = AutoMLContext.from_context()

    try:
        c0.add_scoped_logger_file_handler()
        c0.flow.log("c0.flow.log: begin")

        c1 = c0.push_attr(ProjectDataclass(project_name=DEFAULT_PROJECT))
        try:
            c1.add_scoped_logger_file_handler()
            c1.flow.log("c1.flow.log: begin")

            c1.flow.log("c1.flow.log: some work being done from within c1")

            c1.flow.log("c1.flow.log: end")
        finally:
            c1.free_scoped_logger_file_handler()

        c0.flow.log("c0.flow.log: end")
    finally:
        c0.free_scoped_logger_file_handler()

    l0 = c0.read_scoped_logger_file()
    l1 = c1.read_scoped_logger_file()

    assert l0 != l1
    assert len(l0) > len(l1)
    assert "c0" in l0
    assert "c0" not in l1
    assert "c1" in l0
    assert "c1" in l1
    assert c1.loc != c0.loc


@pytest.mark.parametrize("dataclass_type", list(dataclass_2_id_attr.keys()))
def test_dataclass_id_attr_get_set(dataclass_type):
    _id = 9000
    dc = dataclass_type().set_id(_id)

    assert dc.get_id() == _id


def test_dataclass_from_dict_to_dict():
    root: RootDataclass = SOME_ROOT_DATACLASS

    root_as_dict = root.to_dict()
    root_restored = RootDataclass.from_dict(root_as_dict)

    assert SOME_METRIC_RESULTS_DATACLASS == root_restored[SOME_FULL_SCOPED_LOCATION]
    assert root == root_restored


def test_dataclass_from_json_to_json():
    root: RootDataclass = SOME_ROOT_DATACLASS

    root_as_dict = root.to_dict()
    root_as_json = to_json(root_as_dict)
    root_restored_dict = from_json(root_as_json)
    root_restored_dc = RootDataclass.from_dict(root_restored_dict)

    assert SOME_METRIC_RESULTS_DATACLASS == root_restored_dc[SOME_FULL_SCOPED_LOCATION]
    assert root == root_restored_dc


def test_hyperparams_repository_has_default_client_project():
    cx = AutoMLContext.from_context()
    loc: ScopedLocation = ScopedLocation.default().with_dc(SOME_ROUND_DATACLASS)

    restored_client = cx.repo.load(loc.popped(), deep=True)
    restored_project = cx.repo.load(loc.popped().popped(), deep=True)

    cx.repo.save(copy.deepcopy(SOME_ROUND_DATACLASS), scope=loc, deep=True)
    restored_round = cx.repo.load(loc, deep=True)

    assert restored_round == SOME_ROUND_DATACLASS
    assert restored_project.shallow() == SOME_PROJECT_DATACLASS.shallow()
    assert restored_client.get_id() == SOME_CLIENT_DATACLASS.get_id()


def vanilla_repo_ctor(tmpdir) -> AutoMLContext:
    return AutoMLContext.from_context(CX())


def json_repo_ctor(tmpdir) -> AutoMLContext:
    # TODO: add this function to REPO_CTORS
    return AutoMLContext.from_context(CX(), repo=HyperparamsJSONRepository(tmpdir))


REPO_CTORS = [vanilla_repo_ctor]


@pytest.mark.parametrize('repo_ctor', REPO_CTORS)
@pytest.mark.parametrize('_dataclass', ALL_DATACLASSES)
def test_hyperparams_repository_loads_stored_scoped_info(
    tmpdir, repo_ctor: Callable[[str], AutoMLContext], _dataclass: BaseDataclass
):
    loc: ScopedLocation = SOME_FULL_SCOPED_LOCATION.at_dc(_dataclass)
    cx: AutoMLContext = repo_ctor(tmpdir).with_loc(loc)
    repo: HyperparamsRepository = cx.repo
    repo.save(copy.deepcopy(SOME_ROOT_DATACLASS), scope=loc, deep=True)

    restored_dataclass_deep = repo.load(loc, deep=True)
    restored_dataclass_shallow = repo.load(loc, deep=False)

    assert restored_dataclass_deep == _dataclass
    assert restored_dataclass_shallow == _dataclass.shallow()


@pytest.mark.parametrize('repo_ctor', REPO_CTORS)
@pytest.mark.parametrize('_dataclass', ALL_DATACLASSES[1:])
def test_hyperparams_repository_saves_subsequent_data(
    tmpdir, repo_ctor: Callable[[str], AutoMLContext], _dataclass: BaseDataclass
):
    loc: ScopedLocation = SOME_FULL_SCOPED_LOCATION.at_dc(_dataclass)
    cx: AutoMLContext = repo_ctor(tmpdir).with_loc(loc)
    repo: HyperparamsRepository = cx.repo
    repo.save(copy.deepcopy(SOME_ROOT_DATACLASS), scope=loc, deep=True)
    old_id = _dataclass.get_id()
    new_id = old_id + 1 if isinstance(old_id, int) else old_id + "_next"
    new_dataclass = copy.deepcopy(_dataclass).set_id(new_id)
    new_loc = loc.popped().with_dc(new_dataclass)

    repo.save(new_dataclass, scope=new_loc, deep=False)

    restored_dataclass_empty = repo.load(new_loc, deep=True)
    assert restored_dataclass_empty == new_dataclass.empty()

    restored_dataclass_empty = repo.load(new_loc, deep=False)
    assert restored_dataclass_empty == new_dataclass.empty()

    repo.save(new_dataclass, scope=new_loc, deep=True)

    restored_dataclass_shallow = repo.load(new_loc, deep=False)
    assert restored_dataclass_shallow == new_dataclass.shallow()

    restored_dataclass_deep = repo.load(new_loc, deep=True)
    assert restored_dataclass_deep == new_dataclass
