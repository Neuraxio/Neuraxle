import copy
from typing import List, Type

import pytest
from neuraxle.base import TrialStatus, synchroneous_flow_method
from neuraxle.metaopt.data.vanilla import (DEFAULT_CLIENT, DEFAULT_PROJECT,
                                           NULL_CLIENT, NULL_PROJECT,
                                           NULL_ROUND, NULL_TRIAL,
                                           NULL_TRIAL_SPLIT,
                                           RETRAIN_TRIAL_SPLIT_ID,
                                           BaseDataclass, ClientDataclass,
                                           MetricResultsDataclass,
                                           ProjectDataclass, RootDataclass,
                                           RoundDataclass, ScopedLocation,
                                           TrialDataclass, TrialSplitDataclass,
                                           as_named_odict, dataclass_2_id_attr,
                                           from_json, to_json)

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


@pytest.mark.parametrize("cp", [copy.copy, copy.deepcopy])
def test_scoped_location_can_copy_and_change(cp):
    sl = ScopedLocation(DEFAULT_PROJECT, DEFAULT_CLIENT, 0, 0, 0, SOME_METRIC_NAME)

    sl_copy = cp(sl)
    sl_copy.pop()
    sl_copy.pop()

    assert sl_copy != sl
    assert len(sl_copy) == len(sl) - 2


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
    root_restored_dc = from_json(root_as_json)

    assert SOME_METRIC_RESULTS_DATACLASS == root_restored_dc[SOME_FULL_SCOPED_LOCATION]
    assert root == root_restored_dc


def test_trial_dataclass_can_store_and_contains_retrain_split():
    tc: TrialDataclass = TrialDataclass(trial_number=5)
    tc.store(TrialSplitDataclass(split_number=0))
    tc.store(TrialSplitDataclass(split_number=1))
    tc.store(TrialSplitDataclass(split_number=2))
    sl: ScopedLocation = ScopedLocation.default(
        round_number=0, trial_number=5, split_number=RETRAIN_TRIAL_SPLIT_ID)
    tsc: TrialSplitDataclass = TrialSplitDataclass(split_number=RETRAIN_TRIAL_SPLIT_ID)

    tc.store(tsc)

    assert len(tc) == 3
    assert sl in tc
    assert tsc.get_id() == RETRAIN_TRIAL_SPLIT_ID
    assert tc.retrained_split == tsc
    assert tc[sl] == tc.retrained_split


def test_fill_to_dc():
    nl = ScopedLocation().fill_to_dc(SOME_METRIC_RESULTS_DATACLASS)

    assert nl == ScopedLocation(
        NULL_PROJECT,
        NULL_CLIENT,
        NULL_ROUND,
        NULL_TRIAL,
        NULL_TRIAL_SPLIT,
        SOME_METRIC_RESULTS_DATACLASS.get_id(),
    )


def test_dataclass_tree():
    base_scope = ScopedLocation()
    expected_result: List[ScopedLocation] = []
    for dc in ALL_DATACLASSES:
        base_scope = base_scope.fill_to_dc(dc)
        expected_result.append(base_scope)

    tree = SOME_ROOT_DATACLASS.tree()

    assert tree == expected_result
