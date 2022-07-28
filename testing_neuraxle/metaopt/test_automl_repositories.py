import copy
from typing import Callable, List

import pytest
from neuraxle.base import CX
from neuraxle.metaopt.context import AutoMLContext
from neuraxle.metaopt.data.aggregates import Client, Project, Root
from neuraxle.metaopt.data.vanilla import (DEFAULT_CLIENT, DEFAULT_PROJECT, BaseDataclass, ClientDataclass,
                                           ProjectDataclass, RootDataclass, RoundDataclass, ScopedLocation)
from neuraxle.metaopt.repositories.db import SQLLiteHyperparamsRepository
from neuraxle.metaopt.repositories.json import HyperparamsOnDiskRepository
from neuraxle.metaopt.repositories.repo import HyperparamsRepository
from testing_neuraxle.metaopt.test_automl_dataclasses import (ALL_DATACLASSES, SOME_CLIENT_DATACLASS,
                                                              SOME_FULL_SCOPED_LOCATION, SOME_PROJECT_DATACLASS,
                                                              SOME_ROOT_DATACLASS, SOME_ROUND_DATACLASS)

TmpDir = str


def vanilla_repo_ctor(tmpdir: TmpDir = None) -> AutoMLContext:
    return AutoMLContext.from_context(CX())


def disk_repo_ctor(tmpdir: TmpDir = None) -> AutoMLContext:
    cx = CX()
    tmpdir = tmpdir or cx.get_new_cache_folder()
    return AutoMLContext.from_context(cx, repo=HyperparamsOnDiskRepository(tmpdir))


def db_repo_ctor(tmpdir: TmpDir = None) -> AutoMLContext:
    cx = CX()
    tmpdir = tmpdir or cx.get_new_cache_folder()
    return AutoMLContext.from_context(cx, repo=SQLLiteHyperparamsRepository(tmpdir))


CX_WITH_REPO_CTORS: List[Callable[[TmpDir], AutoMLContext]] = [vanilla_repo_ctor, disk_repo_ctor, db_repo_ctor]


@pytest.mark.parametrize('cx_repo_ctor', CX_WITH_REPO_CTORS)
def test_auto_ml_context_loc_stays_the_same(
    tmpdir, cx_repo_ctor: Callable[[TmpDir], AutoMLContext]
):
    context = cx_repo_ctor(tmpdir)

    c0 = context.push_attr(RootDataclass())
    c1 = c0.push_attr(ProjectDataclass(project_name=DEFAULT_PROJECT))
    c2 = c1.push_attr(ClientDataclass(client_name=DEFAULT_CLIENT))

    assert c0.loc.as_list() == []
    assert c1.loc.as_list() == [DEFAULT_PROJECT]
    assert c2.loc.as_list() == [DEFAULT_PROJECT, DEFAULT_CLIENT]


@pytest.mark.parametrize('cx_repo_ctor', CX_WITH_REPO_CTORS)
def test_context_changes_independently_once_copied(
    tmpdir, cx_repo_ctor: Callable[[TmpDir], AutoMLContext]
):
    cx = cx_repo_ctor(tmpdir)

    copied_cx: AutoMLContext = cx._copy().push_attr(
        ProjectDataclass(project_name=DEFAULT_PROJECT))

    assert copied_cx.loc.as_list() == [DEFAULT_PROJECT]
    assert cx.loc.as_list() == []


@pytest.mark.parametrize('cx_repo_ctor', CX_WITH_REPO_CTORS)
def test_logger_level_works_with_multiple_levels(
    tmpdir, cx_repo_ctor: Callable[[TmpDir], AutoMLContext]
):
    cx = AutoMLContext.from_context(cx_repo_ctor(tmpdir))

    _root: Root = Root.from_context(cx)
    with _root.get_project(DEFAULT_PROJECT) as _proj:
        _proj: Project = _proj
        with _proj.get_client(DEFAULT_CLIENT) as _client:
            _client: Client = _client

            _proj.flow.log(">_proj.flow.log: begin")
            _client.flow.log(">_client.flow.log: begin")
            _client.flow.log(">_client.flow.log: some work being done from within _client")
            _client.flow.log(">_client.flow.log: end")
            _proj.flow.log(">_proj.flow.log: end")

            l0: str = _proj.context.read_scoped_log()
            l1: str = _client.context.read_scoped_log()

            assert l0 != l1
            assert len(l0) > len(l1)
            assert ">_proj" in l0
            assert ">_proj" not in l1
            assert ">_client" in l0
            assert ">_client" in l1
            assert _client.loc != _proj.loc
            assert _client.loc != _root.loc


@pytest.mark.parametrize('cx_repo_ctor', CX_WITH_REPO_CTORS)
def test_hyperparams_repository_has_default_client_project(
    tmpdir, cx_repo_ctor: Callable[[TmpDir], AutoMLContext]
):
    cx = cx_repo_ctor(tmpdir)
    loc: ScopedLocation = ScopedLocation.default().with_dc(SOME_ROUND_DATACLASS)

    restored_client = cx.repo.load(loc.popped(), deep=True)
    restored_project = cx.repo.load(loc.popped().popped(), deep=True)

    cx.repo.save(copy.deepcopy(SOME_ROUND_DATACLASS), scope=loc, deep=True)
    restored_round = cx.repo.load(loc, deep=True)

    assert restored_round == SOME_ROUND_DATACLASS
    assert restored_project.shallow() == SOME_PROJECT_DATACLASS.shallow()
    assert restored_client.get_id() == SOME_CLIENT_DATACLASS.get_id()


@pytest.mark.parametrize('cx_repo_ctor', CX_WITH_REPO_CTORS)
@pytest.mark.parametrize('_dataclass', ALL_DATACLASSES)
def test_hyperparams_repository_loads_stored_scoped_info(
    tmpdir, cx_repo_ctor: Callable[[str], AutoMLContext], _dataclass: BaseDataclass
):
    loc: ScopedLocation = SOME_FULL_SCOPED_LOCATION.at_dc(_dataclass)
    cx: AutoMLContext = cx_repo_ctor(tmpdir).with_loc(loc)
    repo: HyperparamsRepository = cx.repo
    repo.save(copy.deepcopy(SOME_ROOT_DATACLASS), scope=ScopedLocation(), deep=True)

    restored_dataclass_deep = repo.load(loc, deep=True)
    restored_dataclass_shallow = repo.load(loc, deep=False)

    assert restored_dataclass_deep == _dataclass
    assert restored_dataclass_shallow == _dataclass.shallow()


@pytest.mark.parametrize('cx_repo_ctor', CX_WITH_REPO_CTORS)
@pytest.mark.parametrize('_dataclass', ALL_DATACLASSES[1:])
def test_hyperparams_repository_saves_subsequent_data(
    tmpdir, cx_repo_ctor: Callable[[TmpDir], AutoMLContext], _dataclass: BaseDataclass
):
    loc: ScopedLocation = SOME_FULL_SCOPED_LOCATION.at_dc(_dataclass)
    cx: AutoMLContext = cx_repo_ctor(tmpdir).with_loc(loc)
    repo: HyperparamsRepository = cx.repo
    repo.save(copy.deepcopy(SOME_ROOT_DATACLASS), scope=ScopedLocation(), deep=True)
    old_id = _dataclass.get_id()
    next_id = old_id + 1 if isinstance(old_id, int) else old_id + "_next"
    next_dataclass = copy.deepcopy(_dataclass).set_id(next_id)
    next_loc = loc.popped().with_dc(next_dataclass)

    repo.save(next_dataclass, scope=next_loc, deep=False)

    restored_next_dataclass_empty = repo.load(next_loc, deep=True)
    assert restored_next_dataclass_empty == next_dataclass.empty()

    restored_next_dataclass_empty = repo.load(next_loc, deep=False)
    assert restored_next_dataclass_empty == next_dataclass.empty()

    repo.save(next_dataclass, scope=next_loc, deep=True)

    restored_dataclass_shallow = repo.load(next_loc, deep=False)
    assert restored_dataclass_shallow == next_dataclass.shallow()

    restored_dataclass_deep = repo.load(next_loc, deep=True)
    assert restored_dataclass_deep == next_dataclass


@pytest.mark.parametrize('cx_repo_ctor', CX_WITH_REPO_CTORS)
def test_shallow_save_isnt_deep_upon_load(
    tmpdir, cx_repo_ctor: Callable[[TmpDir], AutoMLContext]
):
    zero_loc: ScopedLocation = SOME_FULL_SCOPED_LOCATION.at_dc(SOME_ROUND_DATACLASS)
    cx: AutoMLContext = cx_repo_ctor(tmpdir).with_loc(zero_loc)
    repo: HyperparamsRepository = cx.repo
    next_round_dataclass: RoundDataclass = copy.deepcopy(SOME_ROUND_DATACLASS).set_id(1)
    next_loc = zero_loc.popped().with_dc(next_round_dataclass)

    repo.save(SOME_ROUND_DATACLASS, scope=zero_loc, deep=False)
    repo.save(next_round_dataclass, scope=next_loc, deep=False)
    reloaded_next_round = repo.load(next_loc, deep=True)

    assert reloaded_next_round == next_round_dataclass.empty()
