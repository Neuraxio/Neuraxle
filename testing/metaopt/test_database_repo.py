
import os

import pytest
from neuraxle.metaopt.data.db_hp_repo import (Base, ClientNode,
                                              DatabaseHyperparamRepository,
                                              DataClassNode, ProjectNode,
                                              ScopedLocationTreeNode,
                                              SQLLiteHyperparamsRepository)
from neuraxle.metaopt.data.vanilla import (DEFAULT_CLIENT, DEFAULT_PROJECT,
                                           ClientDataclass, ProjectDataclass,
                                           RootDataclass, ScopedLocation)
from sqlalchemy import (TEXT, Column, ForeignKey, Integer, String, Table, and_,
                        create_engine)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import backref, relationship, sessionmaker
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.schema import Sequence
from sqlalchemy.sql import asc, desc, func
from testing.metaopt.test_automl_dataclasses import (ALL_DATACLASSES,
                                                     SOME_CLIENT_DATACLASS,
                                                     SOME_FULL_SCOPED_LOCATION,
                                                     SOME_PROJECT_DATACLASS,
                                                     SOME_ROOT_DATACLASS,
                                                     SOME_ROUND_DATACLASS)


def get_sqlite_session_with_root(tmpdir):
    sqlite_filepath = os.path.join(tmpdir, "sqlite.db")
    engine = create_engine(f"sqlite:///{sqlite_filepath}", echo=True, future=True)
    Session = sessionmaker()
    Session.configure(bind=engine)
    session = Session()
    Base.metadata.create_all(engine)
    session.commit()

    root_dcn = DataClassNode(RootDataclass())
    root = ScopedLocationTreeNode(root_dcn, None)
    session.add(root)

    session.commit()
    return session, root


def test_sqlalchemy_sqllite_nodes_star_shema_joins(tmpdir):
    session, root = get_sqlite_session_with_root(tmpdir)

    def_proj = ProjectNode(ProjectDataclass(project_name="def_proj"))
    project = ScopedLocationTreeNode(def_proj, parent=root)
    session.add(project)
    session.commit()

    def_client = ClientNode(ClientDataclass(client_name="def_client"))
    client = ScopedLocationTreeNode(def_client, parent=project)
    session.add(client)
    session.commit()

    session.expunge_all()
    q = session.query(
        ScopedLocationTreeNode.project_name, ScopedLocationTreeNode.client_name
    )
    assert q[0] == (None, None)
    assert q[1] == ("def_proj", None)
    assert q[2] == ("def_proj", "def_client")


def test_root_db_node_can_be_queried(tmpdir):
    session = get_sqlite_session_with_root(tmpdir)[0]

    root_tree_node = session.query(ScopedLocationTreeNode).filter(
        and_(*[
            getattr(ScopedLocationTreeNode, attr) == None
            for attr in ScopedLocation.__dataclass_fields__
        ])
    ).one()

    assert root_tree_node.project_name is None
    assert root_tree_node.client_name is None
    assert root_tree_node.round_number is None


@pytest.mark.parametrize("deep", [True, False])
def test_can_use_sqlite_db_repo_to_save_and_load_and_overwrite_simple_project(tmpdir, deep):
    repo = SQLLiteHyperparamsRepository(tmpdir)
    project: ProjectDataclass = SOME_PROJECT_DATACLASS
    project_loc = ScopedLocation.default().at_dc(project)

    repo.save(project, project_loc, deep=deep)
    project_reloaded = repo.load(project_loc, deep=deep)
    repo.save(project_reloaded, project_loc, deep=deep)
