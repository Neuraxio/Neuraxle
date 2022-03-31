# from neuraxle.metaopt.data.db_hp_repo import DatabaseHyperparamRepository

import os

from neuraxle.metaopt.data.vanilla import (DEFAULT_CLIENT, DEFAULT_PROJECT,
                                           ClientDataclass, ProjectDataclass,
                                           RootDataclass, to_json)
from sqlalchemy import (TEXT, Column, ForeignKey, Integer, String, Table, and_,
                        create_engine)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import backref, relationship, sessionmaker
from sqlalchemy.sql import asc, desc, func


def test_sqlalchemy_sqllite(tmpdir):

    Base = declarative_base()

    project_client = Table(
        "project_round",
        Base.metadata,
        Column("project_name", String, ForeignKey("project.project_name")),
        Column("client_name", String, ForeignKey("client.client_name")),
    )

    # client_round = Table(
    #     "client_round",
    #     Base.metadata,
    #     Column("client_name", String, ForeignKey("client.client_name")),
    #     Column("round_number", Integer, ForeignKey("round.round_number")),
    # )

    class Project(Base):
        __tablename__ = "project"
        project_name = Column(String, primary_key=True)
        clients = relationship(
            "Client", secondary=project_client  # , back_populates="parent_project"
        )

    class Client(Base):
        __tablename__ = "client"
        project_name = Column(String, primary_key=True)
        client_name = Column(String, primary_key=True)
        # rounds = relationship(
        #     "Round", secondary=client_round, back_populates="parent_client"
        # )

    # class Round(Base):
    #     __tablename__ = "round"
    #     round_number = Column(Integer, primary_key=True)
    #     main_metric_name = Column(String)
    #
    #     # projects = relationship(
    #     #     "Project", secondary=project_client, back_populates="rounds"
    #     # )
    #     # clients = relationship(
    #     #     "Client", secondary=client_round, back_populates="rounds"
    #     # )

    sqlite_filepath = os.path.join(tmpdir, "sqlite.db")
    engine = create_engine(f"sqlite:///{sqlite_filepath}")
    Session = sessionmaker()
    Session.configure(bind=engine)
    session = Session()
    Base.metadata.create_all(engine)

    session.commit()
    client = Client(project_name="default_project", client_name="default_client")
    session.add(client)
    session.commit()
    proj = Project(project_name="default_project")
    proj.clients.append(client)
    session.add(proj)
    session.commit()

    q = session.query(
        Project.project_name, Client.client_name
    )
    print(q)
    # q = session.query(
    #     Project.project_name, Client.client_name,  # func.count(Round.main_metric_name).label("metrics_count")
    # ).join(Project.clients)  # .order_by(asc("metrics_count"))
    # print(q)


def test_sqlalchemy_sqllite_starshema(tmpdir):

    Base = declarative_base()

    class AutoDataTable(Base):
        __tablename__ = "autodatatable"

        project_name = Column(String, primary_key=True, nullable=True, default=None)  # DEFAULT_PROJECT
        client_name = Column(String, primary_key=True, nullable=True, default=None)  # DEFAULT_CLIENT
        round_number = Column(Integer, primary_key=True, nullable=True, default=None)
        trial_number = Column(Integer, primary_key=True, nullable=True, default=None)
        split_number = Column(Integer, primary_key=True, nullable=True, default=None)
        metric_name = Column(String, primary_key=True, nullable=True, default=None)

        dataclass = Column(String)

    sqlite_filepath = os.path.join(tmpdir, "sqlite.db")
    engine = create_engine(f"sqlite:///{sqlite_filepath}")
    Session = sessionmaker()
    Session.configure(bind=engine)
    session = Session()
    Base.metadata.create_all(engine)
    session.commit()

    # root = AutoDataTable()
    # root.dataclass = to_json(RootDataclass().shallow())
    # session.add(root)
    # session.commit()

    proj = AutoDataTable(
        project_name="default_project",
    )
    proj.dataclass = to_json(ProjectDataclass().shallow())
    session.add(proj)
    session.commit()

    client = AutoDataTable(
        project_name="default_project",
        client_name="default_client",
    )
    client.dataclass = to_json(ClientDataclass().shallow())
    session.add(client)
    session.commit()

    q = session.query(
        AutoDataTable.project_name, AutoDataTable.client_name
    )
    print(q)
    # q = session.query(
    #     Project.project_name, Client.client_name,  # func.count(Round.main_metric_name).label("metrics_count")
    # ).join(Project.clients)  # .order_by(asc("metrics_count"))
    # print(q)
