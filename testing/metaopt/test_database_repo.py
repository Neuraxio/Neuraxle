# from neuraxle.metaopt.data.db_hp_repo import DatabaseHyperparamRepository

import os
from importlib import resources

import pytest
from sqlalchemy import (Column, ForeignKey, Integer, String, Table, and_, TEXT,
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
