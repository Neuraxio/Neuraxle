"""
Neuraxle's SQLAlchemy Hyperparameter Repository Classes
=================================================
Data objects and related repositories used by AutoML, SQL version.

..
    Copyright 2021, Neuraxio Inc.

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
import os
import typing
from dataclasses import dataclass
from typing import List, OrderedDict, Type

from neuraxle.base import TrialStatus
from neuraxle.logging.logging import NeuraxleLogger
from neuraxle.metaopt.data.aggregates import Round, Trial
from neuraxle.metaopt.data.vanilla import (DEFAULT_CLIENT, DEFAULT_PROJECT,
                                           BaseDataclass, BaseDataclassT,
                                           ClientDataclass,
                                           HyperparamsRepository,
                                           ProjectDataclass, RootDataclass,
                                           ScopedLocation, SubDataclassT,
                                           dataclass_2_id_attr, to_json)
from sqlalchemy import (TEXT, Boolean, Column, DateTime, ForeignKey, Integer,
                        MetaData, PickleType, String, Table, and_,
                        create_engine)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import backref, relationship, sessionmaker
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.schema import Sequence
from sqlalchemy.sql import asc, desc, func

Base = declarative_base()


class ScopedLocationTreeNode(Base):
    __tablename__ = "scopedlocationtreenode"

    id = Column(Integer, Sequence('id_seq', metadata=Base.metadata), primary_key=True, index=True)
    # TODO: id to be a foreing key in the node.
    parent_id = Column(Integer, ForeignKey(id))

    project_name = Column(String, nullable=True, index=True, default=None)  # DEFAULT_PROJECT
    client_name = Column(String, nullable=True, index=True, default=None)  # DEFAULT_CLIENT
    round_number = Column(Integer, nullable=True, index=True, default=None)
    trial_number = Column(Integer, nullable=True, index=True, default=None)
    split_number = Column(Integer, nullable=True, index=True, default=None)
    metric_name = Column(String, nullable=True, index=True, default=None)

    dataclass_node = relationship("DataClassNode", back_populates="tree", uselist=False)

    # Children:
    subdataclasses = relationship(
        "ScopedLocationTreeNode",
        backref=backref("parent", remote_side=id),
        collection_class=attribute_mapped_collection("dataclass_node"),
    )

    def __init__(self, dataclass_node: 'DataClassNode', parent=None):
        self.parent = parent

        if parent is not None:
            self._set_location_from(dataclass_node, parent)

        self.dataclass_node = dataclass_node

    def _set_location_from(self, dataclass_node: 'DataClassNode', parent=None):
        # Set the location of the dataclass node partially from the parent:
        self.project_name = parent.project_name
        self.client_name = parent.client_name
        self.round_number = parent.round_number
        self.trial_number = parent.trial_number
        self.split_number = parent.split_number
        self.metric_name = parent.metric_name

        # Then override the location of the dataclass node:
        dc_klass: Type[BaseDataclass] = sqlalchemy_node_2_dataclass[dataclass_node]
        id_name: str = dataclass_2_id_attr[dc_klass]
        setattr(self, id_name, getattr(dataclass_node, id_name))

    @property
    def loc(self):
        return ScopedLocation(
            project_name=self.project_name,
            client_name=self.client_name,
            round_number=self.round_number,
            trial_number=self.trial_number,
            split_number=self.split_number,
            metric_name=self.metric_name,
        )

    def to_dataclass(self, deep=False):
        dc: BaseDataclassT = self.dataclass_node.to_empty_dataclass()

        # TODO: int and str keys review.
        keys = self.subdataclasses.keys()
        dc.set_sublocation_keys(keys)
        if deep:
            sub_dcs = self.subdataclasses.values()
            dc.set_sublocation_values(sub_dcs)
        return dc

    def update_dataclass(self, _dataclass: BaseDataclassT, deep: bool):
        self.dataclass_node.update_dataclass(_dataclass)
        if deep:
            raise NotImplementedError("TODO: figure out how to deep update nodes.")

    def add_dataclass(self, _dataclass: BaseDataclassT):
        new_dataclass_node = ScopedLocationTreeNode(
            dataclass_node=DataClassNode.from_dataclass(_dataclass),
            parent=self
        )
        self.session.add(new_dataclass_node)


class DataClassNode(Base):
    __tablename__ = 'dataclassnode'
    id = Column(String, primary_key=True, nullable=True)

    type = Column(String(50))

    tree_id = Column(ForeignKey('scopedlocationtreenode.id'))
    tree = relationship("ScopedLocationTreeNode", back_populates="dataclass_node")

    logs = relationship("AutoMLLog", back_populates="dataclass_node")

    __mapper_args__ = {
        'polymorphic_identity': 'rootdataclass',
        'polymorphic_on': type
    }

    @staticmethod
    def from_dataclass(_dataclass: BaseDataclass):
        _dc_klass: type = dataclass_2_sqlalchemy_node[_dataclass.__class__]
        dc_node = _dc_klass(id=None, _dataclass=_dataclass)
        return dc_node

    def __init__(self, id=None, _dataclass: BaseDataclass = None):
        self.id = str(id)

    def update_dataclass(self, _dataclass: RootDataclass):
        pass

    def to_empty_dataclass(self) -> RootDataclass:
        return RootDataclass()


class ProjectNode(DataClassNode):
    __tablename__ = 'project'
    id = Column(String, ForeignKey('dataclassnode.id'), primary_key=True)
    project_name = Column(String, nullable=True, default=None)

    __mapper_args__ = {
        'polymorphic_identity': 'projectdataclass',
    }

    def __init__(self, id=None, _dataclass: ProjectDataclass = None):
        if _dataclass is None:
            _dataclass = ProjectDataclass()
        DataClassNode.__init__(self, id=id, _dataclass=_dataclass)
        self.project_name = _dataclass.project_name

    def update_dataclass(self, _dataclass: ProjectDataclass):
        self.project_name = _dataclass.project_name

    def to_empty_dataclass(self) -> ProjectDataclass:
        return ProjectDataclass(project_name=self.project_name)


class ClientNode(DataClassNode):
    __tablename__ = 'client'
    id = Column(String, ForeignKey('dataclassnode.id'), primary_key=True)
    client_name = Column(String)

    __mapper_args__ = {
        'polymorphic_identity': 'clientdataclass',
    }

    def __init__(self, id=None, _dataclass: ClientDataclass = None):
        if _dataclass is None:
            _dataclass = ClientDataclass()
        DataClassNode.__init__(self, id=id, _dataclass=_dataclass)
        self.client_name = _dataclass.client_name

    def to_empty_dataclass(self) -> ClientDataclass:
        return ClientDataclass(client_name=self.client_name)


dataclass_2_sqlalchemy_node: typing.OrderedDict[Type[BaseDataclass], Type[DataClassNode]] = OrderedDict([
    (RootDataclass, DataClassNode),
    (ClientDataclass, ClientNode),
    (ProjectDataclass, ProjectNode),
    # RoundDataclass: RoundNode,
    # TrialDataclass: TrialNode,
    # TrialSplitDataclass: SplitNode,
    # MetricResultDataclass: MetricResultNode,
])

sqlalchemy_node_2_dataclass: typing.OrderedDict[Type[BaseDataclass], Type[DataClassNode]] = OrderedDict([
    (v, k) for k, v in dataclass_2_sqlalchemy_node.items()
])


class AutoMLLog(Base):
    __tablename__ = 'automllog'
    id = Column(Integer, Sequence('id_seq', metadata=Base.metadata), primary_key=True, index=True)
    dataclass_node_id = Column(String, ForeignKey('dataclassnode.id'))
    dataclass_node = relationship("DataClassNode", back_populates="logs", uselist=False)
    datetime = Column(DateTime, default=func.now())

    log_text = Column(TEXT)


class _DatabaseLoggerHandlerMixin:
    """
    Mixin to add a in-memory logging handler to a repository.
    """

    def add_logging_handler(self, logger: NeuraxleLogger, scope: ScopedLocation) -> 'HyperparamsRepository':
        # TODO: use AutoMLLog
        return self

    def get_log_from_logging_handler(self, logger: NeuraxleLogger, scope: ScopedLocation) -> str:
        # TODO: use AutoMLLog
        return logger.get_scoped_string_history()


class DatabaseHyperparamRepository(_DatabaseLoggerHandlerMixin, HyperparamsRepository):
    def __init__(self, engine, session):
        self.engine = engine
        self.session = session

    def create_db(self):
        Base.metadata.create_all(self.engine)
        root_dcn = DataClassNode(id="root")
        self.session.add(root_dcn)
        self.session.commit()

    def load(self, scope: ScopedLocation, deep=False) -> SubDataclassT:

        # Get DataClassNode by parsing the tree with scopedlocation attributes:
        query = self._build_scoped_query(scope)

        if deep:
            n_levels_deeper_to_fetch = 6 - len(scope)
            for i in range(n_levels_deeper_to_fetch):
                query = query.joinedload(ScopedLocationTreeNode.subdataclasses)

        tree_node: ScopedLocationTreeNode = query.one()
        if tree_node is None:
            raise ValueError(f"No data found for {scope}")

        return tree_node.to_dataclass(deep=deep)

    def save(self, _dataclass: SubDataclassT, scope: ScopedLocation, deep=False) -> 'HyperparamsRepository':
        try:
            query = self._build_scoped_query(scope)
            if len(query) > 0:
                # Node exists, then update it:
                tree_node: ScopedLocationTreeNode = query.one()
                tree_node.update_dataclass(_dataclass, deep=deep)

            else:
                # Get parent to attach child by parsing the tree with popped scopedlocation attributes
                parent_tree_node: ScopedLocationTreeNode = self._build_scoped_query(scope.popped()).one()
                parent_tree_node.add_dataclass(_dataclass, deep=deep)

            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e

    def _build_scoped_query(self, scope):
        return self.session.query(ScopedLocationTreeNode).filter(
            and_(*[
                getattr(ScopedLocationTreeNode, attr) == str(getattr(scope, attr))
                if getattr(scope, attr) is not None else getattr(ScopedLocationTreeNode, attr) == None
                for attr in ScopedLocation.__dataclass_fields__
            ])
        )


class SQLLiteHyperparamsRepository(DatabaseHyperparamRepository):
    def __init__(self, sqllite_db_path, echo=True):

        sqlite_filepath = os.path.join(sqllite_db_path, "sqlite.db")
        engine = create_engine(f"sqlite:///{sqlite_filepath}", echo=echo, future=True)

        Session = sessionmaker()
        Session.configure(bind=engine)
        session = Session()

        super().__init__(engine, session)
        self.create_db()


class PostGreSQLHyperparamsRepository(DatabaseHyperparamRepository):

    def __init__(self, postgresql_db_path, echo=True):
        raise NotImplementedError("TODO: implement this.")

    def get_database_path(self, user: str, password: str, host: str, dialect: str, driver: str = ''):
        """
        :param user:
        :param password:
        :param dialect: e.g. mysql
        :param driver: (Optional) e.g. "pymysql"
        :return:
        """
        # TODO: implement this.
        return "dialect[+driver]://user:password@host/dbname"
