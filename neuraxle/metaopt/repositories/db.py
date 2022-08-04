"""
Neuraxle's SQLAlchemy Hyperparameter Repository Classes
========================================================
Data objects and related repositories used by AutoML, SQL version.

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
import os
import typing
from datetime import datetime
from typing import Any, Dict, List, Optional, OrderedDict, Type

from neuraxle.base import TrialStatus
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.logging.logging import NeuraxleLogger
from neuraxle.metaopt.data.vanilla import (DEFAULT_CLIENT, DEFAULT_METRIC_NAME,
                                           DEFAULT_PROJECT, BaseDataclass,
                                           BaseTrialDataclassMixin,
                                           ClientDataclass,
                                           MetricResultsDataclass,
                                           ProjectDataclass, RootDataclass,
                                           RoundDataclass, ScopedLocation,
                                           SubDataclassT, TrialDataclass,
                                           TrialSplitDataclass,
                                           dataclass_2_id_attr)
from neuraxle.metaopt.repositories.repo import HyperparamsRepository
from sqlalchemy import (JSON, TEXT, Boolean, Column, DateTime, Float,
                        ForeignKey, Integer, String, and_, create_engine)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import (backref, declarative_mixin, joinedload,
                            relationship, sessionmaker)
from sqlalchemy.orm.session import Session
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
        # collection_class=attribute_mapped_collection("dataclass_node.id_attr_value"),
    )

    def __init__(
        self, dataclass_node: 'DataClassNode', parent: 'ScopedLocationTreeNode' = None
    ):

        if parent is not None:
            self._set_location_from(dataclass_node, parent)

        self.dataclass_node = dataclass_node

        self.parent = parent
        # TODO: parent push self???
        # if parent is not None:
        #    self.parent.subdataclasses.set(self)

    def _set_location_from(self, dataclass_node: 'DataClassNode', parent=None):
        # Set the location of the dataclass node partially from the parent:
        self.project_name = parent.project_name
        self.client_name = parent.client_name
        self.round_number = parent.round_number
        self.trial_number = parent.trial_number
        self.split_number = parent.split_number
        self.metric_name = parent.metric_name

        # Then override the location of the dataclass node:
        dc_klass: Type[BaseDataclass] = sqlalchemy_node_2_dataclass[dataclass_node.__class__]
        id_name: str = dataclass_2_id_attr[dc_klass]
        setattr(self, id_name, str(getattr(dataclass_node, id_name)))

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

    @property
    def id_attr_value(self):
        return self.dataclass_node.id_attr_value

    def to_dataclass(self, deep=False):
        dc: SubDataclassT = self.dataclass_node.to_empty_dataclass()

        if dc.is_terminal_leaf():
            pass
        elif deep:
            items = [
                (v.id_attr_value, v.to_dataclass(deep=deep)) for v in self.subdataclasses
            ]
            dc.set_sublocation_items(items)
        else:
            dc.set_sublocation_keys([i.id_attr_value for i in self.subdataclasses])

        return dc

    # def update_dataclass(self, _dataclass: SubDataclassT, deep: bool):
    #     if not deep:
    #         _dataclass = _dataclass.empty()
    #     self.dataclass_node.update_dataclass(_dataclass)

    # def add_subdataclass(self, _dataclass: SubDataclassT, deep: bool):
    #     if not deep:
    #         _dataclass = _dataclass.empty()
    #     DataClassNode.add_dataclass(_dataclass=_dataclass, parent=self)

    @staticmethod
    def query(session: Session, scope: ScopedLocation, deep=False) -> Optional['ScopedLocationTreeNode']:
        query = session.query(ScopedLocationTreeNode).filter(
            and_(*[
                getattr(ScopedLocationTreeNode, attr) == str(getattr(scope, attr))
                if getattr(scope, attr) is not None else getattr(ScopedLocationTreeNode, attr) == None
                for attr in ScopedLocation.__dataclass_fields__
            ])
        )

        if deep:
            n_levels_deeper_to_fetch = 6 - len(scope)
            if n_levels_deeper_to_fetch > 0:
                jl = joinedload(ScopedLocationTreeNode.subdataclasses)
                for _ in range(n_levels_deeper_to_fetch - 1):
                    jl = jl.joinedload(ScopedLocationTreeNode.subdataclasses)
                query = query.options(jl)

        query = query.all()

        # the return value is optional:
        if len(query) == 0:
            return None  # TODO: create directly instead?
        elif len(query) == 1:
            return query[0]
        else:
            raise ValueError(f"More than one ScopedLocationTreeNode found for the given scope {scope}: {query}")

    def __str__(self):
        return f"<{self.__class__.__name__}({self.loc.as_list()})>"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.loc.as_list()}, ...)>"


class DataClassTreeNodeUpdater:

    def __init__(
        self, session: Session, scope: ScopedLocation, deep: bool = False,
    ):
        self.session = session
        self.scope = scope
        self.deep = deep

    def add_or_update_dataclass(
        self, _dataclass: SubDataclassT, parent: Optional[ScopedLocationTreeNode]
    ):
        """
        Add or update a dataclass in the database.

        Possible scenarios:
        - dataclass is not in the database:
            - add it to the database. Recursively:
                - add all subdataclasses
        - dataclass is in the database:
            - update it in the database. Recursively, for all subdataclasses wheter found or not:
                - add all subdataclasses
                - update all subdataclasses

        This means we have the following methods chaining into each other:
        - add or update dataclass
        - update dataclass
        - add dataclass
        """
        node: ScopedLocationTreeNode = ScopedLocationTreeNode.query(self.session, self.scope, self.deep)
        if node is not None:
            # The dataclass is already in the database. Update it:
            self.update_dataclass(_dataclass, node)
        else:
            # Add the dataclass to the database:
            self.add_dataclass(_dataclass, parent)

    def update_dataclass(
        self, _dataclass: SubDataclassT, node: ScopedLocationTreeNode
    ):
        self.scope = self.scope.with_dc(_dataclass)

        node.dataclass_node._update_attrs_from_dataclass(_dataclass)
        self.session.commit()  # TODO: REMOVE THAT DEBUGGING.

        if self.deep and not _dataclass.is_terminal_leaf():
            _ids_stored: List[str] = [v.id_attr_value for v in node.subdataclasses]
            for sub_dc_key, sub_dc in _dataclass.get_sublocation_items():
                sub_dc_key = str(sub_dc_key)
                if sub_dc_key in _ids_stored:
                    # TODO: what if parent wasn't deep? That is needed to access the subdataclasses.
                    self.update_dataclass(sub_dc, node.subdataclasses[_ids_stored.index(sub_dc_key)])
                else:
                    self.add_dataclass(sub_dc, node)

    def add_dataclass(
        self, _dataclass: SubDataclassT, parent: Optional[ScopedLocationTreeNode]
    ):
        self.scope = self.scope.with_dc(_dataclass)

        # Create a new dataclass node:
        _dc_klass: Type[DataClassNode] = dataclass_2_sqlalchemy_node[_dataclass.__class__]
        dc_node = _dc_klass(_dataclass=_dataclass)

        # Add the dataclass node to the tree's scope:
        parent = parent or ScopedLocationTreeNode.query(self.session, self.scope.popped(), deep=False)
        tree_node = ScopedLocationTreeNode(
            dataclass_node=dc_node,
            parent=parent  # Linked parent makes the new node added to the session automatically.
        )
        self.session.commit()  # TODO: REMOVE THAT DEBUGGING.

        if self.deep and not _dataclass.is_terminal_leaf():
            for sub_dc in _dataclass.get_sublocation_values():
                self.add_dataclass(sub_dc, tree_node)


class DataClassNode(Base):
    __tablename__ = 'dataclassnode'
    # id = Column(String, primary_key=True, nullable=True)
    # id = Column(String, ForeignKey('scopedlocationtreenode.id'), primary_key=True)
    id_attr_value = Column(String, nullable=True)

    type = Column(String(50))

    # tree_id = Column(ForeignKey('scopedlocationtreenode.id'))
    tree_id = Column(String, ForeignKey('scopedlocationtreenode.id'), primary_key=True)
    tree = relationship("ScopedLocationTreeNode", back_populates="dataclass_node")

    logs = relationship("AutoMLLog", back_populates="dataclass_node")

    __mapper_args__ = {
        'polymorphic_identity': 'rootdataclass',
        'polymorphic_on': type
    }

    def __init__(self, _dataclass: BaseDataclass = None):
        if _dataclass is None:
            _dataclass = sqlalchemy_node_2_dataclass[self.__class__]()
        self._update_attrs_from_dataclass(_dataclass)
        self.id_attr_value = str(_dataclass.get_id())

    def _update_attrs_from_dataclass(self, _dataclass: BaseDataclass):
        pass

    def to_empty_dataclass(self) -> RootDataclass:
        return RootDataclass()


class ProjectNode(DataClassNode):
    __tablename__ = 'project'
    id = Column(String, ForeignKey('dataclassnode.tree_id'), primary_key=True)
    project_name = Column(String, nullable=False, default=DEFAULT_PROJECT)

    __mapper_args__ = {
        'polymorphic_identity': 'projectdataclass',
    }

    def _update_attrs_from_dataclass(self, _dataclass: ProjectDataclass):
        self.project_name = _dataclass.project_name

    def to_empty_dataclass(self) -> ProjectDataclass:
        return ProjectDataclass(project_name=self.project_name)


class ClientNode(DataClassNode):
    __tablename__ = 'client'
    id = Column(String, ForeignKey('dataclassnode.tree_id'), primary_key=True)
    client_name = Column(String, nullable=False, default=DEFAULT_CLIENT)

    __mapper_args__ = {
        'polymorphic_identity': 'clientdataclass',
    }

    def _update_attrs_from_dataclass(self, _dataclass: ClientDataclass):
        self.client_name = _dataclass.client_name

    def to_empty_dataclass(self) -> ClientDataclass:
        return ClientDataclass(client_name=self.client_name)


class RoundNode(DataClassNode):
    __tablename__ = 'round'
    # TODO: is it string or int? for all DC Nodes ids.
    id = Column(String, ForeignKey('dataclassnode.tree_id'), primary_key=True)
    round_number = Column(Integer, nullable=False, default=0)
    main_metric_name = Column(String, nullable=True, default=None)

    __mapper_args__ = {
        'polymorphic_identity': 'rounddataclass',
    }

    def _update_attrs_from_dataclass(self, _dataclass: RoundDataclass):
        self.round_number = _dataclass.round_number
        self.main_metric_name = _dataclass.main_metric_name

    def to_empty_dataclass(self) -> RoundDataclass:
        return RoundDataclass(round_number=self.round_number, main_metric_name=self.main_metric_name)


@declarative_mixin
class BaseTrialNodeMixin:
    hyperparams = Column(JSON, nullable=False, default=dict)
    status = Column(String, nullable=False, default=TrialStatus.PLANNED.value)
    created_time = Column(DateTime, nullable=False, default=func.now())
    start_time = Column(DateTime, nullable=True, default=func.now())
    end_time = Column(DateTime, nullable=True)

    def _update_attrs_from_dataclass(self, _dataclass: BaseTrialDataclassMixin):
        self.hyperparams = dict(_dataclass.hyperparams.to_flat_dict())
        self.status = _dataclass.status.value
        self.created_time = _dataclass.created_time
        self.start_time = _dataclass.start_time
        self.end_time = _dataclass.end_time

    def _attrs_for_to_dataclass(self) -> Dict[str, Any]:
        return {
            'hyperparams': HyperparameterSamples(self.hyperparams),
            'status': TrialStatus(self.status),
            'created_time': self.created_time,
            'start_time': self.start_time,
            'end_time': self.end_time,
        }


class TrialNode(BaseTrialNodeMixin, DataClassNode):
    __tablename__ = 'trial'
    id = Column(String, ForeignKey('dataclassnode.tree_id'), primary_key=True)
    trial_number = Column(Integer, nullable=False, default=0)

    __mapper_args__ = {
        'polymorphic_identity': 'trialdataclass',
    }

    def _update_attrs_from_dataclass(self, _dataclass: TrialDataclass):
        BaseTrialNodeMixin._update_attrs_from_dataclass(self, _dataclass=_dataclass)
        self.trial_number = _dataclass.trial_number

    def to_empty_dataclass(self) -> TrialDataclass:
        return TrialDataclass(
            trial_number=self.trial_number, retrained_split=None,
            **self._attrs_for_to_dataclass()
        )


class TrialSplitNode(BaseTrialNodeMixin, DataClassNode):
    __tablename__ = 'trialsplit'
    id = Column(String, ForeignKey('dataclassnode.tree_id'), primary_key=True)
    split_number = Column(Integer, nullable=False, default=0)

    __mapper_args__ = {
        'polymorphic_identity': 'trialsplitdataclass',
    }

    def _update_attrs_from_dataclass(self, _dataclass: TrialSplitDataclass):
        BaseTrialNodeMixin._update_attrs_from_dataclass(self, _dataclass=_dataclass)
        self.split_number = _dataclass.split_number

    def to_empty_dataclass(self) -> TrialSplitDataclass:
        return TrialSplitDataclass(
            split_number=self.split_number, **self._attrs_for_to_dataclass())


class MetricResultsNode(DataClassNode):
    __tablename__ = 'metricresults'
    id = Column(String, ForeignKey('dataclassnode.tree_id'), primary_key=True)
    metric_name = Column(String, nullable=False, default=DEFAULT_METRIC_NAME)

    valid_values = relationship("ValidMetricResultsValues", back_populates="metric_results_node")
    train_values = relationship("TrainMetricResultsValues", back_populates="metric_results_node")

    higher_score_is_better = Column(Boolean, nullable=False, default=True)

    __mapper_args__ = {
        'polymorphic_identity': 'metricresultsdataclass',
    }

    def _update_attrs_from_dataclass(self, _dataclass: MetricResultsDataclass):
        self.metric_name = _dataclass.metric_name

        # Extending the lists with new values from metric results value tables.
        start: int = len(self.valid_values)
        self.valid_values.extend([
            ValidMetricResultsValues(value=v)
            for v in _dataclass.validation_values[start:]
        ])

        start: int = len(self.train_values)
        self.train_values.extend([
            TrainMetricResultsValues(value=v)
            for v in _dataclass.train_values[start:]
        ])

        self.higher_score_is_better = _dataclass.higher_score_is_better

    def to_empty_dataclass(self) -> MetricResultsDataclass:
        return MetricResultsDataclass(
            metric_name=self.metric_name,
            validation_values=[v.value for v in self.valid_values],
            train_values=[v.value for v in self.train_values],
            higher_score_is_better=self.higher_score_is_better
        )


@declarative_mixin
class MetricResultsValuesMixin:
    id = Column(Integer, Sequence('id_seq', metadata=Base.metadata), primary_key=True, index=True)

    value = Column(Float, nullable=False)
    # TODO: add datetime to metric results dataclass?
    datetime = Column(DateTime, default=func.now())

    def __init__(self, value: float, _datetime: datetime = None):
        self.value = value
        if _datetime is None:
            _datetime = datetime.now()
        self.datetime = _datetime


class TrainMetricResultsValues(MetricResultsValuesMixin, Base):
    __tablename__ = 'trainmetricresultsvalues'
    metric_results_node_id = Column(String, ForeignKey('dataclassnode.tree_id'))
    metric_results_node = relationship("MetricResultsNode", back_populates="train_values", uselist=False)


class ValidMetricResultsValues(MetricResultsValuesMixin, Base):
    __tablename__ = 'validmetricresultsvalues'
    metric_results_node_id = Column(String, ForeignKey('dataclassnode.tree_id'))
    metric_results_node = relationship("MetricResultsNode", back_populates="valid_values", uselist=False)


dataclass_2_sqlalchemy_node: typing.OrderedDict[Type[BaseDataclass], Type[DataClassNode]] = OrderedDict([
    (RootDataclass, DataClassNode),
    (ClientDataclass, ClientNode),
    (ProjectDataclass, ProjectNode),
    (RoundDataclass, RoundNode),
    (TrialDataclass, TrialNode),
    (TrialSplitDataclass, TrialSplitNode),
    (MetricResultsDataclass, MetricResultsNode),
])

sqlalchemy_node_2_dataclass: typing.OrderedDict[Type[BaseDataclass], Type[DataClassNode]] = OrderedDict([
    (v, k) for k, v in dataclass_2_sqlalchemy_node.items()
])


class AutoMLLog(Base):
    __tablename__ = 'automllog'
    id = Column(Integer, Sequence('id_seq', metadata=Base.metadata), primary_key=True, index=True)
    dataclass_node_id = Column(String, ForeignKey('dataclassnode.tree_id'))
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
        HyperparamsRepository.__init__(self)
        _DatabaseLoggerHandlerMixin.__init__(self)
        self.engine = engine
        self.session = session

    def create_db(self) -> 'DatabaseHyperparamRepository':
        Base.metadata.create_all(self.engine)
        _root_dc = RootDataclass()
        root_dcn = ScopedLocationTreeNode(DataClassNode(_root_dc), parent=None)
        self.session.add(root_dcn)
        self.session.commit()
        self.save(_root_dc, ScopedLocation(), deep=True)
        return self

    def load(self, scope: ScopedLocation, deep=False) -> SubDataclassT:

        # Get DataClassNode by parsing the tree with scopedlocation attributes:
        node: ScopedLocationTreeNode = ScopedLocationTreeNode.query(self.session, scope, deep=deep)

        if node is not None:
            return node.to_dataclass(deep=deep)
        else:
            return scope.new_dataclass_from_id()

    def save(self, _dataclass: SubDataclassT, scope: ScopedLocation, deep=False) -> 'HyperparamsRepository':
        try:

            DataClassTreeNodeUpdater(self.session, scope, deep).add_or_update_dataclass(_dataclass, parent=None)

            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e


class SQLLiteHyperparamsRepository(DatabaseHyperparamRepository):
    def __init__(self, sqllite_db_path, echo=False):

        sqlite_filepath = os.path.join(sqllite_db_path, "sqlite.db")
        os.makedirs(sqllite_db_path, exist_ok=True)
        engine = create_engine(f"sqlite:///{sqlite_filepath}", echo=echo, future=True)

        _Session = sessionmaker()
        _Session.configure(bind=engine)
        session = _Session()

        DatabaseHyperparamRepository.__init__(self, engine, session)
        self.create_db()


class PostGreSQLHyperparamsRepository(DatabaseHyperparamRepository):

    def __init__(self, postgresql_db_path, echo=False):
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
