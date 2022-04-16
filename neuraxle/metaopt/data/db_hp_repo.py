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
from email.policy import default
import os
import typing
from dataclasses import dataclass
from typing import List, OrderedDict, Type

from neuraxle.base import TrialStatus
from neuraxle.logging.logging import NeuraxleLogger
from neuraxle.metaopt.data.aggregates import Round, Trial
from neuraxle.metaopt.data.vanilla import (DEFAULT_CLIENT, DEFAULT_METRIC_NAME,
                                           DEFAULT_PROJECT, BaseDataclass,
                                           ClientDataclass,
                                           HyperparamsRepository,
                                           MetricResultsDataclass,
                                           ProjectDataclass, RootDataclass,
                                           RoundDataclass, ScopedLocation,
                                           SubDataclassT, TrialDataclass,
                                           TrialSplitDataclass,
                                           dataclass_2_id_attr, to_json)
from sqlalchemy import (TEXT, Boolean, Column, DateTime, Float, ForeignKey,
                        Integer, MetaData, JSON, String, Table, and_,
                        create_engine)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import (backref, declarative_mixin, relationship,
                            sessionmaker)
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

    def to_dataclass(self, deep=False):
        dc: SubDataclassT = self.dataclass_node.to_empty_dataclass()

        # TODO: int and str keys review.
        keys = self.subdataclasses.keys()
        dc.set_sublocation_keys(keys)
        if deep:
            sub_dcs = self.subdataclasses.values()
            dc.set_sublocation_values(sub_dcs)
        return dc

    def update_dataclass(self, _dataclass: SubDataclassT, deep: bool):
        self.dataclass_node.update_dataclass(_dataclass)
        if deep:
            raise NotImplementedError("TODO: figure out how to deep update nodes.")

    def add_dataclass(self, _dataclass: SubDataclassT, deep: bool):
        ScopedLocationTreeNode(
            dataclass_node=DataClassNode.from_dataclass(_dataclass),
            parent=self  # Linked parent makes the new node added to the session automatically.
        )
        if deep is True:
            raise NotImplementedError("TODO: figure out how to deep update nodes.")


class DataClassNode(Base):
    __tablename__ = 'dataclassnode'
    # id = Column(String, primary_key=True, nullable=True)
    # id = Column(String, ForeignKey('scopedlocationtreenode.id'), primary_key=True)

    type = Column(String(50))

    # tree_id = Column(ForeignKey('scopedlocationtreenode.id'))
    tree_id = Column(String, ForeignKey('scopedlocationtreenode.id'), primary_key=True)
    tree = relationship("ScopedLocationTreeNode", back_populates="dataclass_node")

    logs = relationship("AutoMLLog", back_populates="dataclass_node")

    __mapper_args__ = {
        'polymorphic_identity': 'rootdataclass',
        'polymorphic_on': type
    }

    @staticmethod
    def from_dataclass(_dataclass: BaseDataclass):
        _dc_klass: type = dataclass_2_sqlalchemy_node[_dataclass.__class__]
        dc_node = _dc_klass(_dataclass=_dataclass)
        return dc_node

    def __init__(self, _dataclass: BaseDataclass = None):
        pass

    def update_dataclass(self, _dataclass: RootDataclass):
        raise NotImplementedError("TODO: implement.")

    def to_empty_dataclass(self) -> RootDataclass:
        return RootDataclass()


class ProjectNode(DataClassNode):
    __tablename__ = 'project'
    id = Column(String, ForeignKey('dataclassnode.tree_id'), primary_key=True)
    project_name = Column(String, nullable=False, default=DEFAULT_PROJECT)

    __mapper_args__ = {
        'polymorphic_identity': 'projectdataclass',
    }

    def __init__(self, _dataclass: ProjectDataclass = None):
        if _dataclass is None:
            _dataclass = ProjectDataclass()
        DataClassNode.__init__(self, _dataclass=_dataclass)
        self.project_name = _dataclass.project_name

    def update_dataclass(self, _dataclass: ProjectDataclass):
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

    def __init__(self, _dataclass: ClientDataclass = None):
        if _dataclass is None:
            _dataclass = ClientDataclass()
        DataClassNode.__init__(self, _dataclass=_dataclass)
        self.client_name = _dataclass.client_name

    def update_dataclass(self, _dataclass: ClientDataclass):
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

    def __init__(self, _dataclass: RoundDataclass = None):
        if _dataclass is None:
            _dataclass = RoundDataclass()
        DataClassNode.__init__(self, _dataclass=_dataclass)
        self.round_number = _dataclass.round_number
        self.main_metric_name = _dataclass.main_metric_name

    def update_dataclass(self, _dataclass: RoundDataclass):
        self.round_number = _dataclass.round_number
        self.main_metric_name = _dataclass.main_metric_name

    def to_empty_dataclass(self) -> RoundDataclass:
        return RoundDataclass(round_number=self.round_number, main_metric_name=self.main_metric_name)


@declarative_mixin
class BaseTrialNodeMixin:
    hyperparams = Column(JSON, nullable=False, default=dict)
    status = Column(String, nullable=False, default=TrialStatus.PLANNED)
    created_time = Column(DateTime, nullable=False, default=func.now())
    start_time = Column(DateTime, nullable=True, default=func.now())
    end_time = Column(DateTime, nullable=True)


class TrialNode(BaseTrialNodeMixin, DataClassNode):
    __tablename__ = 'trial'
    id = Column(String, ForeignKey('dataclassnode.tree_id'), primary_key=True)
    trial_number = Column(Integer, nullable=False, default=0)

    __mapper_args__ = {
        'polymorphic_identity': 'trialdataclass',
    }

    def __init__(self, _dataclass: TrialDataclass = None):
        if _dataclass is None:
            _dataclass = TrialDataclass()
        DataClassNode.__init__(self, _dataclass=_dataclass)
        self.trial_number = _dataclass.trial_number

    def update_dataclass(self, _dataclass: TrialDataclass):
        self.trial_number = _dataclass.trial_number

    def to_empty_dataclass(self) -> TrialDataclass:
        return TrialDataclass(trial_number=self.trial_number, retrained_split=None)


class TrialSplitNode(BaseTrialNodeMixin, DataClassNode):
    __tablename__ = 'trialsplit'
    id = Column(String, ForeignKey('dataclassnode.tree_id'), primary_key=True)
    split_number = Column(Integer, nullable=False, default=0)
    # TODO: make split number unique per parent level class.

    __mapper_args__ = {
        'polymorphic_identity': 'trialsplitdataclass',
    }

    def __init__(self, _dataclass: TrialSplitDataclass = None):
        if _dataclass is None:
            _dataclass = TrialSplitDataclass()
        DataClassNode.__init__(self, _dataclass=_dataclass)
        self.split_number = _dataclass.split_number

    def update_dataclass(self, _dataclass: TrialSplitDataclass):
        self.split_number = _dataclass.split_number

    def to_empty_dataclass(self) -> TrialSplitDataclass:
        return TrialSplitDataclass(split_number=self.split_number)


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

    def __init__(self, _dataclass: MetricResultsDataclass = None):
        if _dataclass is None:
            _dataclass = MetricResultsDataclass()
        DataClassNode.__init__(self, _dataclass=_dataclass)
        self.metric_name = _dataclass.metric_name
        self.valid_values = [ValidMetricResultsValues(value=v) for v in _dataclass.validation_values]
        self.train_values = [TrainMetricResultsValues(value=v) for v in _dataclass.train_values]
        self.higher_score_is_better = _dataclass.higher_score_is_better

    def update_dataclass(self, _dataclass: MetricResultsDataclass):
        self.metric_name = _dataclass.metric_name

        # Extending the lists with new values from metric results value tables.
        valid_missing_count: int = len(self.valid_values) - len(_dataclass.validation_values)
        self.valid_values.extend([ValidMetricResultsValues(value=v)
                                 for v in _dataclass.validation_values[-valid_missing_count:]])
        train_missing_count: int = len(self.train_values) - len(_dataclass.train_values)
        self.train_values.extend([TrainMetricResultsValues(value=v)
                                 for v in _dataclass.train_values[-train_missing_count:]])

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

    def __init__(self, value: float, datetime: datetime = None):
        self.value = value
        if datetime is None:
            datetime = datetime.now()
        self.datetime = datetime


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
        self.engine = engine
        self.session = session

    def create_db(self) -> 'DatabaseHyperparamRepository':
        Base.metadata.create_all(self.engine)
        root_dcn = ScopedLocationTreeNode(DataClassNode(RootDataclass()), parent=None)
        self.session.add(root_dcn)
        self.session.commit()
        return self

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
            if len(query.all()) > 0:
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
