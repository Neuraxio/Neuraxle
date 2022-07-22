"""
Neuraxle's Hyperparameter Repository Base Classes
====================================================

Not all repositories can be found here, but the base can be found here.


.. seealso::
    :class:`~neuraxle.metaopt.hyperopt.repositories.json.HyperparamsOnDiskRepository`,
    :class:`~neuraxle.metaopt.hyperopt.repositories.db.DatabaseHyperparamRepository`,
    :class:`~neuraxle.metaopt.hyperopt.repositories.db.SQLLiteHyperparamsRepository`,


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

import copy
import multiprocessing
import os
from abc import abstractmethod

from neuraxle.base import BaseService, MetaServiceMixin
from neuraxle.logging.logging import NeuraxleLogger
from neuraxle.logging.warnings import RaiseDeprecatedClass
from neuraxle.metaopt.data.vanilla import (BaseDataclass, RootDataclass,
                                           ScopedLocation, ScopedLocationAttr,
                                           SubDataclassT,
                                           dataclass_2_subloc_attr)


class HyperparamsRepository(BaseService):
    """
    Hyperparams repository that saves hyperparams, and scores for every AutoML trial.
    Cache folder can be changed to do different round numbers.

    For more information, read this `article by Martin Fowler on DDD Aggregates <https://martinfowler.com/bliki/DDD_Aggregate.html>`_.

    .. seealso::
        :class:`AutoML`,
        :class:`Trainer`,
    """

    def __init__(self):
        BaseService.__init__(self)

    @abstractmethod
    def load(self, scope: ScopedLocation, deep=False) -> SubDataclassT:
        """
        Get metadata from scope.

        The fetched metadata will be the one that is the last item
        that is not a None in the provided scope.

        :param scope: scope to get metadata from.
        :return: metadata from scope.
        """
        raise NotImplementedError("Use a concrete class. This is an abstract class.")

    @abstractmethod
    def save(self, _dataclass: SubDataclassT, scope: ScopedLocation, deep=False) -> 'HyperparamsRepository':
        """
        Save metadata to scope.

        :param metadata: metadata to save.
        :param scope: scope to save metadata to.
        :param deep: if True, save metadata's sublocations recursively.
        """
        raise NotImplementedError("Use a concrete class. This is an abstract class.")

    @abstractmethod
    def add_logging_handler(self, logger: NeuraxleLogger, scope: ScopedLocation) -> 'HyperparamsRepository':
        """
        Add logging handler to repository's provided scope.
        Handler must be set manually for each scope below this scope.

        :param logger: logger to add handler to.
        :param scope: scope to add handler to.
        :return: self.
        """
        raise NotImplementedError("Use a concrete class. This is an abstract class.")

    @abstractmethod
    def get_log_from_logging_handler(self, logger: NeuraxleLogger, scope: ScopedLocation) -> str:
        """
        Get log from repository's provided scope and handler that was set with :func:`add_logging_handler`.

        :param scope: scope to get log from.
        :return: log from scope.
        """
        raise NotImplementedError("Use a concrete class. This is an abstract class.")

    def with_lock(self) -> 'SynchronizedHyperparamsRepositoryWrapper':
        return SynchronizedHyperparamsRepositoryWrapper(self)


def func_with_rlock():
    def decorator(func):
        def _LOCKED_REPO(self, *args, **kwargs):
            with self.lock:
                return func(self, *args, **kwargs)
        return _LOCKED_REPO
    return decorator


class SynchronizedHyperparamsRepositoryWrapper(MetaServiceMixin, HyperparamsRepository):
    """
    A wrapper that makes any HyperparamsRepository thread-safe using locking.
    """

    def __init__(self, wrapped: HyperparamsRepository):
        MetaServiceMixin.__init__(self, wrapped)
        HyperparamsRepository.__init__(self)
        self.lock = multiprocessing.RLock()

    @func_with_rlock()
    def load(self, scope: ScopedLocation, deep=False) -> SubDataclassT:
        repo: HyperparamsRepository = self.get_step()
        return repo.load(scope, deep)

    @func_with_rlock()
    def save(self, _dataclass: SubDataclassT, scope: ScopedLocation, deep=False) -> 'HyperparamsRepository':
        repo: HyperparamsRepository = self.get_step()
        return repo.save(_dataclass, scope, deep)

    @func_with_rlock()
    def add_logging_handler(self, logger: NeuraxleLogger, scope: ScopedLocation) -> 'HyperparamsRepository':
        repo: HyperparamsRepository = self.get_step()
        return repo.add_logging_handler(logger, scope)

    @func_with_rlock()
    def get_log_from_logging_handler(self, logger: NeuraxleLogger, scope: ScopedLocation) -> str:
        repo: HyperparamsRepository = self.get_step()
        return repo.get_log_from_logging_handler(logger, scope)

#     def _get_thread_safe(self):
#         pass
#
#     def _get_pickle_safe(self):
#         pass
#
#     def set_thread_safe():
#         pass

    def with_lock(self) -> 'SynchronizedHyperparamsRepositoryWrapper':
        return self


class _InMemoryRepositoryLoggerHandlerMixin:
    """
    Mixin to add a in-memory logging handler to a repository.
    """

    def add_logging_handler(self, logger: NeuraxleLogger, scope: ScopedLocation) -> 'HyperparamsRepository':
        return self

    def get_log_from_logging_handler(self, logger: NeuraxleLogger, scope: ScopedLocation) -> str:
        return logger.get_scoped_string_history()


class VanillaHyperparamsRepository(_InMemoryRepositoryLoggerHandlerMixin, HyperparamsRepository):
    """
    Hyperparams repository that saves data AutoML-related info.
    """

    def __init__(
        self,
        cache_folder: str
    ):
        """
        :param cache_folder: folder to store trials.
        :param hyperparams_repo_class: class to use to save hyperparams.
        :param hyperparams_repo_kwargs: kwargs to pass to hyperparams_repo_class.
        """
        HyperparamsRepository.__init__(self)
        _InMemoryRepositoryLoggerHandlerMixin.__init__(self)
        self.cache_folder = os.path.join(cache_folder, self.__class__.__name__)
        self.root: RootDataclass = RootDataclass()

    @staticmethod
    def from_root(root: RootDataclass, cache_folder: str) -> 'VanillaHyperparamsRepository':

        return VanillaHyperparamsRepository(
            cache_folder=cache_folder,
        ).save(root, ScopedLocation(), deep=True)

    def load(self, scope: ScopedLocation, deep=False) -> SubDataclassT:
        try:
            ret: BaseDataclass = self.root[scope]
            if not deep:
                ret = ret.shallow()
        except KeyError:
            ret: BaseDataclass = scope.new_dataclass_from_id()

        return copy.deepcopy(ret)

    def save(self, _dataclass: SubDataclassT, scope: ScopedLocation, deep=False) -> 'VanillaHyperparamsRepository':
        # Sanitizing
        _dataclass: SubDataclassT = copy.deepcopy(_dataclass)
        scope = scope.at_dc(_dataclass)

        # Sanity checks: good type
        if not isinstance(_dataclass, BaseDataclass):
            raise ValueError(f"Was expecting a dataclass. Got `{_dataclass.__class__.__name__}`.")
        # Sanity checks: sufficient scope depth
        scope = scope[:_dataclass.__class__]  # Sanitizing scope to dtype loc.
        _id_from_scope: ScopedLocationAttr = scope[_dataclass.__class__]
        if _id_from_scope is not None:
            if _id_from_scope != _dataclass.get_id():
                raise ValueError(
                    f"The scope `{scope}` with {_dataclass.__class__.__name__} id `{_id_from_scope}` does not match the provided dataclass id `{_dataclass.get_id()}` for `{_dataclass}`."
                )
            scope.pop()
            # else check if the scope is at least of the good class length:
        elif len(scope) != list(dataclass_2_subloc_attr.keys()).index(_dataclass.__class__):
            raise ValueError(
                f"The scope `{scope}` is not of the good length for dataclass type `{_dataclass.__class__.__name__}`."
            )

        # Passthrough dc's sublocation when saving shallow:
        if not deep:
            if isinstance(_dataclass, RootDataclass):
                _dataclass.set_sublocation(self.root.get_sublocation())
            elif scope.with_dc(_dataclass) in self.root:
                prev_dc: SubDataclassT = self.root[scope.with_dc(_dataclass)]
                _dataclass.set_sublocation(prev_dc.get_sublocation())
            else:
                _dataclass = _dataclass.empty()

        # Finally storing.
        if isinstance(_dataclass, RootDataclass):
            self.root = _dataclass
        else:
            self.root[scope].store(_dataclass)
        return self


class InMemoryHyperparamsRepository(RaiseDeprecatedClass, VanillaHyperparamsRepository):
    """
    In memory hyperparams repository that can print information about trials.
    Useful for debugging.
    """

    def __init__(self, *kargs, **kwargs):
        RaiseDeprecatedClass.__init__(
            self,
            replacement_class=VanillaHyperparamsRepository,
            since_version="0.7.0",
        )
        VanillaHyperparamsRepository.__init__(self, *kargs, **kwargs)
