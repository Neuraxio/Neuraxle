"""
Neuraxle's AutoML Context Management
====================================================

The :class:`~neuraxle.base.ExecutionContext`, is a special object
that needs to be adapted in AutoML loops.


.. seealso::
    :class:`~neuraxle.metaopt.hyperopt.tpe.TreeParzenEstimator`,


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

import logging

from neuraxle.base import ExecutionContext as CX, ExecutionPhase
from neuraxle.logging.logging import NeuraxleLogger
from neuraxle.metaopt.data.vanilla import (BaseDataclass, ScopedLocation,
                                           SubDataclassT)
from neuraxle.metaopt.repositories.repo import (HyperparamsRepository, SynchronizedHyperparamsRepositoryWrapper,
                                                VanillaHyperparamsRepository)


class AutoMLContext(CX):

    @property
    def logger(self) -> NeuraxleLogger:
        # self.add_scoped_logger_file_handler()  # TODO: this is perhaps why logs are duplicated.
        return CX.logger.fget(self)

    @property
    def logger_at_scoped_loc(self) -> NeuraxleLogger:
        return logging.getLogger(self.get_identifier(include_step_names=False))

    def add_scoped_logger_file_handler(self) -> 'AutoMLContext':
        """
        Add a file handler to the logger at the current scoped location to capture logs
        at this scope and below this scope.
        """
        self.repo.add_logging_handler(self.logger_at_scoped_loc, self.loc)
        return self

    def free_scoped_logger_file_handler(self) -> 'AutoMLContext':
        """
        Remove file handlers from logger to free file lock (especially on Windows).
        """
        self.logger_at_scoped_loc.without_file_handler()
        return self

    def read_scoped_log(self) -> str:
        """
        Read the scoped logger file.
        """
        # TODO: with self.lock:
        return self.repo.get_log_from_logging_handler(self.logger, self.loc)

    def _copy(self, copy_func: str = '_copy'):
        copy_kwargs = self._get_copy_kwargs(copy_func)
        return AutoMLContext(**copy_kwargs)

    def _get_copy_kwargs(self, copy_func: str):
        return CX._get_copy_kwargs(self, copy_func)

    def new_trial(self) -> 'CX':
        """
        Set the context's execution phase to train.
        """
        new_self = self._copy(copy_func='_copy_trial')
        new_self.set_execution_phase(ExecutionPhase.PRETRAIN)
        return new_self

    def new_trial_split(self) -> 'CX':
        """
        Set the context's execution phase to train.
        """
        new_self = self._copy(copy_func='_copy_trial_split')
        new_self.set_execution_phase(ExecutionPhase.PRETRAIN)
        return new_self

    def train(self) -> 'CX':
        """
        Set the context's execution phase to train.
        """
        new_self = self._copy(copy_func='_copy_train')
        new_self.set_execution_phase(ExecutionPhase.TRAIN)
        return new_self

    def validation(self) -> 'CX':
        """
        Set the context's execution phase to validation.
        """
        new_self = self._copy(copy_func='_copy_validation')
        new_self.set_execution_phase(ExecutionPhase.VALIDATION)
        return new_self

    @staticmethod
    def from_context(
        context: CX = None,
        repo: HyperparamsRepository = None,
        loc: ScopedLocation = None,
        disable_repo_lock: bool = False,
    ) -> 'AutoMLContext':
        """
        Create a new AutoMLContext from an ExecutionContext.

        :param context: ExecutionContext
        """
        new_context: AutoMLContext = AutoMLContext._copy(
            context if context is not None else AutoMLContext()
        )
        if not new_context.has_service(HyperparamsRepository):
            new_context.register_service(
                HyperparamsRepository,
                (repo or VanillaHyperparamsRepository(new_context.get_path())).with_lock()
            )
        if not new_context.has_service(ScopedLocation):
            new_context.register_service(
                ScopedLocation,
                loc or ScopedLocation()
            )
        if not disable_repo_lock:
            new_context = new_context.synchroneous()
        return new_context

    @property
    def loc(self) -> ScopedLocation:
        return self.get_service(ScopedLocation)

    @property
    def repo(self) -> SynchronizedHyperparamsRepositoryWrapper:
        repo: HyperparamsRepository = self.get_service(HyperparamsRepository)
        return repo.with_lock()

    def push_attr(self, subdataclass: SubDataclassT) -> 'AutoMLContext':
        """
        Push a new attribute into the ScopedLocation.

        :param name: attribute name
        :param value: attribute value
        :return: an AutoMLContext copy with the new loc attribute.
        """
        if not isinstance(subdataclass, BaseDataclass) and isinstance(subdataclass, (str, int)):
            return self.with_loc(self.loc.with_id(subdataclass))  # ID instead.
        return self.with_loc(self.loc.with_dc(subdataclass))

    def pop_attr(self) -> 'AutoMLContext':
        """
        Pop an attribute from the ScopedLocation.

        :return: an AutoMLContext copy with the new popped loc attribute.
        """
        return self.with_loc(self.loc.popped())

    def with_loc(self, loc: ScopedLocation) -> 'AutoMLContext':
        """
        Replace the ScopedLocation by the one provided.

        :param loc: ScopedLocation
        :return: an AutoMLContext copy with the new loc attribute.
        """
        new_self: AutoMLContext = self._copy()
        new_self.register_service(ScopedLocation, loc)
        return new_self

    def load_dc(self, deep=True) -> BaseDataclass:
        """
        Load the current dc from the repo.
        """
        with self.repo.lock:
            return self.repo.load(self.loc, deep)
