"""
Neuraxle's AutoML Scope Manager Classes
====================================
Trial objects used by AutoML algorithm classes.

Classes are splitted like this for the AutoML:
- Projects
- Clients
- Rounds (runs)
- Trials
- TrialSplits
- MetricResults

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


import datetime
import hashlib
import json
import logging
import os
import traceback
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from json.encoder import JSONEncoder
from logging import FileHandler, Logger
from typing import Any, Callable, Dict, Iterable, List, Tuple, Type, Generic, TypeVar, Union, Optional

import numpy as np
from neuraxle.base import BaseStep, ExecutionContext, Flow, TrialStatus
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace, RecursiveDict)
from neuraxle.metaopt.data.vanilla import (AutoMLContext, AutoMLFlow,
                                           BaseDataclass,
                                           BaseHyperparameterOptimizer,
                                           ClientDataclass,
                                           HyperparamsRepository,
                                           InMemoryHyperparamsRepository,
                                           MetricResultsDataclass,
                                           ProjectDataclass, RecursiveDict,
                                           RootMetadata, RoundDataclass,
                                           ScopedLocation, ScopedLocationAttr,
                                           SubDataclassT, TrialDataclass,
                                           TrialSplitDataclass)


class BaseScope(ABC):
    def __init__(self, context: AutoMLContext):
        self.context: AutoMLContext = context

    @property
    def flow(self) -> AutoMLFlow:
        return self.context.flow

    @property
    def loc(self) -> ScopedLocation:
        return self.context.loc

    @property
    def repo(self) -> HyperparamsRepository:
        return self.context.repo

    def log(self, message: str, level: int = logging.INFO):
        return self.flow.log(message, level)


class ProjectScope(BaseScope):
    def __init__(self, context: AutoMLContext, project_name: ScopedLocationAttr):
        super().__init__(context.push_attr(ProjectDataclass, project_name))

    def __enter__(self) -> 'ProjectScope':
        return self

    def new_client(self, client_name: ScopedLocationAttr) -> 'ClientScope':
        return ClientScope(self.context, client_name)

    def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: traceback):
        self.flow.log_error(exc_val)


class ClientScope(BaseScope):

    def __init__(self, context: AutoMLContext, client_name: ScopedLocationAttr):
        super().__init__(context.push_attr(ClientDataclass, client_name))

    def __enter__(self) -> 'ClientScope':
        return self

    def optim_round(self, hp_space: HyperparameterSamples, start_new_run: bool = True) -> 'RoundScope':
        with self.context.lock:
            if start_new_run:
                round_dataclass: RoundDataclass = self.repo.new_round(self.loc)
            else:
                round_dataclass: RoundDataclass = self.repo.get_last_round(self.loc)
        return RoundScope(self, round_dataclass, hp_space)

    def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: traceback):
        self.flow.log_error(exc_val)


class RoundScope(BaseScope):
    def __init__(self, context: AutoMLContext, hp_space: HyperparameterSpace):
        super().__init__(context)
        self.hp_space: HyperparameterSpace = hp_space

    def __enter__(self) -> 'RoundScope':
        # self.flow.log_
        self.flow.add_file_to_logger(self.repo.get_logger_path(self.loc))
        return self

    def new_hyperparametrized_trial(
        self, hp_optimizer: BaseHyperparameterOptimizer, continue_loop_on_error: bool
    ) -> 'TrialScope':
        with self.context.lock:
            new_hps: HyperparameterSamples = hp_optimizer.find_next_best_hyperparams(self)
            ts = TrialScope(self.context, new_hps, continue_loop_on_error)
        return ts

        # TODO: def retrain_best_model?

    def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: traceback):
        self.flow.log_error(exc_val)
        flow: AutoMLFlow = self.context.flow
        self.flow.log_best_hps(flow.repo.get_best_hyperparams(loc=self.loc))
        self.flow.log('Finished round hp search!')
        self.flow._free_logger_files(self.repo.log_path(self.loc))

    def _free_logger_file(self):
        """Remove file handlers from logger to free file lock on Windows."""
        for h in self.logger.handlers:
            if isinstance(h, FileHandler):
                self.logger.removeHandler(h)


class TrialScope(BaseScope):

    def __init__(self, context: AutoMLContext, trial: 'TrialManager', continue_loop_on_error: bool):
        super().__init__(context)

        self.trial: TrialManager = trial
        self.error_types_to_raise = (
            SystemError, SystemExit, EOFError, KeyboardInterrupt
        ) if continue_loop_on_error else (Exception,)

        self.flow.log_planned(trial.get_hyperparams())

    def __enter__(self) -> 'TrialScope':
        self.flow.log_start()
        raise NotImplementedError("TODO: ???")

        self.trial.status = TrialStatus.RUNNING
        self.logger.info(
            '\nnew trial: {}'.format(
                json.dumps(self.hyperparams.to_nested_dict(), sort_keys=True, indent=4)))
        self.save_trial()
        return self

    def new_trial_split(self) -> 'TrialSplitScope':
        trial_split: TrialSplitManager = self.trial.new_validation_split()
        self.repo.set(self.loc, trial_split, False)
        self.repo.set(self.loc.with_dc(trial_split), trial_split, True)
        return TrialSplitScope(self.context, self.repo, trial_split)

    def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: traceback):
        gc.collect()
        raise NotImplementedError("TODO: log split result with MetricResultMetadata?")

        try:
            if exc_type is None:
                self.trial.end(status=TrialStatus.SUCCESS)
            else:
                # TODO: if ctrl+c, raise KeyboardInterrupt? log aborted or log failed?
                self.flow.log_error(exc_val)
                self.trial.end(status=TrialStatus.FAILED)
                raise exc_val
        finally:
            self.save_trial()
            self._free_logger_file()


class TrialSplitScope(BaseScope):
    def __init__(self, context: AutoMLContext, repo: HyperparamsRepository, n_epochs: int):
        super().__init__(context, repo)
        self.n_epochs = n_epochs

    def __enter__(self):
        """
        Start trial, and set the trial status to PLANNED.
        """
        # TODO: ??
        self.trial_split.status = TrialStatus.RUNNING
        self.save_parent_trial()
        return self

    def new_epoch(self, epoch: int) -> 'EpochScope':
        return EpochScope(self.context, self.repo, epoch, self.n_epochs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop trial, and save end time.

        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        self.end_time = datetime.datetime.now()
        if self.delete_pipeline_on_completion:
            del self.pipeline
        if exc_type is not None:
            self.set_failed(exc_val)
            self.trial_split.end(TrialStatus.FAILED)
            self.save_parent_trial()
            return False

        self.trial_split.end(TrialStatus.SUCCESS)
        self.save_parent_trial()
        return True

    def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: traceback):
        raise NotImplementedError("TODO: log split result with MetricResultMetadata?")

        if exc_val is None:
            self.flow.log_success()
        elif exc_type in self.error_types_to_raise:
            self.flow.log_failure(exc_val)
            return False  # re-raise.
        else:
            self.flow.log_error(exc_val)
            self.flow.log_aborted()
            return True  # don't re-raise.


class EpochScope(BaseScope):
    def __init__(self, context: AutoMLContext, repo: HyperparamsRepository, epoch: int, n_epochs: int):
        super().__init__(context, repo)
        self.epoch: int = epoch
        self.n_epochs: int = n_epochs

    def __enter__(self) -> 'EpochScope':
        self.flow.log_epoch(self.epoch, self.n_epochs)
        return self

    def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: traceback):
        self.flow.log_error(exc_val)
        raise NotImplementedError("TODO: log MetricResultMetadata?")
