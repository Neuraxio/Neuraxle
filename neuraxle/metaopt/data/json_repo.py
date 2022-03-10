"""
Neuraxle's SQLAlchemy Hyperparameter Repository Classes
=================================================
Data objects and related repositories used by AutoML, SQL version.

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
from copy import copy, deepcopy
import glob
import json
import os
import time
import traceback
from typing import List, Tuple

from neuraxle.base import TrialStatus
from neuraxle.logging.logging import NeuraxleLogger
from neuraxle.metaopt.data.aggregates import Round, Trial
from neuraxle.metaopt.data.vanilla import AutoMLContext, HyperparamsRepository, VanillaHyperparamsRepository, BaseDataclass, ScopedLocation, SubDataclassT, dataclass_2_id_attr, dataclass_2_subdataclass, to_json, from_json
from neuraxle.metaopt.observable import _ObservableRepo


class _OnDiskRepositoryLoggerHandlerMixin:
    """
    Mixin to add a disk logging handler to a repository. It has a cache_folder.
    """

    def __init__(self, cache_folder: str):
        self.cache_folder = cache_folder

    def add_logging_handler(self, logger: NeuraxleLogger, scope: ScopedLocation) -> 'HyperparamsRepository':
        """
        Adds an on-disk logging handler to the repository.
        The file at this scope can be retrieved with the method :func:`get_scoped_logger_path`.
        """
        logging_file = self.get_scoped_logger_path(scope)
        os.makedirs(os.path.dirname(logging_file), exist_ok=True)
        logger.with_file_handler(logging_file)
        return self

    def get_log_from_logging_handler(self, logger: NeuraxleLogger, scope: ScopedLocation) -> str:
        return ''.join(logger.read_log_file())
        # TODO: what to do with this code:
        # def read_scoped_logger_file(self) -> str:
        #     with open(self.repo.get_scoped_logger_path(self.loc), "r") as f:
        #         l: str = "".join(f.readlines())
        #     return l

    def get_folder_at_scope(self, scope: ScopedLocation) -> str:
        _scope_attrs = scope.as_list(stringify=True)
        return os.path.join(self.cache_folder, *_scope_attrs)

    def get_scoped_logger_path(self, scope: ScopedLocation) -> str:
        scoped_path: str = self.get_folder_at_scope(scope)
        return os.path.join(scoped_path, 'log.txt')


class HyperparamsOnDiskRepository(_OnDiskRepositoryLoggerHandlerMixin, HyperparamsRepository):
    """
    Hyperparams repository that saves json files for every AutoML trial.

    .. seealso::
        :class:`AutoML`,
        :class:`Trainer`,
    """

    def __init__(
            self,
            cache_folder: str = None,
            for_unit_testing: bool = False,
    ):
        HyperparamsRepository.__init__(self)
        _OnDiskRepositoryLoggerHandlerMixin.__init__(self, cache_folder=cache_folder)
        self._vanilla = VanillaHyperparamsRepository(cache_folder=cache_folder)
        self.for_unit_testing: bool = for_unit_testing

    def load(self, scope: ScopedLocation, deep=False) -> SubDataclassT:
        """
        Get metadata from scope.

        The fetched metadata will be the one that is the last item
        that is not a None in the provided scope.

        :param scope: scope to get metadata from.
        :return: metadata from scope.
        """
        loaded = self._load_dc(scope=scope, deep=deep)
        self._vanilla.save(loaded, scope=scope, deep=deep)
        # inmem_loaded = self._vanilla.load(scope=scope, deep=deep)
        # if self.for_unit_testing is True:
        #     assert(loaded == inmem_loaded), f"disk-loaded and in-memory-loaded are not the same: {loaded} != {inmem_loaded}"
        return loaded

    def save(self, _dataclass: SubDataclassT, scope: ScopedLocation, deep=False) -> 'HyperparamsRepository':
        """
        Save metadata to scope.

        :param metadata: metadata to save.
        :param scope: scope to save metadata to.
        :param deep: if True, save metadata's sublocations recursively so as to update.
        """
        self._vanilla.save(_dataclass=_dataclass, scope=scope, deep=deep)
        self._save_dc(_dataclass=_dataclass, scope=scope, deep=deep)
        return self

    def _load_dc(self, scope: ScopedLocation, deep=False) -> SubDataclassT:
        scope, _, load_file = self._get_dataclass_filename_path(None, scope)

        if not os.path.exists(load_file):
            # raise FileNotFoundError(f"{load_file} not found.")
            return self._vanilla.load(scope=scope, deep=deep)

        with open(load_file, 'r') as f:
            _dataclass: SubDataclassT = from_json(json.load(f)).shallow()
            if deep is True and _dataclass.__class__ in list(dataclass_2_subdataclass.keys())[:-1]:
                for sub_dc_id in _dataclass.get_sublocation_keys():
                    sub_dc = self._load_dc(scope=scope.with_id(sub_dc_id), deep=deep)
                    _dataclass.store(sub_dc)
            return _dataclass

    def _save_dc(self, _dataclass: SubDataclassT, scope: ScopedLocation, deep=False):
        scope, save_folder, save_file = self._get_dataclass_filename_path(_dataclass, scope)

        os.makedirs(save_folder, exist_ok=True)
        with open(save_file, 'w') as f:
            _dc_to_save = _dataclass.shallow() if deep else _dataclass.empty()
            json.dump(to_json(_dc_to_save), f, indent=4)

        if deep is True and _dataclass.__class__ in list(dataclass_2_subdataclass.keys())[:-1]:
            for sub_dc in _dataclass.get_sublocation_values():
                self._save_dc(sub_dc, scope=scope, deep=deep)

    def _get_dataclass_filename_path(self, _dataclass: SubDataclassT, scope: ScopedLocation):
        scope = self._patch_scope_for_dataclass(_dataclass, scope)

        save_folder = self.get_folder_at_scope(scope)
        save_file = os.path.join(save_folder, 'metadata.json')

        return scope, save_folder, save_file

    def _patch_scope_for_dataclass(self, _dataclass: BaseDataclass, scope: ScopedLocation):
        scope = deepcopy(scope)
        if _dataclass is not None and _dataclass.get_id() is not None:
            scope = scope.at_dc(_dataclass)
            setattr(scope, dataclass_2_id_attr[_dataclass.__class__], _dataclass.get_id())
        return scope


class _HyperparamsOnDiskRepositoryDEPRECATED(HyperparamsRepository):
    """
    Hyperparams repository that saves json files for every AutoML trial.

    .. seealso::
        :class:`AutoML`,
        :class:`Trainer`,
    """

    def __init__(
            self,
            cache_folder: str = None,
            best_retrained_model_folder: str = None  # TODO: prepend a keyword and remove this param.
    ):
        HyperparamsRepository.__init__(self)
        cache_folder: str = cache_folder if cache_folder is not None else 'json_repo_cache'
        best_retrained_model_folder: str = (
            best_retrained_model_folder if best_retrained_model_folder is not None else 'json_repo_best_model')
        self.json_path_remove_on_update = None

    def _save_trial(self, trial: 'Trial'):
        """
        Save trial json.

        :param trial: trial to save
        :return:
        """
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        self._remove_previous_trial_state_json()

        trial_path_func = {
            TrialStatus.SUCCESS: self._get_successful_trial_json_file_path,
            TrialStatus.FAILED: self._get_failed_trial_json_file_path,
            TrialStatus.RUNNING: self._get_ongoing_trial_json_file_path,
            TrialStatus.PLANNED: self._get_new_trial_json_file_path
        }
        trial_file_path = trial_path_func[trial.status](trial)

        with open(trial_file_path, 'w+') as outfile:
            json.dump(trial.to_json(), outfile)

        if trial.status in (TrialStatus.SUCCESS, TrialStatus.FAILED):
            self.json_path_remove_on_update = None
        else:
            self.json_path_remove_on_update = trial_file_path

        # Sleeping to have a valid time difference between files when reloading them to sort them by creation time:
        time.sleep(0.1)

    def new_trial(self, auto_ml_container) -> Trial:
        """
        Create new hyperperams trial json file.

        :param auto_ml_container: auto ml container
        :return:
        """
        trial: Trial = HyperparamsRepository.new_trial(self, auto_ml_container)
        self._save_trial(trial)

        return trial

    def load_trials(self, status: 'TrialStatus' = None) -> 'Round':
        """
        Load all hyperparameter trials with their corresponding score.
        Reads all the saved trial json files, sorted by creation date.

        :param status: (optional) filter to select only trials with this status.
        :return: (hyperparams, scores)
        """
        trials = Round()

        files = glob.glob(os.path.join(self.cache_folder, '*.json'))

        # sort by created date:
        def getmtimens(filename):
            return os.stat(filename).st_mtime_ns

        files.sort(key=getmtimens)

        for base_path in files:
            with open(base_path) as f:
                try:
                    trial_json = json.load(f)
                except Exception as err:
                    print('invalid trial json file'.format(base_path))
                    print(traceback.format_exc())
                    continue

            if status is None or trial_json['status'] == status.value:
                trials.append(Trial.from_json(
                    update_trial_function=self.save_trial,
                    trial_json=trial_json,
                    cache_folder=self.cache_folder
                ))

        return trials

    def _get_successful_trial_json_file_path(self, trial: 'Trial') -> str:
        """
        Get the json path for the given successful trial.

        :param trial: trial
        :return: str
        """
        trial_hash = self.get_trial_id(trial.hyperparams.to_flat_dict())
        return os.path.join(
            self.cache_folder,
            str(float(trial.get_avg_validation_score())).replace('.', ',') + "_" + trial_hash
        ) + '.json'

    def _get_failed_trial_json_file_path(self, trial: 'Trial'):
        """
        Get the json path for the given failed trial.

        :param trial: trial
        :return: str
        """
        trial_hash = self.get_trial_id(trial.hyperparams.to_flat_dict())
        return os.path.join(self.cache_folder, 'FAILED_' + trial_hash) + '.json'

    def _get_ongoing_trial_json_file_path(self, trial: 'Trial'):
        """
        Get ongoing trial json path.
        """
        hp_dict = trial.hyperparams.to_flat_dict()
        current_hyperparameters_hash = self.get_trial_id(hp_dict)
        return os.path.join(self.cache_folder, "ONGOING_" + current_hyperparameters_hash) + '.json'

    def _get_new_trial_json_file_path(self, trial: 'Trial'):
        """
        Get new trial json path.
        """
        hp_dict = trial.hyperparams.to_flat_dict()
        current_hyperparameters_hash = self.get_trial_id(hp_dict)
        return os.path.join(self.cache_folder, "NEW_" + current_hyperparameters_hash) + '.json'

    def _remove_previous_trial_state_json(self):
        if self.json_path_remove_on_update and os.path.exists(self.json_path_remove_on_update):
            os.remove(self.json_path_remove_on_update)
