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
from copy import copy
import glob
import json
import os
from this import d
import time
import traceback
from typing import List, Tuple

from neuraxle.base import TrialStatus
from neuraxle.metaopt.data.aggregates import Round, Trial
from neuraxle.metaopt.data.vanilla import AutoMLContext, HyperparamsRepository, VanillaHyperparamsRepository, BaseDataclass, ScopedLocation, SubDataclassT, dataclass_2_id_attr, to_json, from_json
from neuraxle.metaopt.observable import _ObservableRepo


class HyperparamsOnDiskRepository(HyperparamsRepository):
    """
    Hyperparams repository that saves json files for every AutoML trial.

    .. seealso::
        :class:`AutoML`,
        :class:`Trainer`,
    """

    def __init__(
            self,
            cache_folder: str = None,
    ):
        HyperparamsRepository.__init__(self)
        self._vanilla = VanillaHyperparamsRepository(cache_folder=cache_folder)
        # self.cache_folder = cache_folder

    def load(self, scope: ScopedLocation, deep=False) -> SubDataclassT:
        """
        Get metadata from scope.

        The fetched metadata will be the one that is the last item
        that is not a None in the provided scope.

        :param scope: scope to get metadata from.
        :return: metadata from scope.
        """
        self._load_dc(scope=scope, deep=deep)
        loaded = self._vanilla.load(scope=scope, deep=deep)
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
        _, _, load_file = self._filenameof(None, scope)

        if not os.path.exists(load_file):
            raise FileNotFoundError(f"{load_file} not found.")

        with open(load_file, 'r') as f:
            _dataclass = from_json(json.load(f))
            if deep:
                raise NotImplemented("TODO")
            return _dataclass

    def _save_dc(self, _dataclass: SubDataclassT, scope: ScopedLocation, deep=False):
        scope, save_folder, save_file = self._filenameof(_dataclass, scope)

        os.makedirs(save_folder, exist_ok=True)
        with open(save_file, 'w') as f:
            json.dump(to_json(_dataclass.shallow()), f, indent=4)

        if deep:
            for sub_dc in _dataclass.get_sublocation_values():
                self._save_dc(sub_dc, scope=scope, deep=deep)

    def _filenameof(self, _dataclass, scope):
        scope = copy.deepcopy(scope)
        _dc_id = _dataclass.get_id()
        if _dc_id is not None:
            setattr(scope, dataclass_2_id_attr[_dc_id.__class__], _dc_id)
        suffixes = scope.as_list()
        prefix = self._vanilla.cache_folder

        save_folder = os.path.join(prefix, *suffixes)
        save_file = os.path.join(save_folder, 'metadata.json')

        return scope, save_folder, save_file

    def get_scoped_logger_path(self, scope: ScopedLocation) -> str:
        raise NotImplemented("TODO")


class _HyperparamsOnDiskRepository(HyperparamsRepository):
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
