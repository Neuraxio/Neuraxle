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
import glob
import json
import logging
import os
import time
import traceback
import warnings
from typing import List, Tuple

from neuraxle.base import TrialStatus
from neuraxle.data_container import DataContainer as DACT
from neuraxle.metaopt.data.trial import RoundManager, TrialManager
from neuraxle.metaopt.data.vanilla import HyperparamsRepository
from neuraxle.metaopt.observable import _ObservableRepo


class HyperparamsJSONRepository(HyperparamsRepository):
    """
    Hyperparams repository that saves json files for every AutoML trial.

    Example usage :

    .. code-block:: python

        HyperparamsJSONRepository(
            hyperparameter_selection_strategy=RandomSearchHyperparameterSelectionStrategy(),
            cache_folder='cache',
            best_retrained_model_folder='best'
        )


    .. seealso::
        :class:`AutoML`,
        :class:`Trainer`,
        :class:`~neuraxle.metaopt.data.trial.Trial`,
        :class:`InMemoryHyperparamsRepository`,
        :class:`HyperparamsJSONRepository`,
        :class:`BaseHyperparameterSelectionStrategy`,
        :class:`RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.hyperparams.trial.HyperparameterSamples`
    """

    def __init__(
            self,
            cache_folder: str = None,
            best_retrained_model_folder: str = None
    ):
        HyperparamsRepository.__init__(self)
        cache_folder: str = cache_folder if cache_folder is not None else 'json_repo_cache'
        best_retrained_model_folder: str = (
            best_retrained_model_folder if best_retrained_model_folder is not None else 'json_repo_best_model')
        self.json_path_remove_on_update = None

    def _save_trial(self, trial: 'TrialManager'):
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

    def new_trial(self, auto_ml_container) -> TrialManager:
        """
        Create new hyperperams trial json file.

        :param auto_ml_container: auto ml container
        :return:
        """
        trial: TrialManager = HyperparamsRepository.new_trial(self, auto_ml_container)
        self._save_trial(trial)

        return trial

    def load_trials(self, status: 'TrialStatus' = None) -> 'RoundManager':
        """
        Load all hyperparameter trials with their corresponding score.
        Reads all the saved trial json files, sorted by creation date.

        :param status: (optional) filter to select only trials with this status.
        :return: (hyperparams, scores)
        """
        trials = RoundManager()

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
                trials.append(TrialManager.from_json(
                    update_trial_function=self.save_trial,
                    trial_json=trial_json,
                    cache_folder=self.cache_folder
                ))

        return trials

    def _get_successful_trial_json_file_path(self, trial: 'TrialManager') -> str:
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

    def _get_failed_trial_json_file_path(self, trial: 'TrialManager'):
        """
        Get the json path for the given failed trial.

        :param trial: trial
        :return: str
        """
        trial_hash = self.get_trial_id(trial.hyperparams.to_flat_dict())
        return os.path.join(self.cache_folder, 'FAILED_' + trial_hash) + '.json'

    def _get_ongoing_trial_json_file_path(self, trial: 'TrialManager'):
        """
        Get ongoing trial json path.
        """
        hp_dict = trial.hyperparams.to_flat_dict()
        current_hyperparameters_hash = self.get_trial_id(hp_dict)
        return os.path.join(self.cache_folder, "ONGOING_" + current_hyperparameters_hash) + '.json'

    def _get_new_trial_json_file_path(self, trial: 'TrialManager'):
        """
        Get new trial json path.
        """
        hp_dict = trial.hyperparams.to_flat_dict()
        current_hyperparameters_hash = self.get_trial_id(hp_dict)
        return os.path.join(self.cache_folder, "NEW_" + current_hyperparameters_hash) + '.json'

    def _remove_previous_trial_state_json(self):
        if self.json_path_remove_on_update and os.path.exists(self.json_path_remove_on_update):
            os.remove(self.json_path_remove_on_update)

    def subscribe_to_cache_folder_changes(self, refresh_interval_in_seconds: int,
                                          observer: _ObservableRepo[Tuple[HyperparamsRepository, TrialManager]]):
        """
        Every refresh_interval_in_seconds

        :param refresh_interval_in_seconds: number of seconds to wait before sending updates to the observers
        :param observer:
        :return:
        """
        self._observers.add(observer)
        # TODO: start a process that notifies observers anytime a the file of a trial changes
        # possibly use this ? https://github.com/samuelcolvin/watchgod
        # note: this is how you notify observers self.on_next((self, updated_trial))
