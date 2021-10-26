import json
import os

from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.metaopt.auto_ml import HyperparamsJSONRepository

HYPERPARAMS = {'learning_rate': 0.01}
FIRST_SCORE_FOR_TRIAL = 1


def test_hyperparams_repository_should_create_new_trial(tmpdir):
    hyperparams_json_repository = HyperparamsJSONRepository(tmpdir)
    hyperparams = HyperparameterSamples(HYPERPARAMS)

    hyperparams_json_repository.create_new_trial(hyperparams)

    trial_hash = hyperparams_json_repository._get_trial_hash(hyperparams.to_flat_dict())
    file_name = 'NEW_' + trial_hash + '.json'
    path = os.path.join(tmpdir, file_name)
    with open(path) as f:
        trial_json = json.load(f)
    assert trial_json['hyperparams'] == hyperparams.to_flat_dict()
    assert trial_json['score'] is None


def test_hyperparams_repository_should_load_all_trials(tmpdir):
    tmpdir = os.path.join(tmpdir, "__json__")
    os.mkdir(tmpdir)
    hyperparams_json_repository = HyperparamsJSONRepository(tmpdir)
    n_trials = 3
    for i in range(n_trials):
        hyperparams = HyperparameterSamples({'learning_rate': 0.01 + i * 0.01})
        hyperparams_json_repository.save_score_for_success_trial(hyperparams, i)

    trials = hyperparams_json_repository.load_all_trials()

    assert len(trials) == n_trials
    for i in range(n_trials):
        assert trials[i].hyperparams == HyperparameterSamples(
            {'learning_rate': 0.01 + i * 0.01}).to_flat_dict(), (i, str(trials))


def test_hyperparams_repository_should_save_failed_trial(tmpdir):
    hyperparams_json_repository = HyperparamsJSONRepository(tmpdir)
    hyperparams = HyperparameterSamples(HYPERPARAMS)

    hyperparams_json_repository._save_failed_trial_json(hyperparams, Exception('trial failed'))

    trial_hash = hyperparams_json_repository._get_trial_hash(hyperparams.to_flat_dict())
    file_name = 'FAILED_' + trial_hash + '.json'
    path = os.path.join(tmpdir, file_name)
    with open(path) as f:
        trial_json = json.load(f)
    assert trial_json['hyperparams'] == hyperparams.to_flat_dict()
    assert 'exception' in trial_json.keys()
    assert trial_json['score'] is None


def test_hyperparams_repository_should_save_success_trial(tmpdir):
    hyperparams_json_repository = HyperparamsJSONRepository(tmpdir)
    hyperparams = HyperparameterSamples(HYPERPARAMS)

    hyperparams_json_repository._save_successful_trial_json(hyperparams, FIRST_SCORE_FOR_TRIAL)

    trial_hash = hyperparams_json_repository._get_trial_hash(hyperparams.to_flat_dict())
    file_name = str(float(FIRST_SCORE_FOR_TRIAL)).replace('.', ',') + '_' + trial_hash + '.json'
    path = os.path.join(tmpdir, file_name)
    with open(path) as f:
        trial_json = json.load(f)
    assert trial_json['hyperparams'] == hyperparams.to_flat_dict()
    assert trial_json['score'] == FIRST_SCORE_FOR_TRIAL
