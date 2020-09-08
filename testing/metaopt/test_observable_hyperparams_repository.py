from typing import List, Tuple

from neuraxle.metaopt.auto_ml import InMemoryHyperparamsRepository, HyperparamsJSONRepository, HyperparamsRepository
from neuraxle.metaopt.observable import _Observer
from neuraxle.metaopt.trial import Trial


class SomeObserver(_Observer[Tuple[HyperparamsRepository, Trial]]):
    def __init__(self):
        self.events: List[Trial] = []

    def on_next(self, value: Tuple[HyperparamsRepository, Trial]):
        self.events.append(value)


def test_in_memory_hyperparams_repository_should_be_observable():
    repo: InMemoryHyperparamsRepository = InMemoryHyperparamsRepository()
    # todo: make a tests that asserts that an observer can receive updates from the InMemoryHyperparamsRepository
    # todo: given trial, a repo, and an observer
    pass

    # todo: when repo.subscribe(observer)
    # todo: when repo.save_trial(trial)

    # todo: then observer.events[0] == trial


def test_hyperparams_json_repository_should_be_observable_in_memory():
    # todo: make a tests that asserts that an observer can receive updates from the HyperparamsJSONRepository
    # todo: given trial, a repo, and an observer
    repo: HyperparamsJSONRepository = HyperparamsJSONRepository()

    # todo: when repo.subscribe(observer)
    # todo: when repo.save_trial(trial)

    # todo: then observer.events[0] == trial
    pass


def test_hyperparams_json_repository_should_be_observable_with_file_system_changes():
    # todo: make a tests that asserts that an observer can receive updates from the HyperparamsJSONRepository
    # todo: given trial, a repo, and an observer
    repo: HyperparamsJSONRepository = HyperparamsJSONRepository()

    # todo: when repo.subscribe_to_cache_folder_changes(observer)
    # todo: when repo.save_trial(trial)

    # todo: then observer.events[0] == trial
    pass
