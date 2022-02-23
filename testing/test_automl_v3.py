from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from operator import attrgetter
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import numpy as np
import pytest
from neuraxle.base import (CX, BaseService, BaseStep, BaseTransformer, Flow,
                           HandleOnlyMixin, Identity, MetaStep)
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.distributions import DiscreteHyperparameterDistribution, RandInt, Uniform, PriorityChoice
from neuraxle.hyperparams.space import (FlatDict, HyperparameterSamples,
                                        HyperparameterSpace)
from neuraxle.metaopt.auto_ml import (ControlledAutoML, DefaultLoop,
                                      RandomSearchSampler, Trainer)
from neuraxle.metaopt.callbacks import (CallbackList, EarlyStoppingCallback,
                                        MetricCallback)
from neuraxle.metaopt.data.aggregates import (Client, Project, Root, Round,
                                              Trial, TrialSplit)
from neuraxle.metaopt.data.vanilla import (DEFAULT_CLIENT, DEFAULT_PROJECT,
                                           AutoMLContext, BaseDataclass,
                                           BaseHyperparameterOptimizer,
                                           ClientDataclass,
                                           MetricResultsDataclass,
                                           ProjectDataclass, RootDataclass,
                                           RoundDataclass, ScopedLocation,
                                           TrialDataclass, TrialSplitDataclass,
                                           VanillaHyperparamsRepository,
                                           dataclass_2_id_attr)
from neuraxle.metaopt.validation import (GridExplorationSampler,
                                         ValidationSplitter)
from neuraxle.pipeline import Pipeline
from neuraxle.steps.data import DataShuffler
from neuraxle.steps.flow import TrainOnlyWrapper
from neuraxle.steps.misc import AssertFalseStep
from neuraxle.steps.numpy import AddN, MultiplyByN
from sklearn.metrics import median_absolute_error


class StepThatAssertsContextIsSpecifiedAtTrain(Identity):
    def __init__(self, expected_loc: ScopedLocation, up_to_dc: Type[BaseDataclass] = RoundDataclass):
        BaseStep.__init__(self)
        HandleOnlyMixin.__init__(self)
        self.expected_loc = expected_loc
        self.up_to_dc: Type[BaseDataclass] = up_to_dc

    def _did_process(self, data_container: DACT, context: CX) -> DACT:
        if self.is_train:
            context: AutoMLContext = context  # typing annotation for IDE
            self._assert_equals(
                self.expected_loc[:self.up_to_dc], context.loc[:self.up_to_dc],
                f'Context is not at the expected location. '
                f'Expected {self.expected_loc}, got {context.loc}.',
                context)
            self._assert_equals(
                context.loc in context.repo.root, True,
                "Context should have the dataclass, but it doesn't", context)
        return data_container


def test_automl_context_is_correctly_specified_into_trial_with_full_automl_scenario(tmpdir):
    # This is a large test
    dact = DACT(di=list(range(10)), eo=list(range(10, 20)))
    cx = CX(root=tmpdir)
    expected_deep_cx_loc = ScopedLocation.default(0, 0, 0)
    assertion_step = StepThatAssertsContextIsSpecifiedAtTrain(expected_loc=expected_deep_cx_loc)
    automl: ControlledAutoML[Pipeline] = _create_automl_test_loop(tmpdir, assertion_step)
    automl = automl.handle_fit(dact, cx)

    pred: DACT = automl.handle_predict(dact.without_eo(), cx)

    round: Round = automl.get_automl_context(
        cx).with_loc(ScopedLocation.default(round_number=0)).load_agg()
    best: Tuple[float, int, FlatDict] = round.best_result_summary()
    best_score = best[0]
    assert best_score == 0
    best_add_n: int = list(best[-1].values())[0]
    assert best_add_n == 10  # We expect the AddN step to make use of the value "10" for 0 MAE error.
    assert np.array_equal(list(pred.di), list(dact.eo))


def test_automl_step_can_interrupt_on_fail_with_full_automl_scenario(tmpdir):
    # This is a large test
    dact = DACT(di=list(range(10)), eo=list(range(10, 20)))
    cx = CX(root=tmpdir)
    assertion_step = AssertFalseStep()
    automl = _create_automl_test_loop(tmpdir, assertion_step)

    with pytest.raises(AssertionError):
        automl.handle_fit(dact, cx)


def _create_automl_test_loop(tmpdir, assertion_step: BaseStep, n_trials: int = 4):
    automl = ControlledAutoML(
        pipeline=Pipeline([
            TrainOnlyWrapper(DataShuffler()),
            AddN().with_hp_range(range(8, 12)),
            assertion_step
        ]),
        loop=DefaultLoop(
            trainer=Trainer(
                validation_splitter=ValidationSplitter(validation_size=0.2),
                n_epochs=4,
                callbacks=[MetricCallback('MAE', median_absolute_error, False)],
            ),
            hp_optimizer=GridExplorationSampler(n_trials),
            n_trials=n_trials,
            continue_loop_on_error=False,
            n_jobs=2,
        ),
        repo=VanillaHyperparamsRepository(tmpdir),
        main_metric_name='MAE',
        start_new_round=True,
        refit_best_trial=True,
    )

    return automl


@pytest.mark.parametrize('n_trials', [1, 3, 4, 8, 12, 13, 16, 17, 20])
def test_grid_sampler_fulls_grid(n_trials):
    round, ges = _get_optimization_scenario(n_trials)

    tried: Set[Dict[str, Any]] = set()
    for i in range(n_trials):
        hp = ges.find_next_best_hyperparams(round).to_flat_dict()
        round.add_testing_optim_result(hp)
        flat = frozenset(hp.items())

        assert flat not in tried, f"Found the same hyperparameter samples twice: `{flat}`, with {i+1} past trials."
        tried.add(flat)
    assert len(tried) == n_trials


@pytest.mark.parametrize('n_trials', [1, 3, 4, 12, 13, 16, 17, 20, 40, 50, 100])
def test_grid_sampler_fulls_individual_params(n_trials):
    round, ges = _get_optimization_scenario(n_trials=n_trials)
    ges: GridExplorationSampler = ges  # typing
    round: Round = round  # typing

    tried_params: Dict[str, Set[Any]] = defaultdict(set)
    for i in range(n_trials):
        hp = ges.find_next_best_hyperparams(round).to_flat_dict()
        round.add_testing_optim_result(hp)
        for hp_k, value_set in hp.items():
            tried_params[hp_k].add(value_set)

    for hp_k, value_set in tried_params.items():
        hp_k: str = hp_k  # typing
        value_set: Set[Any] = value_set  # typing

        if isinstance(round.hp_space[hp_k], DiscreteHyperparameterDistribution):
            round_numbs = min(n_trials, len(round.hp_space[hp_k].values()))
            assert len(value_set) == round_numbs
        else:
            round_numbs = min(n_trials, len(ges.flat_hp_grid_values[hp_k]))
            assert len(value_set) >= round_numbs, (
                f"value_set={value_set} has a len={len(value_set)}, but len={round_numbs} was expected.")


@dataclass
class RoundStub:
    hp_space: HyperparameterSpace = field(default_factory=HyperparameterSpace)
    _all_tried_hyperparams: List[FlatDict] = field(default_factory=list)

    def add_testing_optim_result(self, rdict: HyperparameterSamples):
        self._all_tried_hyperparams.append(rdict)

    def get_all_hyperparams(self):
        return self._all_tried_hyperparams


def _get_optimization_scenario(n_trials):
    round: Round = RoundStub(hp_space=HyperparameterSpace({
        'a__add_n': Uniform(0, 4),
        'b__multiply_n': RandInt(0, 4),
        'c__Pchoice': PriorityChoice(["one", "two"]),
    }))
    ges = GridExplorationSampler(n_trials)
    return round, ges


def test_parallel_automl_can_contribute_to_the_same_hp_repository():
    assert False


def test_parallel_automl_can_keep_all_trials_to_force_refit_best_trial(tmpdir):
    # AutoML.to_force_refit_best_trial
    assert False

def test_on_disk_repo_is_structured_accordingly():
    assert False
