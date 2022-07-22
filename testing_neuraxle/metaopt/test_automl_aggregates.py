import time
from typing import Callable, Optional, Type

import pytest
from neuraxle.base import BaseStep
from neuraxle.base import ExecutionContext as CX
from neuraxle.base import NonFittableMixin
from neuraxle.data_container import ARG_X_INPUTTED, ARG_Y_PREDICTD, IDT
from neuraxle.data_container import DataContainer as DACT
from neuraxle.data_container import PredsDACT, TrainDACT
from neuraxle.distributed.streaming import ParallelQueuedFeatureUnion
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.callbacks import (CallbackList, EarlyStoppingCallback,
                                        MetricCallback)
from neuraxle.metaopt.context import AutoMLContext
from neuraxle.metaopt.data.aggregates import (BaseAggregate, Client,
                                              MetricResults, Project, Root,
                                              Round, Trial, TrialSplit,
                                              aggregate_2_dataclass,
                                              aggregate_2_subaggregate)
from neuraxle.metaopt.data.vanilla import (DEFAULT_CLIENT, DEFAULT_PROJECT,
                                           DEFAULT_ROUND, BaseDataclass,
                                           MetricResultsDataclass,
                                           ScopedLocation, dataclass_2_id_attr,
                                           dataclass_2_subdataclass)
from neuraxle.metaopt.optimizer import (BaseHyperparameterOptimizer,
                                        GridExplorationSampler,
                                        RandomSearchSampler)
from neuraxle.metaopt.repositories.json import HyperparamsOnDiskRepository
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import AddN, MultiplyByN
from sklearn.metrics import median_absolute_error
from testing_neuraxle.metaopt.test_automl_dataclasses import (
    SOME_FULL_SCOPED_LOCATION, SOME_ROOT_DATACLASS)
from testing_neuraxle.metaopt.test_automl_repositories import (
    CX_WITH_REPO_CTORS, TmpDir)


class SomeException(Exception):
    pass


def _raise_if_is_at_level(aggregate: BaseAggregate, level_type_to_raise: Type[BaseAggregate], current_phase: str, phase_to_raise: str):
    if isinstance(aggregate, level_type_to_raise) and phase_to_raise == current_phase:
        raise SomeException(f"==> Raising at `{phase_to_raise}`of `{level_type_to_raise.__name__}` as planned. <==\n"
                            f"(This should not be catched).")
    else:
        return


BEGIN = "beginning"
END = "the end"


@pytest.mark.parametrize('lvlno, level_to_raise', list(enumerate(list(aggregate_2_subaggregate.keys())[1:])))
@pytest.mark.parametrize('phase_to_raise', [BEGIN, END])
@pytest.mark.parametrize('cx_repo_ctor', CX_WITH_REPO_CTORS)
def test_aggregate_exceptions_will_raise(
    lvlno: int, level_to_raise: Type[BaseAggregate],
    phase_to_raise: str,
    cx_repo_ctor: Callable[[Optional[TmpDir]], AutoMLContext],
):

    with pytest.raises(SomeException):
        root: Root = Root.vanilla(cx_repo_ctor())
        _raise_if_is_at_level(root, level_to_raise, BEGIN, phase_to_raise)
        with root.default_project() as ps:
            ps: Project = ps
            _raise_if_is_at_level(ps, level_to_raise, BEGIN, phase_to_raise)
            with ps.default_client() as cs:
                cs: Client = cs
                _raise_if_is_at_level(cs, level_to_raise, BEGIN, phase_to_raise)
                with cs.new_round(main_metric_name="some_metric") as rs:
                    rs: Round = rs.with_optimizer(GridExplorationSampler(1), HyperparameterSpace())
                    _raise_if_is_at_level(rs, level_to_raise, BEGIN, phase_to_raise)
                    with rs.new_rvs_trial() as ts:
                        ts: Trial = ts
                        _raise_if_is_at_level(ts, level_to_raise, BEGIN, phase_to_raise)
                        with ts.new_validation_split(continue_loop_on_error=False) as tss:
                            tss: TrialSplit = tss.with_n_epochs(1)
                            _raise_if_is_at_level(tss, level_to_raise, BEGIN, phase_to_raise)
                            with tss.managed_metric(rs.main_metric_name, True) as mrs:
                                mrs: MetricResults = mrs
                                _raise_if_is_at_level(mrs, level_to_raise, BEGIN, phase_to_raise)

                                mrs.add_train_result(0.0)
                                mrs.add_valid_result(0.0)

                                _raise_if_is_at_level(mrs, level_to_raise, END, phase_to_raise)
                            _raise_if_is_at_level(tss, level_to_raise, END, phase_to_raise)
                        _raise_if_is_at_level(ts, level_to_raise, END, phase_to_raise)
                    _raise_if_is_at_level(rs, level_to_raise, END, phase_to_raise)
                _raise_if_is_at_level(cs, level_to_raise, END, phase_to_raise)
            _raise_if_is_at_level(ps, level_to_raise, END, phase_to_raise)
        _raise_if_is_at_level(root, level_to_raise, END, phase_to_raise)
    pass


@pytest.mark.parametrize('cx_repo_ctor', CX_WITH_REPO_CTORS)
def test_scoped_cascade_does_the_right_logging(tmpdir, cx_repo_ctor: Callable[[Optional[TmpDir]], AutoMLContext]):
    dact_train: DACT[IDT, ARG_X_INPUTTED, ARG_Y_PREDICTD] = DACT(
        ids=list(range(0, 10)), di=list(range(0, 10)), eo=list(range(100, 110)))
    dact_valid: DACT[IDT, ARG_X_INPUTTED, ARG_Y_PREDICTD] = DACT(
        ids=list(range(10, 20)), di=list(range(10, 20)), eo=list(range(110, 120)))
    hp_optimizer: BaseHyperparameterOptimizer = GridExplorationSampler(expected_n_trials=1)
    n_epochs = 3
    n_early_stopping_sentinel = 1
    n_effective_epochs = min(n_epochs, 1 + n_early_stopping_sentinel)
    callbacks = CallbackList([
        MetricCallback('MAE', median_absolute_error, False),
        EarlyStoppingCallback(n_early_stopping_sentinel, 'MAE')
    ])
    expected_scope = ScopedLocation(
        project_name=DEFAULT_PROJECT,
        client_name=DEFAULT_CLIENT,
        round_number=0,
        trial_number=0,
        split_number=0,
        metric_name='MAE',
    )
    p = Pipeline([
        MultiplyByN().with_hp_range(range(1, 3)),
        AddN().with_hp_range(range(99, 103)),
    ])
    hps: HyperparameterSpace = p.get_hyperparams_space()
    root: Root = Root.vanilla(cx_repo_ctor())

    with root.default_project() as ps:
        ps: Project = ps
        with ps.default_client() as cs:
            cs: Client = cs
            with cs.new_round(main_metric_name='MAE') as rs:
                rs: Round = rs.with_optimizer(hp_optimizer=hp_optimizer, hp_space=hps)
                with rs.new_rvs_trial() as ts:
                    ts: Trial = ts
                    with ts.new_validation_split(continue_loop_on_error=False) as tss:
                        tss: TrialSplit = tss.with_n_epochs(n_epochs)

                        for _ in range(n_epochs):
                            e = tss.next_epoch()

                            p = p.handle_fit(
                                dact_train.copy(),
                                tss.train_context())

                            eval_dact_train = p.handle_predict(
                                dact_train.without_eo(),
                                tss.validation_context())
                            eval_dact_valid = p.handle_predict(
                                dact_valid.without_eo(),
                                tss.validation_context())

                            if callbacks.call(
                                tss,
                                eval_dact_train.with_eo(dact_train.eo),
                                eval_dact_valid.with_eo(dact_valid.eo),
                                e == n_epochs
                            ):
                                break

    for dc_type in dataclass_2_id_attr.keys():
        dc: BaseDataclass = root.repo.load(expected_scope[:dc_type])
        assert isinstance(dc, dc_type)
        assert dc.get_id() == expected_scope[dc_type]

        if dc_type == MetricResultsDataclass:
            assert len(dc.get_sublocation()) == n_effective_epochs
        else:
            assert len(dc.get_sublocation()) == 1


@pytest.mark.parametrize("_type", list(aggregate_2_dataclass.keys())[:-1])
def test_ensure_consistent_subtypes_lozenge(_type: BaseAggregate):
    # subdataclass from dataclass_2_subdataclass:
    _dataclass = aggregate_2_dataclass[_type]
    _subdataclass_dataclass = dataclass_2_subdataclass[_dataclass]
    # subdataclass from aggregate_2_subaggregate (lozenge):
    _subaggregate = aggregate_2_subaggregate[_type]
    _subdataclass_agg = aggregate_2_dataclass[_subaggregate]

    # pre-requisite type side assertions:
    assert issubclass(_type, BaseAggregate)
    assert issubclass(_subaggregate, BaseAggregate)
    assert issubclass(_dataclass, BaseDataclass)
    assert issubclass(_subdataclass_agg, BaseDataclass)
    assert issubclass(_subdataclass_dataclass, BaseDataclass)
    # final lozenge consistency assertion:
    assert _subdataclass_dataclass == _subdataclass_agg


cant_be_shallow_aggs = [
    TrialSplit,
    MetricResults,
]


@pytest.mark.parametrize("aggregate_class", list(aggregate_2_subaggregate.keys())[1:])
@pytest.mark.parametrize("is_deep", [True, False])
@pytest.mark.parametrize('cx_repo_ctor', CX_WITH_REPO_CTORS)
def test_aggregates_creation(
    aggregate_class: Type[BaseAggregate],
    is_deep: bool,
    cx_repo_ctor: Callable[[Optional[TmpDir]], AutoMLContext]
):
    # Create repo from deep root DC:
    dataclass_class: Type = aggregate_2_dataclass[aggregate_class]
    scoped_loc: ScopedLocation = SOME_FULL_SCOPED_LOCATION[:dataclass_class]
    context = cx_repo_ctor()
    context.repo.save(SOME_ROOT_DATACLASS, ScopedLocation(), deep=True)

    # Retrieve scoped DC from root
    dataclass = SOME_ROOT_DATACLASS[scoped_loc]
    if not is_deep:
        dataclass = dataclass.shallow()

    # It shouldbe the same from loading it from the repo:
    assert dataclass == context.repo.load(scoped_loc, deep=is_deep)

    # Finally create aggregate from the DC loaded above:
    if aggregate_class in cant_be_shallow_aggs and not is_deep:
        with pytest.raises(AssertionError):
            aggregate = aggregate_class(dataclass, context.with_loc(
                scoped_loc.popped()), is_deep=is_deep)
        return  # skip the rest of the test when class can't be shallow
    else:
        aggregate = aggregate_class(dataclass, context.with_loc(
            scoped_loc.popped()), is_deep=is_deep)

    # Its data should stay intact right after construction:
    assert aggregate._dataclass == dataclass
    assert isinstance(aggregate, aggregate_class)

    # Save aggregate to repo, shallowly:
    aggregate.save(deep=False)

    # try to load its dataclass back to ensure consistency
    assert aggregate._dataclass == context.repo.load(scoped_loc, deep=is_deep)


class RandomSearchSamplerThatSleeps(RandomSearchSampler):

    def __init__(self, sleep_time_secs: int):
        RandomSearchSampler.__init__(self)
        self.sleep_time_secs: int = sleep_time_secs
        self.called = False

    def find_next_best_hyperparams(self, *args, **kwargs):
        time.sleep(self.sleep_time_secs)
        self.called = True
        return RandomSearchSampler.find_next_best_hyperparams(self, *args, **kwargs)


class NewTrialAtTransform(NonFittableMixin, BaseStep):
    def __init__(self, round_loc: ScopedLocation, sleep_time_secs: int):
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)
        self.sleep_time_secs: int = sleep_time_secs
        self.round_loc: ScopedLocation = round_loc

    def _transform_data_container(self, data_container: TrainDACT, context: CX) -> PredsDACT:
        _round: Round = Round.from_context(context.with_loc(self.round_loc))
        _round.with_optimizer(RandomSearchSamplerThatSleeps(self.sleep_time_secs), HyperparameterSpace())
        assert _round.hp_optimizer.called is False
        with _round.new_rvs_trial() as ts:
            assert _round.hp_optimizer.called is True
            ts: Trial = ts
            time.sleep(self.sleep_time_secs)
        return data_container

    def transform(self, data_inputs: ARG_X_INPUTTED) -> ARG_Y_PREDICTD:
        raise NotImplementedError("Must use handle_transform method.")


def test_trial_aggregates_wait_before_locking_in_parallel_processes():
    # setup
    tmpdir = CX.get_new_cache_folder()
    cx = AutoMLContext.from_context(repo=HyperparamsOnDiskRepository(tmpdir))
    loc = ScopedLocation.default(DEFAULT_ROUND)
    step = NewTrialAtTransform(loc, sleep_time_secs=1)
    batch_size = 10
    one_batch = DACT(di=list(range(batch_size)))
    n_trials = 3

    # act
    stream = ParallelQueuedFeatureUnion([step] * n_trials, n_workers_per_step=1, batch_size=batch_size, use_processes=True)
    stream.handle_transform(one_batch, cx)
    # # Here is the non-parallel equivalent, for debugging:
    # for _ in range(n_trials):
    #     step.handle_transform(one_batch, cx)

    # assert
    round_dc = cx.repo.load(loc, deep=True)
    assert len(round_dc) == n_trials
