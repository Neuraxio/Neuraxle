from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pytest
from neuraxle.base import (BaseService, BaseStep, BaseTransformer,
                           CX, Flow, HandleOnlyMixin, Identity,
                           MetaStep)
from neuraxle.data_container import IDT
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.distributions import RandInt, Uniform
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace)
from neuraxle.metaopt.auto_ml import ControlledAutoML, DefaultLoop, RandomSearchSampler, Trainer
from neuraxle.metaopt.callbacks import (ARG_X_INPUTTED, ARG_Y_EXPECTED,
                                        ARG_Y_PREDICTD, CallbackList,
                                        EarlyStoppingCallback, MetricCallback)
from neuraxle.metaopt.data.aggregates import (BaseAggregate, Client,
                                              MetricResults, Project, Root,
                                              Round, Trial, TrialSplit,
                                              aggregate_2_dataclass,
                                              aggregate_2_subaggregate)
from neuraxle.metaopt.data.vanilla import (DEFAULT_CLIENT, DEFAULT_PROJECT,
                                           AutoMLContext, BaseDataclass,
                                           BaseHyperparameterOptimizer,
                                           ClientDataclass,
                                           MetricResultsDataclass,
                                           ProjectDataclass, RootDataclass,
                                           RoundDataclass, ScopedLocation,
                                           TrialDataclass, TrialSplitDataclass,
                                           VanillaHyperparamsRepository,
                                           dataclass_2_id_attr,
                                           dataclass_2_subdataclass)
from neuraxle.metaopt.validation import (GridExplorationSampler,
                                         ValidationSplitter)
from neuraxle.pipeline import Pipeline
from neuraxle.steps.data import DataShuffler
from neuraxle.steps.flow import TrainOnlyWrapper
from neuraxle.steps.numpy import AddN, MultiplyByN
from sklearn.metrics import median_absolute_error
from testing.metaopt.test_repo_dataclasses import (SOME_FULL_SCOPED_LOCATION,
                                                   SOME_METRIC_NAME,
                                                   SOME_ROOT_DATACLASS)


def test_scoped_cascade_does_the_right_logging(tmpdir):
    dact_train: DACT[IDT, ARG_X_INPUTTED, ARG_Y_PREDICTD] = DACT(
        ids=list(range(0, 10)), di=list(range(0, 10)), eo=list(range(100, 110)))
    dact_valid: DACT[IDT, ARG_X_INPUTTED, ARG_Y_PREDICTD] = DACT(
        ids=list(range(10, 20)), di=list(range(10, 20)), eo=list(range(110, 120)))
    hp_optimizer: BaseHyperparameterOptimizer = GridExplorationSampler(expected_n_trials=1)
    n_epochs = 3
    callbacks = CallbackList([
        MetricCallback('MAE', median_absolute_error, False),
        EarlyStoppingCallback(1, 'MAE')
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
    root: Root = Root.vanilla(CX())

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
            assert len(dc.get_sublocation()) == n_epochs
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
def test_aggregates_creation(aggregate_class: Type[BaseAggregate], is_deep):
    # Create repo from deep root DC:
    dataclass_class: Type = aggregate_2_dataclass[aggregate_class]
    scoped_loc: ScopedLocation = SOME_FULL_SCOPED_LOCATION[:dataclass_class]
    context = CX()
    context: AutoMLContext = AutoMLContext().from_context(
        context,
        VanillaHyperparamsRepository.from_root(SOME_ROOT_DATACLASS, context.get_path())
    )

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
