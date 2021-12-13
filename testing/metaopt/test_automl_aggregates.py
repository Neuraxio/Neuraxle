from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pytest
from neuraxle.base import (BaseService, BaseStep, BaseTransformer,
                           ExecutionContext, Flow, HandleOnlyMixin, Identity,
                           MetaStep)
from neuraxle.data_container import DataContainer as DACT
from neuraxle.hyperparams.distributions import RandInt, Uniform
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace)
from neuraxle.metaopt.auto_ml import AutoML, DefaultLoop, RandomSearch, Trainer
from neuraxle.metaopt.callbacks import (CallbackList, EarlyStoppingCallback,
                                        MetricCallback)
from neuraxle.metaopt.data.aggregates import (Client, MetricResults, Project,
                                              Root, Round, Trial, TrialSplit,
                                              aggregate_2_dataclass,
                                              aggregate_2_subaggregate)
from neuraxle.metaopt.data.vanilla import (DEFAULT_CLIENT, DEFAULT_PROJECT,
                                           AutoMLContext, AutoMLFlow,
                                           BaseDataclass,
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
from neuraxle.steps.numpy import AddN, MultiplyByN
from sklearn.metrics import median_absolute_error
from testing.metaopt.test_repo_dataclasses import (SOME_FULL_SCOPED_LOCATION,
                                                   SOME_METRIC_NAME,
                                                   SOME_ROOT_DATACLASS)


@pytest.mark.parametrize(
    "aggregate_class, context", [
        (Root, AutoMLContext),
        (Project, AutoMLContext),
        (Round, AutoMLContext),
        (Client, AutoMLContext),
        (Trial, AutoMLContext),
        (TrialSplit, AutoMLContext),
        (MetricResults, AutoMLContext),
    ]
)
def test_aggregates_creation(aggregate_class, context):
    dataclass_class: Type = aggregate_2_dataclass[aggregate_class]
    scoped_loc: ScopedLocation = SOME_FULL_SCOPED_LOCATION[:dataclass_class]
    context = ExecutionContext()
    context: AutoMLContext = AutoMLContext().from_context(
        context,
        VanillaHyperparamsRepository.from_root(SOME_ROOT_DATACLASS, context.get_path())
    )

    dataclass = SOME_ROOT_DATACLASS[scoped_loc]
    assert dataclass == context.repo.load(scoped_loc, deep=True)

    aggregate = aggregate_class(SOME_ROOT_DATACLASS, context.push_attrs(scoped_loc.popped()))

    assert isinstance(aggregate, dataclass_class)
