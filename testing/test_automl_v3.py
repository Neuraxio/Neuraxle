import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, SGDClassifier
from sklearn.metrics import median_absolute_error

from neuraxle.base import BaseService, ExecutionContext, HandleOnlyMixin, Identity, BaseStep, MetaStep, \
    MixinForBaseService, MixinForBaseTransformer, NonFittableMixin, NonTransformableMixin
from neuraxle.hyperparams.distributions import RandInt, Uniform
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.callbacks import ScoringCallback
from neuraxle.pipeline import Pipeline
from neuraxle.steps.data import DataShuffler
from neuraxle.steps.sklearn import SKLearnWrapper


def test_typable_mixin_can_be_typed(tmpdir):
    class MyService(NonFittableMixin, NonTransformableMixin, MixinForBaseService, BaseStep):

        def __init__(self):
            BaseStep.__init__(self)
            MixinForBaseService.__init__(self)
            NonTransformableMixin.__init__(self)
            NonFittableMixin.__init__(self)

    MyService()


@pytest.mark.skip(reason="Not implemented yet")
def test_automl_context_is_correctly_specified_into_trial(tmpdir):
    expected_context = ExecutionContext(
        root=tmpdir,
        feed=AutoMLFeed()
    )

    class StepThatAssertsContextIsSpecified(HandleOnlyMixin):
        def __init__(self, expected_context: ExecutionContext):
            self.context = expected_context

        def fit_transform(self, X, y=None):
            assert self.context.is_specified()
            return X


def test_automl_feed_logs_the_data_of_the_logger():
    pass


def test_automl_feed_logs_the_data_of_the_metrics():
    pass


def test_automl_feed_logs_the_data_of_the_introspection():
    pass


def test_automl_feed_logs_the_data_of_the_status():
    pass


def test_on_disk_repo_is_structured_accordingly():
    pass


def test_basic_automl_scenario():
    pass


def test_automl_will_use_logger_for_each_trial():
    pass


def test_automl_default_loop_does_the_right_number_of_trials():
    pass


def test_automl_rounds_does_the_right_number_of_trials():
    pass


def test_trial_split_collects_good_metadata():
    pass
