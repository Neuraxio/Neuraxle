import numpy as np
import pytest

from neuraxle.base import ForceHandleMixin, Identity, BaseStep, ExecutionContext
from neuraxle.data_container import DataContainer
from neuraxle.pipeline import Pipeline


class BadForceHandleStep(ForceHandleMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        ForceHandleMixin.__init__(self)


class ForceHandleIdentity(ForceHandleMixin, Identity):
    def __init__(self):
        Identity.__init__(self)
        ForceHandleMixin.__init__(self)


def test_raises_exception_if_method_not_redefined(tmpdir):
    # Now that I think about it, this really just is a complicated way to test the self._ensure_method_overriden function.
    with pytest.raises(NotImplementedError) as exception_info:
        BadForceHandleStep()

    assert "_fit_data_container" in exception_info.value.args[0]

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> '_FittableStep':
        return self

    BadForceHandleStep._fit_data_container = _fit_data_container

    with pytest.raises(NotImplementedError) as exception_info:
        BadForceHandleStep()

    assert "_fit_transform_data_container" in exception_info.value.args[0]

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> \
            ('BaseStep', DataContainer):
        return self, data_container

    BadForceHandleStep._fit_data_container = _fit_data_container

    with pytest.raises(NotImplementedError) as exception_info:
        BadForceHandleStep()

    assert "_transform_data_container" in exception_info.value.args[0]


def test_forcehandleidentity_does_not_crash(tmpdir):
    p = Pipeline([
        ForceHandleIdentity()
    ])
    data_inputs = np.array([0, 1, 2, 3])
    expected_outputs = data_inputs * 2
    p.fit(data_inputs, expected_outputs)
    p.fit_transform(data_inputs, expected_outputs)
    p.transform(data_inputs=data_inputs)

# def test_automl(tmpdir):
#     data_inputs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     expected_outputs = data_inputs * 2
#     p = Pipeline([
#         Identity()
#     ])
#     auto_ml = _make_autoML_loop(tmpdir, p)
#     auto_ml.fit(data_inputs, expected_outputs)
#     auto_ml.fit_transform(data_inputs, expected_outputs)
#     auto_ml.transform(data_inputs)
#
#
# #
# def _make_autoML_loop(tmpdir, p: Pipeline):
#     from neuraxle.metaopt.auto_ml import InMemoryHyperparamsRepository
#     from neuraxle.metaopt.auto_ml import AutoML
#     from neuraxle.metaopt.auto_ml import RandomSearchHyperparameterSelectionStrategy
#     from neuraxle.metaopt.auto_ml import ValidationSplitter
#     from neuraxle.metaopt.callbacks import ScoringCallback
#     from sklearn.metrics import mean_squared_error
#
#     hp_repository = InMemoryHyperparamsRepository(cache_folder=str(tmpdir) + "_hp")
#     n_epochs = 1
#
#     return AutoML(
#         pipeline=p,
#         hyperparams_optimizer=RandomSearchHyperparameterSelectionStrategy(),
#         validation_splitter=ValidationSplitter(0.20),
#         scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
#         n_trials=1,
#         refit_trial=True,
#         epochs=n_epochs,
#         hyperparams_repository=hp_repository,
#         cache_folder_when_no_handle=str(tmpdir)
#     )
