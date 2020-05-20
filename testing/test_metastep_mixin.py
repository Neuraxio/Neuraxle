from neuraxle.pipeline import Pipeline

from neuraxle.base import MetaStep, BaseStep, NonTransformableMixin
from neuraxle.union import Identity


class SomeMetaStep(MetaStep, BaseStep):
    def __init__(self, wrapped: BaseStep):
        BaseStep.__init__(self)
        MetaStep.__init__(self, wrapped)

    def transform(self, data_inputs):
        self.wrapped.transform(data_inputs)


def test_metastepmixin_set_train_should_set_train_to_false():
    p = SomeMetaStep(Pipeline([
        Identity()
    ]))

    p.set_train(False)

    assert not p.is_train
    assert not p.wrapped[0].is_train
    assert not p.wrapped.is_train


def test_metastepmixin_set_train_should_set_train_to_true():
    p = SomeMetaStep(Pipeline([
        Identity()
    ]))

    assert p.is_train
    assert p.wrapped[0].is_train
    assert p.wrapped.is_train


from testing.mocks.step_mocks import SomeMetaStepWithHyperparams

EXPECTED_STR_OUTPUT = """SomeMetaStepWithHyperparams(
	wrapped=SomeStepWithHyperparams(
	name=MockStep,
	hyperparameters=HyperparameterSamples([('learning_rate', 0.1),
                       ('l2_weight_reg', 0.001),
                       ('hidden_size', 32),
                       ('num_layers', 3),
                       ('num_lstm_layers', 1),
                       ('use_xavier_init', True),
                       ('use_max_pool_else_avg_pool', True),
                       ('dropout_drop_proba', 0.5),
                       ('momentum', 0.1)])
),
	hyperparameters=HyperparameterSamples()
)"""


def test_basestep_representation_works_correctly():
    output = str(SomeMetaStepWithHyperparams())
    assert output == EXPECTED_STR_OUTPUT
