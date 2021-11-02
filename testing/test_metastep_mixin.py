from typing import Generic
from neuraxle.pipeline import Pipeline
from neuraxle.base import BaseStep, MetaStepMixin, BaseServiceT
from neuraxle.union import Identity

from testing.mocks.step_mocks import SomeMetaStepWithHyperparams


class SomeMetaStep(MetaStepMixin, BaseStep):
    def __init__(self, wrapped: BaseStep):
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)

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


def test_basestep_str_representation_works_correctly():
    output = str(SomeMetaStepWithHyperparams())
    assert output == "SomeMetaStepWithHyperparams(SomeStepWithHyperparams(name='MockStep'), name='SomeMetaStepWithHyperparams')"


def test_subtyping_of_metastep_works_correctly():
    some_step: SomeMetaStep[Identity] = SomeMetaStep(Identity())

    assert issubclass(SomeMetaStep, Generic)
    assert isinstance(some_step, SomeMetaStep)
    assert isinstance(some_step.get_step(), Identity)
