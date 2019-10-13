from neuraxle.pipeline import Pipeline

from neuraxle.base import MetaStepMixin, BaseStep, NonFittableMixin, NonTransformableMixin
from neuraxle.union import Identity


class SomeMetaStep(NonFittableMixin, MetaStepMixin, BaseStep):
    def __init__(self, wrapped: BaseStep, train):
        BaseStep.__init__(self, train=train)
        MetaStepMixin.__init__(self, wrapped)

    def transform(self, data_inputs):
        self.wrapped.transform(data_inputs)


def test_metastepmixin_set_train_should_set_train_to_false():
    p = SomeMetaStep(Pipeline([
        Identity()
    ]), train=True)

    p.set_train(False)

    assert not p.train
    assert not p.wrapped[0].train
    assert not p.wrapped.train


def test_metastepmixin_set_train_should_set_train_to_true():
    p = SomeMetaStep(Pipeline([
        Identity()
    ]), train=False)

    p.set_train(True)

    assert p.train
    assert p.wrapped[0].train
    assert p.wrapped.train
