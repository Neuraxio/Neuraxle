from neuraxle.pipeline import Pipeline

from neuraxle.base import MetaStepMixin, BaseStep, NonFittableMixin, NonTransformableMixin
from neuraxle.union import Identity


class SomeMetaStep(NonFittableMixin, MetaStepMixin, BaseStep):
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
