from typing import Generic, TypeVar, Generic
import typing
from neuraxle.pipeline import Pipeline
from neuraxle.base import BaseStep, MetaStepMixin, MetaStep, NonFittableMixin, BaseService
from neuraxle.union import Identity

from testing.mocks.step_mocks import SomeMetaStepWithHyperparams


def test_metastepmixin_set_train_should_set_train_to_false():
    p = MetaStep(Pipeline([
        Identity()
    ]))

    p.set_train(False)

    assert not p.is_train
    assert not p.wrapped[0].is_train
    assert not p.wrapped.is_train


def test_metastepmixin_set_train_should_set_train_to_true():
    p = MetaStep(Pipeline([
        Identity()
    ]))

    assert p.is_train
    assert p.wrapped[0].is_train
    assert p.wrapped.is_train


def test_basestep_str_representation_works_correctly():
    output = str(SomeMetaStepWithHyperparams())
    assert output == "SomeMetaStepWithHyperparams(SomeStepWithHyperparams(name='MockStep'), name='SomeMetaStepWithHyperparams')"


def test_subtyping_of_metastep_works_correctly():
    some_step: MetaStep[Identity] = MetaStep(Identity())

    assert issubclass(MetaStep, Generic)
    assert isinstance(some_step, MetaStep)
    assert isinstance(some_step.get_step(), Identity)


def test_typable_mixin_can_hold_type_annotation(tmpdir):
    # Testing the type annotation "MetaStep[MyService]":
    wrapped_service: MetaStep[Identity] = MetaStep(Identity())

    g: Generic = wrapped_service.__orig_bases__[-1]
    assert isinstance(wrapped_service.get_step(), g.__parameters__[0].__bound__)
    bt: TypeVar = typing.get_args(g)[0]
    assert isinstance(wrapped_service.get_step(), bt.__bound__)

    assert isinstance(wrapped_service.get_step(), Identity)
    assert isinstance(wrapped_service.get_step(), NonFittableMixin)
    assert isinstance(wrapped_service.get_step(), BaseService)
