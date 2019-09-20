from neuraxle.base import MetaStepMixin, BaseStep
from neuraxle.hyperparams.distributions import RandInt
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace
from testing.test_pipeline import SomeStep

SOME_STEP_HP_KEY = 'somestep_hyperparam'
RAND_INT_SOME_STEP = RandInt(-10, 0)
RAND_INT_META_STEP = RandInt(0, 10)


class SomeMetaStepMixin(MetaStepMixin, BaseStep):
    pass


META_STEP_HP = 'metastep_hyperparam'
SOME_STEP_HP = "SomeStep__somestep_hyperparam"
META_STEP_HP_VALUE = 1
SOME_STEP_HP_VALUE = 2


def test_meta_step_mixin_should_get_hyperparams():
    p = SomeMetaStepMixin(SomeStep())
    p.set_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE,
        SOME_STEP_HP: SOME_STEP_HP_VALUE
    }))

    hyperparams = p.get_hyperparams()

    assert hyperparams[META_STEP_HP] == META_STEP_HP_VALUE
    assert hyperparams[SOME_STEP_HP] == SOME_STEP_HP_VALUE


def test_meta_step_mixin_should_set_hyperparams():
    p = SomeMetaStepMixin(SomeStep())

    p.set_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE,
        SOME_STEP_HP: SOME_STEP_HP_VALUE
    }))

    assert p.hyperparams[META_STEP_HP] == META_STEP_HP_VALUE
    assert p.wrapped.get_hyperparams()['somestep_hyperparam'] == SOME_STEP_HP_VALUE


def test_meta_step_mixin_should_set_hyperparams_space():
    p = SomeMetaStepMixin(SomeStep())

    p.set_hyperparams_space(HyperparameterSpace({
        META_STEP_HP: RAND_INT_META_STEP,
        SOME_STEP_HP: RAND_INT_SOME_STEP
    }))

    assert p.hyperparams_space[META_STEP_HP] == RAND_INT_META_STEP
    assert p.wrapped.hyperparams_space[SOME_STEP_HP_KEY] == RAND_INT_SOME_STEP


def test_meta_step_mixin_should_get_hyperparams_space():
    p = SomeMetaStepMixin(SomeStep())
    p.set_hyperparams_space(HyperparameterSpace({
        META_STEP_HP: RAND_INT_META_STEP,
        SOME_STEP_HP: RAND_INT_SOME_STEP
    }))

    hyperparams_space = p.get_hyperparams_space()

    assert hyperparams_space[META_STEP_HP] == RAND_INT_META_STEP
    assert hyperparams_space[SOME_STEP_HP] == RAND_INT_SOME_STEP
