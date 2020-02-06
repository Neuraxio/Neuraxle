from neuraxle.base import MetaStepMixin, BaseStep, NonFittableMixin, NonTransformableMixin
from neuraxle.hyperparams.distributions import RandInt, Boolean
from neuraxle.hyperparams.space import HyperparameterSpace, HyperparameterSamples
from neuraxle.steps.loop import StepClonerForEachDataInput
from testing.test_pipeline import SomeStep

SOME_STEP_HP_KEY = 'somestep_hyperparam'
RAND_INT_SOME_STEP = RandInt(-10, 0)
RAND_INT_STEP_CLONER = RandInt(0, 10)

META_STEP_HP = 'metastep_hyperparam'
SOME_STEP_HP = "SomeStep__somestep_hyperparam"
META_STEP_HP_VALUE = 1
SOME_STEP_HP_VALUE = 2

HYPE_SPACE = HyperparameterSpace({
    "a__test": Boolean()
})

HYPE_SAMPLE = HyperparameterSamples({
    "a__test": True
})


class SomeMetaStepMixin(NonTransformableMixin, NonFittableMixin, MetaStepMixin, BaseStep):
    pass


class SomeStepInverseTransform(SomeStep):
    def fit_transform(self, data_inputs, expected_outputs=None):
        return self, 'fit_transform'

    def inverse_transform(self, processed_outputs):
        return 'inverse_transform'


def test_step_cloner_should_get_hyperparams():
    p = StepClonerForEachDataInput(SomeStep())
    p.set_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE,
        SOME_STEP_HP: SOME_STEP_HP_VALUE
    }))

    hyperparams = p.get_hyperparams()

    assert hyperparams[META_STEP_HP] == META_STEP_HP_VALUE
    assert hyperparams[SOME_STEP_HP] == SOME_STEP_HP_VALUE


def test_step_cloner_should_set_hyperparams():
    p = StepClonerForEachDataInput(SomeStep())

    p.set_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE,
        SOME_STEP_HP: SOME_STEP_HP_VALUE
    }))

    assert isinstance(p.hyperparams, HyperparameterSamples)
    assert p.hyperparams[META_STEP_HP] == META_STEP_HP_VALUE
    assert p.get_step().get_hyperparams()[SOME_STEP_HP_KEY] == SOME_STEP_HP_VALUE


def test_step_cloner_update_hyperparams_should_update_step_cloner_hyperparams():
    p = StepClonerForEachDataInput(SomeStep())
    p.set_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE,
        SOME_STEP_HP: SOME_STEP_HP_VALUE
    }))

    p.update_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE + 1,
    }))

    assert isinstance(p.hyperparams, HyperparameterSamples)
    assert p.hyperparams[META_STEP_HP] == META_STEP_HP_VALUE + 1
    assert p.get_step().get_hyperparams()[SOME_STEP_HP_KEY] == SOME_STEP_HP_VALUE


def test_step_cloner_update_hyperparams_should_update_wrapped_step_hyperparams():
    p = StepClonerForEachDataInput(SomeStep())
    p.set_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE,
        SOME_STEP_HP: SOME_STEP_HP_VALUE
    }))

    p.update_hyperparams(HyperparameterSamples({
        SOME_STEP_HP: SOME_STEP_HP_VALUE + 1,
    }))

    assert isinstance(p.hyperparams, HyperparameterSamples)
    assert p.hyperparams[META_STEP_HP] == META_STEP_HP_VALUE
    assert p.get_step().get_hyperparams()[SOME_STEP_HP_KEY] == SOME_STEP_HP_VALUE + 1


def test_step_cloner_should_set_steps_hyperparams():
    p = StepClonerForEachDataInput(SomeStep())

    p.set_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE,
        SOME_STEP_HP: SOME_STEP_HP_VALUE
    }))

    assert isinstance(p.hyperparams, HyperparameterSamples)
    assert isinstance(p.get_step().hyperparams, HyperparameterSamples)
    assert p.get_step().get_hyperparams()[SOME_STEP_HP_KEY] == SOME_STEP_HP_VALUE


def test_step_cloner_should_set_steps_hyperparams_space():
    p = StepClonerForEachDataInput(SomeStep())

    p.set_hyperparams_space(HyperparameterSpace({
        META_STEP_HP: RAND_INT_STEP_CLONER,
        SOME_STEP_HP: RAND_INT_SOME_STEP
    }))

    assert isinstance(p.get_step().hyperparams_space, HyperparameterSpace)
    assert p.get_step().hyperparams_space[SOME_STEP_HP_KEY] == RAND_INT_SOME_STEP


def test_step_cloner_should_set_hyperparams_space():
    p = StepClonerForEachDataInput(SomeStep())

    p.set_hyperparams_space(HyperparameterSpace({
        META_STEP_HP: RAND_INT_STEP_CLONER,
        SOME_STEP_HP: RAND_INT_SOME_STEP
    }))

    assert isinstance(p.hyperparams_space, HyperparameterSpace)
    assert p.hyperparams_space[META_STEP_HP] == RAND_INT_STEP_CLONER
    assert p.get_step().hyperparams_space[SOME_STEP_HP_KEY] == RAND_INT_SOME_STEP


def test_step_cloner_should_get_hyperparams_space():
    p = StepClonerForEachDataInput(SomeStep())
    p.set_hyperparams_space(HyperparameterSpace({
        META_STEP_HP: RAND_INT_STEP_CLONER,
        SOME_STEP_HP: RAND_INT_SOME_STEP
    }))

    hyperparams_space = p.get_hyperparams_space()

    assert hyperparams_space[META_STEP_HP] == RAND_INT_STEP_CLONER
    assert hyperparams_space[SOME_STEP_HP] == RAND_INT_SOME_STEP


RAND_INT_META_STEP = RandInt(0, 10)


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

    assert isinstance(p.hyperparams, HyperparameterSamples)
    assert p.hyperparams[META_STEP_HP] == META_STEP_HP_VALUE
    assert p.get_step().get_hyperparams()['somestep_hyperparam'] == SOME_STEP_HP_VALUE


def test_meta_step_mixin_update_hyperparams_should_update_meta_step_hyperparams():
    p = SomeMetaStepMixin(SomeStep())
    p.set_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE,
        SOME_STEP_HP: SOME_STEP_HP_VALUE
    }))

    p.update_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE + 1
    }))

    assert p.hyperparams[META_STEP_HP] == META_STEP_HP_VALUE + 1
    assert p.get_step().get_hyperparams()['somestep_hyperparam'] == SOME_STEP_HP_VALUE


def test_meta_step_mixin_update_hyperparams_should_update_wrapped_step_hyperparams():
    p = SomeMetaStepMixin(SomeStep())
    p.set_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE,
        SOME_STEP_HP: SOME_STEP_HP_VALUE
    }))

    p.update_hyperparams(HyperparameterSamples({
        SOME_STEP_HP: SOME_STEP_HP_VALUE + 1
    }))

    assert p.hyperparams[META_STEP_HP] == META_STEP_HP_VALUE
    assert p.get_step().get_hyperparams()['somestep_hyperparam'] == SOME_STEP_HP_VALUE + 1


def test_meta_step_mixin_should_set_hyperparams_space():
    p = SomeMetaStepMixin(SomeStep())
    p.set_hyperparams_space(HyperparameterSpace({
        META_STEP_HP: RAND_INT_META_STEP,
        SOME_STEP_HP: RAND_INT_SOME_STEP
    }))

    assert p.hyperparams_space[META_STEP_HP] == RAND_INT_META_STEP
    assert p.get_step().hyperparams_space[SOME_STEP_HP_KEY] == RAND_INT_SOME_STEP


def test_meta_step_mixin_should_get_hyperparams_space():
    p = SomeMetaStepMixin(SomeStep())
    p.set_hyperparams_space(HyperparameterSpace({
        META_STEP_HP: RAND_INT_META_STEP,
        SOME_STEP_HP: RAND_INT_SOME_STEP
    }))

    hyperparams_space = p.get_hyperparams_space()

    assert hyperparams_space[META_STEP_HP] == RAND_INT_META_STEP
    assert hyperparams_space[SOME_STEP_HP] == RAND_INT_SOME_STEP


def test_get_set_params_base_step():
    s = SomeStep()

    s.set_params(learning_rate=0.1)
    hyperparams = s.get_params()

    assert hyperparams == {"learning_rate": 0.1}
