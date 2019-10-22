from neuraxle.hyperparams.distributions import RandInt
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace
from neuraxle.pipeline import Pipeline
from testing.mocks.step_mocks import SomeStep


def main():
    p = Pipeline([
        SomeStep()
    ])

    p = p.set_hyperparams(HyperparameterSamples({
        'pipeline_hp': 1,
        'SomeStep__hp': 2
    }))

    assert p['SomeStep'].hyperparams['hp'] == 2

    hyperparams = p.get_hyperparams()

    assert hyperparams['pipeline_hp'] == 1
    assert hyperparams['SomeStep__hp'] == 2

    p = p.set_hyperparams_space(HyperparameterSpace({
        'pipeline_hp': RandInt(0, 10),
        'SomeStep__hp': RandInt(-10, 0)
    }))

    assert p.hyperparams_space['pipeline_hp'].min_included == 0
    assert p.hyperparams_space['pipeline_hp'].max_included == 10

    assert p['SomeStep'].hyperparams_space['hp'].min_included == -10
    assert p['SomeStep'].hyperparams_space['hp'].max_included == 0

    hyperparams_space = p.get_hyperparams_space()

    assert hyperparams_space['pipeline_hp'].min_included ==  0
    assert hyperparams_space['pipeline_hp'].max_included == 10

    assert hyperparams_space['SomeStep__hp'].min_included == -10
    assert hyperparams_space['SomeStep__hp'].max_included == 0
