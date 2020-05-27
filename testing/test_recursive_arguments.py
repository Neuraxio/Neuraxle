from neuraxle.base import _RecursiveArguments
from neuraxle.hyperparams.space import HyperparameterSamples


def test_recursive_arguments_should_get_root_level():
    ra = _RecursiveArguments(hyperparams=HyperparameterSamples({
        'hp0': 0,
        'hp1': 1,
        'pipeline__stepa__hp2': 2,
        'pipeline__stepb__hp3': 3
    }))

    root_ra = ra[None]

    root_ra.args == []
    root_ra.kwargs == {'hyperparams': HyperparameterSamples({
        'hp0': 0,
        'hp1': 1
    })}


def test_recursive_arguments_should_get_recursive_levels():
    ra = _RecursiveArguments(hyperparams=HyperparameterSamples({
        'hp0': 0,
        'hp1': 1,
        'stepa__hp2': 2,
        'stepb__hp3': 3,
        'stepb__stepd__hp4': 4
    }))

    ra = ra['stepb']

    ra.args == []
    ra.kwargs == {'hyperparams': HyperparameterSamples({
        'stepb__hp3': 2,
        'stepb__stepd__hp4': 4
    })}


def test_recursive_arguments_should_have_copy_constructor():
    ra = _RecursiveArguments(
        ra=_RecursiveArguments(hyperparams=HyperparameterSamples({
            'hp0': 0,
            'hp1': 1
        }))
    )

    ra.args == []
    ra.kwargs == {'hyperparams': HyperparameterSamples({
        'hp0': 0,
        'hp1': 1,
    })}


