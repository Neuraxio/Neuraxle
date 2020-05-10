from neuraxle.base import _RecursiveArguments, Identity
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline


def test_recursive_arguments_should_get_root_level():
    ra = _RecursiveArguments(hyperparams=HyperparameterSamples({
        'hp0': 0,
        'hp1': 1,
        'pipeline__stepa__hp2': 2,
        'pipeline__stepb__hp3': 3
    }))

    root_ra = ra[None]

    root_ra.kargs == []
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

    ra.kargs == []
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

    ra.kargs == []
    ra.kwargs == {'hyperparams': HyperparameterSamples({
        'hp0': 0,
        'hp1': 1,
    })}


def test_has_children_mixin_apply_should_apply_method_to_direct_childrends():
    p = Pipeline([
        ('a', Identity()),
        ('b', Identity()),
        Pipeline([
            ('c', Identity()),
            ('d', Identity())
        ]),
    ])

    p.apply('set_hyperparams', ra=None, hyperparams=HyperparameterSamples({
        'a__hp': 0,
        'b__hp': 1,
        'Pipeline__hp': 2
    }))

    assert p['a'].hyperparams.to_flat_as_dict_primitive()['hp'] == 0
    assert p['b'].hyperparams.to_flat_as_dict_primitive()['hp'] == 1
    assert p['Pipeline'].hyperparams.to_flat_as_dict_primitive()['hp'] == 2


def test_has_children_mixin_apply_should_apply_method_to_recursive_childrends():
    p = Pipeline([
        ('a', Identity()),
        ('b', Identity()),
        Pipeline([
            ('c', Identity()),
            ('d', Identity())
        ]),
    ])

    p.apply('set_hyperparams', ra=None, hyperparams=HyperparameterSamples({
        'Pipeline__c__hp': 3,
        'Pipeline__d__hp': 4
    }))

    assert p['Pipeline']['c'].hyperparams.to_flat_as_dict_primitive()['hp'] == 3
    assert p['Pipeline']['d'].hyperparams.to_flat_as_dict_primitive()['hp'] == 4


def test_has_children_mixin_apply_should_return_recursive_dict_to_direct_childrends():
    p = Pipeline([
        ('a', Identity().set_hyperparams(HyperparameterSamples({'hp': 0}))),
        ('b', Identity().set_hyperparams(HyperparameterSamples({'hp': 1})))
    ])

    results = p.apply('get_hyperparams', ra=None)

    assert results.to_flat_as_dict_primitive()['a__hp'] == 0
    assert results.to_flat_as_dict_primitive()['b__hp'] == 1


def test_has_children_mixin_apply_should_return_recursive_dict_to_recursive_childrends():
    p = Pipeline([
        Pipeline([
            ('c', Identity().set_hyperparams(HyperparameterSamples({'hp': 3}))),
            ('d', Identity().set_hyperparams(HyperparameterSamples({'hp': 4})))
        ]).set_hyperparams(HyperparameterSamples({'hp': 2})),
    ])

    results = p.apply('get_hyperparams', ra=None)
    results = results.to_flat_as_dict_primitive()

    assert results['Pipeline__hp'] == 2
    assert results['Pipeline__c__hp'] == 3
    assert results['Pipeline__d__hp'] == 4
