import pytest

from neuraxle.base import Identity
from neuraxle.hyperparams.space import RecursiveDict, HyperparameterSamples

POINT_SEPARATOR = '.'


@pytest.mark.parametrize("separator", ["__", ".", "___"])
def test_recursive_dict_to_flat(separator):
    dict_values = {
        'hp': 1,
        'stepa': {
            'hp': 2,
            'stepb': {
                'hp': 3
            }
        }
    }
    r = RecursiveDict(separator=separator, **dict_values)

    r = r.to_flat_as_dict_primitive()

    expected_dict_values = {
        'hp': 1,
        'stepa{}hp'.format(separator): 2,
        'stepa{0}stepb{0}hp'.format(separator): 3
    }
    assert r == expected_dict_values


def test_recursive_dict_to_flat_different_separator():
    dict_values = {
        'hp': 1,
        'stepa': {
            'hp': 2,
            'stepb': {
                'hp': 3
            }
        }
    }
    r = RecursiveDict(separator='__', **dict_values)
    r['stepa'] = RecursiveDict(r['stepa'], separator='.')
    r['stepa']['stepb'] = RecursiveDict(r['stepa']['stepb'], separator='$$$')

    r = r.to_flat()

    expected_dict_values = {
        'hp': 1,
        'stepa.hp': 2,
        'stepa.stepb$$$hp': 3
    }
    assert r.to_flat_as_dict_primitive() == expected_dict_values


def test_recursive_dict_to_nested_dict():
    dict_values = {
        'hp': 1,
        'stepa__hp': 2,
        'stepa__stepb__hp': 3
    }
    r = HyperparameterSamples(**dict_values)

    r = r.to_nested_dict()

    expected_dict_values = {
        'hp': 1,
        'stepa': {
            'hp': 2,
            'stepb': {
                'hp': 3
            }
        }
    }
    assert r == HyperparameterSamples(**expected_dict_values)


def test_recursive_dict_get_item():
    dict_values = {
        'hp': 1,
        'stepa__hp': 2,
        'stepa__stepb__hp': 3
    }
    r = HyperparameterSamples(**dict_values)

    assert r[None].to_flat_as_dict_primitive() == {'hp': 1}
    assert r['stepa'].to_flat_as_dict_primitive() == {'hp': 2, 'stepb__hp': 3}


def test_hyperparams_to_nested_dict():
    dict_values = {
        'hp': 1,
        'stepa__hp': 2,
        'stepa__stepb__hp': 3
    }
    r = HyperparameterSamples(**dict_values)

    r = r.to_nested_dict()

    expected_dict_values = {
        'hp': 1,
        'stepa': {
            'hp': 2,
            'stepb': {
                'hp': 3
            }
        }
    }
    assert r.to_nested_dict_as_dict_primitive() == expected_dict_values


def test_recursive_dict_copy_constructor():
    dict_values = {
        'hp': 1,
        'stepa__hp': 2,
        'stepa__stepb__hp': 3
    }
    r = RecursiveDict(RecursiveDict(**dict_values), separator='__')

    assert r == RecursiveDict(**dict_values)


def test_recursive_dict_copy_constructor_should_set_separator():
    dict_values = {
        'hp': 1,
        'stepa__hp': 2,
        'stepa__stepb__hp': 3
    }
    r = RecursiveDict(RecursiveDict(**dict_values, separator=POINT_SEPARATOR))

    assert r.separator == POINT_SEPARATOR


def test_recursive_dict_should_raise_when_item_missing():
    with pytest.raises(ValueError):
        r = RecursiveDict()
        missing = r['missing']


def test_hyperparams_copy_constructor():
    dict_values = {
        'hp': 1,
        'stepa__hp': 2,
        'stepa__stepb__hp': 3
    }
    r = HyperparameterSamples(HyperparameterSamples(**dict_values))

    assert r == HyperparameterSamples(**dict_values)


def test_hyperparams_to_flat():
    dict_values = {
        'hp': 1,
        'stepa': {
            'hp': 2,
            'stepb': {
                'hp': 3
            }
        }
    }
    r = HyperparameterSamples(**dict_values)

    r = r.to_flat()

    expected_dict_values = {
        'hp': 1,
        'stepa__hp': 2,
        'stepa__stepb__hp': 3
    }
    assert r == HyperparameterSamples(**expected_dict_values)
