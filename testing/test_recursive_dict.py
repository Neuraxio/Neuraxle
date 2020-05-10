from neuraxle.hyperparams.space import RecursiveDict, HyperparameterSamples


def test_recursive_dict_to_flat():
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

    r = r.to_flat()

    expected_dict_values = {
        'hp': 1,
        'stepa__hp': 2,
        'stepa__stepb__hp': 3
    }
    assert r == RecursiveDict(separator='__', **expected_dict_values)


def test_recursive_dict_to_nested_dict():
    dict_values = {
        'hp': 1,
        'stepa__hp': 2,
        'stepa__stepb__hp': 3
    }
    r = HyperparameterSamples(**dict_values)

    r = r.flat_to_nested_dict()

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

    r = r.flat_to_nested_dict()

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
    r = RecursiveDict('__', RecursiveDict(**dict_values))

    assert r == RecursiveDict(**dict_values)


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
