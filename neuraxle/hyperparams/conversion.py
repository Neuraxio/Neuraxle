"""
Hyperparameter Dictionary Conversions
====================================
Ways to convert from a nested dictionary of hyperparameters to a flat dictionary, and vice versa.

Here is a nested dictionary:

```python
{
    "b": {
        "a": {
            "learning_rate": 7
        },
        "learning_rate": 9
    }
}
```

Here is an equivalent flat dictionary for the previous nested one:

```python
{
    "b.a.learning_rate": 7,
    "b.learning_rate": 9
}
```

Notice that if you have a `SKLearnWrapper` on a sklearn Pipeline object, the hyperparameters past that point will use
double underscores "__" as a separator rather than a dot in flat dictionaries, and in nested dictionaries the
sklearn params will appear as a flat past the sklearn wrapper, which is fine.

"""

from collections import OrderedDict

from neuraxle.typing import DictHyperparams, FlatHyperparams

PARAMS_SPLIT_SEQ = "__"

def dict_to_flat(hyperparams: DictHyperparams, dict_ctor=OrderedDict) -> FlatHyperparams:
    """
    Convert a nested hyperparameter dictionary to a flat one.

    :param hyperparams: a nested hyperparameter dictionary.
    :param dict_ctor: `OrderedDict` by default. Will use this as a class to create the new returned dict.
    :return: a flat hyperparameter dictionary.
    """
    ret = dict_ctor()
    for k, v in hyperparams.items():
        if isinstance(v, dict) or isinstance(v, OrderedDict) or isinstance(v, dict_ctor):
            _ret = dict_to_flat(v)
            for key, val in _ret.items():
                ret[k + PARAMS_SPLIT_SEQ + key] = val
        else:
            ret[k] = v
    return ret


def flat_to_dict(hyperparams: FlatHyperparams, dict_ctor=OrderedDict) -> DictHyperparams:
    """
    Convert a flat hyperparameter dictionary to a nested one.

    :param hyperparams: a flat hyperparameter dictionary.
    :param dict_ctor: `OrderedDict` by default. Will use this as a class to create the new returned dict.
    :return: a nested hyperparameter dictionary.
    """
    pre_ret = dict_ctor()
    ret = dict_ctor()
    for k, v in hyperparams.items():
        k, _, key = k.partition(PARAMS_SPLIT_SEQ)
        if len(key) > 0:
            if k not in pre_ret.keys():
                pre_ret[k] = dict_ctor()
            pre_ret[k][key] = v
        else:
            ret[k] = v
    for k, v in pre_ret.items():
        ret[k] = flat_to_dict(v)
    return ret
