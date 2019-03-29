from collections import OrderedDict

from neuraxle.typing import DictHyperparams, FlatHyperparams


def dict_to_flat(hyperparams: DictHyperparams, dict_ctor=OrderedDict) -> FlatHyperparams:
    ret = dict_ctor()
    for k, v in hyperparams.items():
        if isinstance(v, dict) or isinstance(v, OrderedDict) or isinstance(v, dict_ctor):
            _ret = dict_to_flat(v)
            for key, val in _ret.items():
                ret[k + "." + key] = val
        else:
            ret[k] = v
    return ret


def flat_to_dict(hyperparams: FlatHyperparams, dict_ctor=OrderedDict) -> DictHyperparams:
    pre_ret = dict_ctor()
    ret = dict_ctor()
    for k, v in hyperparams.items():
        k, _, key = k.partition(".")
        if len(key) > 0:
            if k not in pre_ret.keys():
                pre_ret[k] = dict_ctor()
            pre_ret[k][key] = v
        else:
            ret[k] = v
    for k, v in pre_ret.items():
        ret[k] = flat_to_dict(v)
    return ret
