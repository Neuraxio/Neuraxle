"""
Hyperparameter Dictionary Conversions
=====================================
Ways to convert from a nested dictionary of hyperparameters to a flat dictionary, and vice versa.

Here is a nested dictionary:

.. code-block:: python

    {
        "b": {
            "a": {
                "learning_rate": 7
            },
            "learning_rate": 9
        }
    }

Here is an equivalent flat dictionary for the previous nested one:

.. code-block:: python

    {
        "b.a.learning_rate": 7,
        "b.learning_rate": 9
    }

Notice that if you have a ``SKLearnWrapper`` on a sklearn Pipeline object, the hyperparameters past that point will use
double underscores ``__`` as a separator rather than a dot in flat dictionaries, and in nested dictionaries the
sklearn params will appear as a flat past the sklearn wrapper, which is fine.

By default, hyperparameters are stored inside a HyperparameterSpace or inside a HyperparameterSamples object, which
offers methods to do the conversions above, and also using ordered dicts (OrderedDict) to store parameters in-order.

A HyperparameterSpace can be sampled by calling the ``.rvs()`` method on it, which will recursively call ``.rvs()`` on all
the HyperparameterSpace and HyperparameterDistribution that it contains. It will return a HyperparameterSamples object.
A HyperparameterSpace can also be narrowed towards an better, finer subspace, which is itself a HyperparameterSpace.
This can be done by calling the ``.narrow_space_from_best_guess`` method of the HyperparameterSpace which will also
recursively apply the changes to its contained HyperparameterSpace and to all its contained HyperparameterDistribution.

The HyperparameterSamples contains sampled hyperparameter, that is, a valued point in the possible space. This is
ready to be sent to an instance of the pipeline to try and score it, for example.

..
    Copyright 2019, Neuraxio Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

..
    Thanks to Umaneo Technologies Inc. for their contributions to this Machine Learning
    project, visit https://www.umaneo.com/ for more information on Umaneo Technologies Inc.

"""

import typing
from collections import Counter, OrderedDict, defaultdict
from copy import copy
from typing import Any, Dict, Iterable, List, Set, Union

from neuraxle.hyperparams.distributions import (FixedHyperparameter,
                                                HPSampledValue,
                                                HyperparameterDistribution)
from neuraxle.hyperparams.scipy_distributions import (
    ScipyContinuousDistributionWrapper, ScipyDiscreteDistributionWrapper)
from scipy.stats import rv_continuous, rv_discrete
from scipy.stats._distn_infrastructure import rv_generic

FlatDict = typing.OrderedDict[str, HPSampledValue]
RecursiveDictValue = Union[Any, 'RecursiveDict']

RList = List  # reversed list.


# class RecursiveDict(OrderedDict[str, RecursiveDictValue]):
class RecursiveDict(OrderedDict):
    """
    A data structure that provides an interface to access nested dictionaries with "flattened keys", and a few more functions.

    e.g.
        dct = RecursiveDict({'a':{'b':2}})
        assert dct["a__b"] == 2
        dct["a__b__c"] = 3
        assert dct['a']['b']['c'] == dct["a__b__c"] == dct.to_flat_dict()["a__b__c"]

    This class serves as a base for HyperparameterSamples and HyperparameterSpace
    """
    DEFAULT_SEPARATOR: str = '__'

    def __init__(self, *args, separator=None, **kwds):

        if separator is None:
            if self._is_only_arg_a_recursive_dict(args, kwds):
                separator = args[0].separator
            else:
                separator = self.DEFAULT_SEPARATOR
        self.separator = separator

        OrderedDict.__init__(self)
        for arg in args:
            self.update(arg)
        self.update(kwds)
        self._patch_args()

    def _is_only_arg_a_recursive_dict(self, args, kwds):
        return len(args) == 1 and isinstance(args[0], RecursiveDict) and len(kwds) == 0

    def _patch_args(self):
        to_patch_key_values = []
        for k, v in self.items():
            if isinstance(v, RecursiveDict):
                v._patch_args()
            else:
                patched_arg, did_patch = self._patch_arg(v)
                if did_patch:
                    to_patch_key_values.append((k, patched_arg))

        for k, v in to_patch_key_values:
            self[k] = v

    def same_class_new_instance(self, *args, **kwds):
        return type(self)(*args, separator=self.separator, **kwds)

    def _patch_arg(self, arg):
        """
        Patches argument if needed.
        :param arg: arg to patch if needed.
        :return: (patched_arg, did_patch)
        """
        return arg, False

    def get(self, key) -> RecursiveDictValue:
        try:
            return self[key]
        except KeyError:
            return self.same_class_new_instance()

    def __getitem__(self, key) -> RecursiveDictValue:
        return self._rec_get(key)

    def _rec_get(self, key) -> RecursiveDictValue:
        """
        Split the keys and call getter recursively until we get to the desired element.
        None returns every non-recursive elements.
        """
        if key is None:
            return self.get_root_leaf_data()

        lkey, _, rkey = key.partition(self.separator)
        rec_dict: RecursiveDict = OrderedDict.__getitem__(self, lkey)
        if rkey == "":
            return rec_dict
        else:
            # Splitted on sep and recursively call getter
            return rec_dict._rec_get(rkey)

    def get_root_leaf_data(self) -> FlatDict:
        """
        Returns a dictionary of all the non-recursive elements.
        That is, all the elements that are not RecursiveDict in the
        current root OrderedDict.
        """
        return dict(filter(lambda x: not isinstance(x[1], RecursiveDict), self.items()))

    def __setitem__(self, key: str, value: RecursiveDictValue):
        lkey, _, rkey = key.partition(self.separator)
        if rkey == "":
            if isinstance(value, dict) and not isinstance(value, RecursiveDict):
                value = self.same_class_new_instance(value.items())
            OrderedDict.__setitem__(self, lkey, value)
        else:
            if lkey not in self:
                OrderedDict.__setitem__(self, lkey, self.same_class_new_instance())
            self[lkey][rkey] = value

    def __contains__(self, key: str) -> bool:
        try:
            _ = self[key]
            return True
        except KeyError:
            return False

    def iter_flat(self, pre_key="", values_only=False) -> Iterable:
        """
        Returns a generator which yield (flatenned_key, value) pairs. value is never a RecursiveDict instance.
        Keys are sorted, then values are sorted as well.
        """
        for k, v in sorted(self.items()):
            if isinstance(v, RecursiveDict):
                yield from v.iter_flat(pre_key + k + self.separator, values_only=values_only)
            else:
                if values_only:
                    yield v
                else:
                    yield (pre_key + k, v)

    def to_flat_dict(self, use_wildcards=False) -> FlatDict:
        """
        Returns a FlatDict, that is a totally flatened OrderedDict[str, HPSampledValue],
        with no recursively nested elements, i.e.: {fully__flattened__params: value}.

        .. info::
            The returned FlatDict is sorted in a new alphabetical order.

        :param use_wildcards: If True, wildcards are used in the keys, as if calling self.to_wildcards() directly. See :func:`to_wildcards` for more info.
        """
        if use_wildcards is True:
            return self.to_wildcards()
        return OrderedDict(self.iter_flat())

    def to_nested_dict(self) -> Dict[str, RecursiveDictValue]:
        """
        Returns a dictionary counterpart which is still nested but that contains no RecursiveDict.
        """
        out_dict = dict()
        for k, v in self.items():
            if isinstance(v, RecursiveDict):
                v = v.to_nested_dict()
            out_dict[k] = v
        return out_dict

    def to_wildcards(self) -> FlatDict:
        """
        Returns a FlatDict with wildcards. Shorten the keys to the shortest possible while
        mainting the same order and value, as well as the ability to differentiate each key.
        """
        flat: FlatDict = self.to_flat_dict()
        rkeys: List[RList[str]] = list([
            list(reversed(i.split(self.separator))) for i in flat.keys()
        ])

        rkeys = self._wildcards_reduce(rkeys)

        # reverse the keys back to the original order:
        rkeys = [
            self.separator.join(list(reversed(i))).replace(
                f"{self.separator}*", "*").replace(
                f"*{self.separator}", "*")
            for i in rkeys
        ]
        flat = OrderedDict(zip(rkeys, flat.values()))
        return flat

    def _wildcards_reduce(self, rkeys: List[RList[str]], depth=0):
        rkeys = copy(rkeys)

        glob_ctr = Counter()
        key_attribution: Dict[str, int] = dict()

        def permutations(rkey: RList[str]) -> Set[str]:
            out: Set[str] = set()
            out.add(self.separator.join(rkey))

            for i in range(1, len(rkey)):
                for j in range(i, len(rkey)):
                    for k in range(j, len(rkey)):

                        rk = rkey[:i] + (["*"] if i != j else []) + rkey[j:k] + (["*"] if k != len(rkey) else [])
                        rks = self.separator.join(rk)
                        rks = rks.replace(
                            f"*{self.separator}*", "*").replace(
                            f"*{self.separator}*", "*")

                        out.add(rks)
            return out

        for i, rkey in enumerate(rkeys):
            rk_perms: Set[str] = permutations(rkey)
            glob_ctr.update(rk_perms)
            for rk in rk_perms:
                key_attribution[rk] = i

        # remove the keys that have a count of more than 1 in the counter:
        glob_ctr = {k: v for k, v in glob_ctr.items() if v == 1}
        attribution_keys: Dict[int, List[str]] = defaultdict(list)
        key_attribution = {attribution_keys[idx].append(rks)
                           for rks, idx in key_attribution.items() if rks in glob_ctr.keys()}

        # keep the shortest key:
        attributed_key: Dict[str, str] = {
            k: min(v, key=lambda x: len(
                x.replace(
                    f"{self.separator}*", "*").replace(
                    f"*{self.separator}", "*")
            ))
            for k, v in attribution_keys.items()
        }
        attributed_key = {k: v.split(self.separator) for k, v in attributed_key.items()}
        keys = [v for k, v in sorted(attributed_key.items())]
        return keys

    def with_separator(self, separator: str):
        """
        Create a new recursive dict (from copy) that uses the given separator at each level.
        """
        return type(self)(
            separator=separator, **{
                key: value
                if not isinstance(value, RecursiveDict)
                else value.with_separator(separator) for key, value in self.items()
            })

    def is_empty(self) -> bool:
        return len(self) == 0 or len(self.to_flat_dict()) == 0


class HyperparameterSamples(RecursiveDict):
    """
    Wraps an hyperparameter nested dict or flat dict, and offer a few more functions.

    This can be set on a Pipeline with the method ``set_hyperparams``.

    HyperparameterSamples are often the result of calling ``.rvs()`` on an HyperparameterSpace.
    """

    def __init__(self, *args, separator=None, **kwds):
        super().__init__(*args, separator=separator, **kwds)


class HyperparameterSpace(RecursiveDict):
    """
    Wraps an hyperparameter nested dict or flat dict, and offer a few more functions to process
    all contained HyperparameterDistribution.

    This can be set on a Pipeline with the method ``set_hyperparams_space``.

    Calling ``.rvs()`` on an ``HyperparameterSpace`` results in ``HyperparameterSamples``.
    """

    def __init__(self, *args, separator=None, **kwds):
        super().__init__(*args, separator=separator, **kwds)

    def _patch_arg(self, arg):
        """
        Override of the RecursiveDict's default _patch_arg(arg) method.

        :param arg: arg to patch if needed.
        :return: (patched_arg, did_patch)
        """
        did_patch = False
        if hasattr(arg, 'dist') and isinstance(arg.dist, rv_generic):
            if isinstance(arg.dist, rv_continuous):
                arg, did_patch = ScipyContinuousDistributionWrapper(arg), True
            elif isinstance(arg.dist, rv_discrete):
                arg, did_patch = ScipyDiscreteDistributionWrapper(arg), True
        assert isinstance(arg, HyperparameterDistribution), (
            f"Hyperparameter space's distributions must be a dict of `HyperparameterDistributions`. "
            f"got `{arg}` of type `{type(arg)}` instead. You might consider using a "
            f"{FixedHyperparameter.__name__} class like {FixedHyperparameter}({arg}) to have a fixed value."
        )
        return arg, did_patch

    def rvs(self) -> 'HyperparameterSamples':
        """
        Sample the space of random variables.

        :return: a random HyperparameterSamples, sampled from a point of the present HyperparameterSpace.
        """
        new_items = []
        for k, v in self.iter_flat():
            if isinstance(v, HyperparameterDistribution):
                v = v.rvs()
            new_items.append((k, v))
        return HyperparameterSamples(new_items)

    # TODO : The following functions aren't really used, or tested. They should work though.

    def nullify(self):
        new_items = []
        for k, v in self.iter_flat():
            if isinstance(v, HyperparameterDistribution) or isinstance(v, HyperparameterSpace):
                v = v.nullify()
            new_items.append((k, v))
        return HyperparameterSamples(new_items)

    def narrow_space_from_best_guess(
            self,
            best_guesses: 'HyperparameterSpace',
            kept_space_ratio: float = 0.5
    ) -> 'HyperparameterSpace':
        """
        Takes samples estimated to be the best ones of the space as of yet, and restrict the whole space towards that.

        :param best_guesses: sampled HyperparameterSpace (the result of rvs on each parameter, but still stored as a HyperparameterSpace).
        :param kept_space_ratio: what proportion of the space is kept. Should be between 0.0 and 1.0. Default is 0.5.
        :return: a new HyperparameterSpace containing the narrowed HyperparameterDistribution objects.
        """
        new_items = []
        for k, v in self.iter_flat():
            if isinstance(v, HyperparameterDistribution) or isinstance(v, HyperparameterSpace):
                best_guess_v = best_guesses[k]
                v = v.narrow_space_from_best_guess(best_guess_v, kept_space_ratio)
            new_items.append((k, v))
        return HyperparameterSpace(new_items)

    def unnarrow(self) -> 'HyperparameterSpace':
        """
        Return the original space before narrowing of the distribution. If the distribution was never narrowed,
        the values in the dict will be copies.

        :return: the original HyperparameterSpace before narrowing.
        """
        new_items = []
        for k, v in self.iter_flat():
            if isinstance(v, HyperparameterDistribution) or isinstance(v, HyperparameterSpace):
                v = v.unnarrow()
            new_items.append((k, v))
        return HyperparameterSpace(new_items)
