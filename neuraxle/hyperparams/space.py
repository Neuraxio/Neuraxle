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
from collections import OrderedDict, ItemsView
from copy import deepcopy
from dataclasses import dataclass
from typing import List

from scipy.stats import rv_continuous, rv_discrete
from scipy.stats._distn_infrastructure import rv_generic

from neuraxle.hyperparams.distributions import HyperparameterDistribution
from neuraxle.hyperparams.scipy_distributions import ScipyDiscreteDistributionWrapper, \
    ScipyContinuousDistributionWrapper


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
    DEFAULT_SEPARATOR = '__'

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

    def __getitem__(self, key):
        return self._rec_get(key)

    def _rec_get(self, key):
        """
        Split the keys and call getter recursively until we get to the desired element.
        None returns every non-recursive elements.
        """
        if key is None:
            return dict(filter(lambda x: not isinstance(x[1], RecursiveDict), self.items()))
        lkey, _, rkey = key.partition(self.separator)

        if rkey == "":
            return OrderedDict.__getitem__(self, lkey)

        rec_dict: RecursiveDict = OrderedDict.__getitem__(self, lkey)
        return rec_dict._rec_get(rkey)

    def __setitem__(self, key, value):
        lkey, _, rkey = key.partition(self.separator)
        if rkey == "":
            if isinstance(value, dict) and not isinstance(value, RecursiveDict):
                value = self.same_class_new_instance(value.items())
            OrderedDict.__setitem__(self, lkey, value)
        else:
            if lkey not in self:
                OrderedDict.__setitem__(self, lkey, self.same_class_new_instance())
            self[lkey][rkey] = value

    def __contains__(self, key) -> bool:
        try:
            _ = self[key]
            return True
        except KeyError:
            return False

    def get(self, key):
        try:
            return self[key]
        except KeyError:
            return self.same_class_new_instance()

    def iter_flat(self, pre_key="", values_only=False):
        """
        Returns a generator which yield (flatenned_key, value) pairs. value is never a RecursiveDict instance.
        """
        for k, v in self.items():
            if isinstance(v, RecursiveDict):
                yield from v.iter_flat(pre_key + k + self.separator, values_only=values_only)
            else:
                if values_only:
                    yield v
                else:
                    yield (pre_key + k, v)

    def to_flat_dict(self) -> dict:
        """
        Returns a dictionary with no recursively nested elements, i.e. {flattened_key -> value}.
        """
        return dict(self.iter_flat())

    def to_nested_dict(self) -> dict:
        """
        Returns a dictionary counterpart which is still nested but that contains no RecursiveDict.
        """
        out_dict = dict()
        for k, v in self.items():
            if isinstance(v, RecursiveDict):
                v = v.to_nested_dict()
            out_dict[k] = v
        return out_dict

    def with_separator(self, separator):
        """
        Create a new recursive dict that uses the given separator at each level.

        :param separator:
        :return:
        """
        return type(self)(
            separator=separator,
            **{key: value if not isinstance(value, RecursiveDict) \
                else value.with_separator(separator) for key, value in self.items()
               })


class HyperparameterSamples(RecursiveDict):
    """
    Wraps an hyperparameter nested dict or flat dict, and offer a few more functions.

    This can be set on a Pipeline with the method ``set_hyperparams``.

    HyperparameterSamples are often the result of calling ``.rvs()`` on an HyperparameterSpace.
    """

    def __init__(self, *args, separator=None, **kwds):
        super().__init__(*args, separator=separator, **kwds)

    def compress(self) -> 'CompressedHyperparameterSamples':
        """Compresses the HyperparameterSamples representation."""

        return CompressedHyperparameterSamples(self)


@dataclass
class CompressedHyperparameter:
    """CompressedHyperParameter"""

    step_name: str
    hyperparams: dict
    ancestor_steps: list


class CompressedHyperparameterSamples:
    """
    Short-hand representation of `HyperparameterSamples`
    """

    def __init__(self, hps: HyperparameterSamples):
        """Takes in `HyperparameterSamples` object and generates a compressed _seq."""
        if not hps or not isinstance(hps, HyperparameterSamples):
            raise ValueError("pass a valid `HyperparameterSamples` object")
        self.separator: str = hps.separator
        self._seq: List[dict] = self._convert(hps)

    def __str__(self) -> str:
        """Prints the `CompressedHyperparameterSamples._seq`."""
        return f"{self._seq}"

    def _group_hps_by_step(self, flat_steps_hps: ItemsView) -> 'OrderedDict[str, dict]':
        """
        Groups hyperparams
        :param flat_steps_hps: Flat list of steps and hyper parameter
        :return: OrderedDict containning grouped hyperparams as values and keys as pipeline stages & steps

        >>>hyper_params = [('AddFeatures__PCA__copy', True), ('AddFeatures__PCA__iterated_power', 'auto')]
        >>>self._group_hps_by_step(hyper_params = hyper_params)
        ...OrderedDict([('AddFeatures__PCA', dict([('copy', True), ('iterated_power', 'auto')]))])
        """
        grouped_hyper_params = OrderedDict()
        for steps_and_hps, hyper_param_val in flat_steps_hps:
            *steps, hyper_param_key = steps_and_hps.split(self.separator)
            grouped_hyper_params.setdefault(f"{self.separator}".join(steps), dict()).update({
                hyper_param_key: hyper_param_val})

        return grouped_hyper_params

    def _convert_to_compressed_format(self, group_by_step: 'OrderedDict[str, dict]') -> 'List[dict]':
        """Converts grouped hyper params to Compressed format."""
        compressed = []
        for steps in group_by_step:
            *prev_steps, current_step = steps.split(self.separator)
            hps: dict = group_by_step[steps]

            compressed.append(CompressedHyperparameter(step_name=current_step, hyperparams=hps,
                                                       ancestor_steps=prev_steps).__dict__)
        return compressed

    def _convert(self, hps: HyperparameterSamples) -> 'List[dict]':
        grouped_result: OrderedDict[str, dict] = self._group_hps_by_step(flat_steps_hps=hps.to_flat_dict().items())
        return self._convert_to_compressed_format(group_by_step=grouped_result)


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
        """
        if hasattr(arg, 'dist') and isinstance(arg.dist, rv_generic):
            if isinstance(arg.dist, rv_discrete):
                return ScipyDiscreteDistributionWrapper(arg), True
            if isinstance(arg.dist, rv_continuous):
                return ScipyContinuousDistributionWrapper(arg), True
        else:
            return arg, False

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

    # TODO : The following functions aren't used, or tested, anywhere. They should work though.

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
