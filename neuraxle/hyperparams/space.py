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

from collections import OrderedDict

from neuraxle.hyperparams.distributions import HyperparameterDistribution


class RecursiveDict(OrderedDict):
    """
    Wraps an hyperparameter nested dict or flat dict, and offer a few more functions.

    This can be set on a Pipeline with the method ``set_hyperparams``.

    HyperparameterSamples are often the result of calling ``.rvs()`` on an HyperparameterSpace.
    """
    DEFAULT_SEPARATOR = '__'

    def __init__(self, *args, separator=None, **kwds):
        if len(args) == 1 and isinstance(args[0], RecursiveDict) and len(kwds) == 0:
            super().__init__(args[0].items())
            separator = args[0].separator
        else:
            super().__init__(*args, **kwds)

        if separator is None:
            separator = self.DEFAULT_SEPARATOR

        self.separator = separator

    def __getitem__(self, item: str = None):
        item_values = type(self)()

        for name, values in self.items():
            if item is None and not self.separator in name:
                item_values[name] = values

            if self.separator in name:
                name_split = name.split(self.separator)
                if str(name_split[0]) == item:
                    item_values[self.separator.join(name_split[1:])] = values

            if item == name:
                item_values = values

        return item_values

    def to_flat(self) -> 'RecursiveDict':
        """
        Will create an equivalent flat HyperparameterSamples.

        :return: an HyperparameterSamples like self, flattened.
        """
        return self.nested_dict_to_flat(dict_ctor=type(self))

    def to_nested_dict(self) -> 'RecursiveDict':
        """
        Will create an equivalent nested dict HyperparameterSamples.

        :return: an HyperparameterSamples like self, as a nested dict.
        """
        return self.flat_to_nested_dict(dict_ctor=type(self))

    def to_flat_as_dict_primitive(self) -> dict:
        """
        Will create an equivalent flat HyperparameterSpace, as a dict.

        :return: an HyperparameterSpace like self, flattened.
        """
        return self.nested_dict_to_flat(dict_ctor=dict)

    def to_nested_dict_as_dict_primitive(self) -> dict:
        """
        Will create an equivalent nested dict HyperparameterSpace, as a dict.

        :return: a nested primitive dict type of self.
        """
        return self.flat_to_nested_dict(dict_ctor=dict)

    def to_flat_as_ordered_dict_primitive(self) -> OrderedDict:
        """
        Will create an equivalent flat HyperparameterSpace, as a dict.

        :return: an HyperparameterSpace like self, flattened.
        """
        return self.nested_dict_to_flat(dict_ctor=OrderedDict)

    def to_nested_dict_as_ordered_dict_primitive(self) -> OrderedDict:
        """
        Will create an equivalent nested dict HyperparameterSpace, as a dict.

        :return: a nested primitive dict type of self.
        """
        return self.flat_to_nested_dict(dict_ctor=OrderedDict)

    def nested_dict_to_flat(self, dict_ctor):
        """
        Convert a nested hyperparameter dictionary to a flat one.

        :param dict_ctor: ``OrderedDict`` by default. Will use this as a class to create the new returned dict.
        :return: a flat hyperparameter dictionary.
        """
        if isinstance(dict_ctor, RecursiveDict):
            ret = dict_ctor(self.separator)
        else:
            ret = dict_ctor()

        for k, v in self.items():
            if isinstance(v, dict) or isinstance(v, OrderedDict) or isinstance(v, dict_ctor):
                nested_dict = v
                if not isinstance(v, RecursiveDict):
                    nested_dict = RecursiveDict(v, separator=self.separator)
                _ret = nested_dict.nested_dict_to_flat(dict_ctor=type(nested_dict))
                for key, val in _ret.items():
                    ret[k + nested_dict.separator + key] = val
            else:
                ret[k] = v
        return ret

    def flat_to_nested_dict(self, dict_ctor=dict):
        """
        Convert a flat hyperparameter dictionary to a nested one.

        :param dict_ctor: ``OrderedDict`` by default. Will use this as a class to create the new returned dict.
        :return: a nested hyperparameter dictionary.
        """
        if isinstance(dict_ctor, RecursiveDict):
            pre_ret = dict_ctor(self.separator)
            ret = dict_ctor(self.separator)
        else:
            pre_ret = dict_ctor()
            ret = dict_ctor()

        for k, v in self.items():
            k, _, key = k.partition(self.separator)
            if len(key) > 0:
                if k not in pre_ret.keys():
                    pre_ret[k] = dict_ctor()
                pre_ret[k][key] = v
            else:
                ret[k] = v
        for k, v in pre_ret.items():
            flat_dict = v
            if not isinstance(v, RecursiveDict):
                flat_dict = RecursiveDict(v, separator=self.separator)
            ret[k] = flat_dict.flat_to_nested_dict(dict_ctor=dict_ctor)
        return ret

    def with_separator(self, separator):
        """
        Create a new recursive dict that uses the given separator at each level.

        :param separator:
        :return:
        """
        return type(self)(
            separator=separator,
            **{
                key.replace(self.separator, separator): value if not isinstance(value, RecursiveDict) \
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


class HyperparameterSpace(RecursiveDict):
    """
    Wraps an hyperparameter nested dict or flat dict, and offer a few more functions to process
    all contained HyperparameterDistribution.

    This can be set on a Pipeline with the method ``set_hyperparams_space``.

    Calling ``.rvs()`` on an ``HyperparameterSpace`` results in ``HyperparameterSamples``.
    """

    def __init__(self, *args, separator=None, **kwds):
        super().__init__(*args, separator=separator, **kwds)

    def rvs(self) -> 'HyperparameterSamples':
        """
        Sample the space of random variables.

        :return: a random HyperparameterSamples, sampled from a point of the present HyperparameterSpace.
        """
        new_items = []
        for k, v in self.items():
            if isinstance(v, HyperparameterDistribution) or isinstance(v, HyperparameterSpace):
                v = v.rvs()
            new_items.append((k, v))
        return HyperparameterSamples(new_items)

    def nullify(self):
        new_items = []
        for k, v in self.items():
            if isinstance(v, HyperparameterDistribution) or isinstance(v, HyperparameterSpace):
                v = v.nullify()
            new_items.append((k, v))
        return HyperparameterSamples(new_items)

    def narrow_space_from_best_guess(
            self, best_guesses: 'HyperparameterSpace', kept_space_ratio: float = 0.5
    ) -> 'HyperparameterSpace':
        """
        Takes samples estimated to be the best ones of the space as of yet, and restrict the whole space towards that.

        :param best_guess: sampled HyperparameterSpace (the result of rvs on each parameter, but still stored as a HyperparameterSpace).
        :param kept_space_ratio: what proportion of the space is kept. Should be between 0.0 and 1.0. Default is 0.5.
        :return: a new HyperparameterSpace containing the narrowed HyperparameterDistribution objects.
        """
        new_items = []
        for k, v in self.items():
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
        for k, v in self.items():
            if isinstance(v, HyperparameterDistribution) or isinstance(v, HyperparameterSpace):
                v = v.unnarrow()
            new_items.append((k, v))
        return HyperparameterSpace(new_items)
