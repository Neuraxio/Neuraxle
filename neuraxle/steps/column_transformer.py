"""
Neuraxle's Column Transformer Steps
====================================

Pipeline steps to apply N-Dimensional column transformations to different columns.

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
from operator import itemgetter
from typing import List, Tuple, Union, Iterable

import numpy as np

from neuraxle.base import MetaStep, BaseTransformer
from neuraxle.pipeline import Pipeline
from neuraxle.steps.loop import ForEach
from neuraxle.union import FeatureUnion

ColumnSelectionType = Union[int, Iterable[int], str, Iterable[str], slice]
ColumnChooserTupleList = List[Tuple[ColumnSelectionType, BaseTransformer]]


class ColumnSelector2D(BaseTransformer):
    """
    A ColumnSelector2D selects column in a sequence.

    It can be used to select:

    - a single column,
    - a range of columns,
    - a slice of columns,
    - a list of columns.

    The columns are expected to be integers.
    A special case is a string, which will be used as
    a pandas DataFrame column name.
    """

    def __init__(self, columns_selection: ColumnSelectionType):
        super().__init__()
        if isinstance(columns_selection, range):
            columns_selection = slice(
                columns_selection.start,
                columns_selection.stop,
                columns_selection.step
            )
        elif isinstance(columns_selection, int):
            columns_selection = slice(columns_selection, columns_selection + 1)

        self.columns_selection = columns_selection

    def transform(self, data_inputs):
        dtype = type(data_inputs)

        if isinstance(self.columns_selection, slice):
            ret = list(map(itemgetter(self.columns_selection), data_inputs))
        elif isinstance(self.columns_selection, list):
            if 'DataFrame' in str(type(data_inputs)):
                ret = data_inputs.loc[:, self.columns_selection]
            else:
                columns = [
                    list(map(itemgetter(i), data_inputs))
                    for i in self.columns_selection
                ]
                ret = list(zip(*columns))
        elif isinstance(self.columns_selection, str):
            ret = data_inputs.loc[:, [self.columns_selection]].values
        elif self.columns_selection is None:
            ret = data_inputs
        else:
            raise ValueError(
                'column selection type not supported : {0}\nSupported types'.format(
                    self.columns_selection,
                    repr(ColumnSelectionType)
                ))

        if dtype == np.ndarray and not isinstance(ret, np.ndarray):
            return np.array(ret)
        return ret


class NumpyColumnSelector2D(BaseTransformer):
    """
    A numpy version of the :class:`~neuraxle.steps.column_transformer.ColumnSelector2D`.
    """

    def __init__(self, columns_selection: ColumnSelectionType):
        super().__init__()
        self.column_selection = columns_selection

    def transform(self, data_inputs):
        if isinstance(self.column_selection, range):
            self.column_selection = slice(
                self.column_selection.start,
                self.column_selection.stop,
                self.column_selection.step
            )

        if isinstance(self.column_selection, int):
            return np.expand_dims(np.array(data_inputs)[:, self.column_selection], axis=-1)

        if isinstance(self.column_selection, slice):
            return np.array(data_inputs)[:, self.column_selection]

        if isinstance(self.column_selection, list):
            columns = [
                np.expand_dims(np.array(data_inputs)[:, i], axis=-1)
                for i in self.column_selection
            ]
            return np.concatenate(columns, axis=-1)

        if self.column_selection is None:
            return data_inputs

        raise ValueError(
            'column selection type not supported : {0}\nSupported types'.format(
                self.column_selection,
                repr(ColumnSelectionType)
            ))


class ColumnsSelectorND(MetaStep):
    """
    ColumnSelectorND wraps a ColumnSelector2D by as many ForEach step as needed to select the last dimension.
    n_dimension must therefore be greater or equal to 2.
    """

    def __init__(self, columns_selection, n_dimension=2):
        assert n_dimension >= 2
        col_selector: ColumnSelector2D = ColumnSelector2D(columns_selection=columns_selection)
        for _ in range(max(0, n_dimension - 2)):
            col_selector = ForEach(col_selector)

        MetaStep.__init__(self, col_selector)
        self.n_dimension = n_dimension


class ColumnTransformer(FeatureUnion):
    """
    A ColumnChooser can apply custom transformations to different columns.
    The ColumnChooser accepts a list of tuples for the transformations,
    and will name the steps accordingly (because of the TruncableSteps' constructor)
    by converting each indexer object to a string. Indexer objects can be ranges, an int, or a list of ints.
    The input data can be `N`-dimensionnal (ND), in which case the axis must be specified. The columns
    data passed to the sub-steps will still be ND.

    Usage example:

    .. code-block:: python

        ColumnChooser([
            (range(0, 2), CyclicTimes()),
            (3, CategoricalEnum(categories_count=5, starts_at_zero=True)),
            (4, CategoricalEnum(categories_count=5, starts_at_zero=True)),
            ([10, 13, 15], CategoricalEnum(categories_count=5, starts_at_zero=True)),
        ])

    .. seealso::
        :class:`~neuraxle.union.FeatureUnion`,
    """

    def __init__(self, column_chooser_steps_as_tuple: ColumnChooserTupleList, n_dimension: int = 3, n_jobs=None,
                 joiner: BaseTransformer = None):
        # Make unique names from the indices in case we have many steps for transforming the same column(s).
        self.string_indices = [
            str(name) + "_" + str(step.__class__.__name__)
            for name, step in column_chooser_steps_as_tuple
        ]

        FeatureUnion.__init__(self, [
            (string_indices, Pipeline([
                ColumnsSelectorND(indices, n_dimension=n_dimension),
                step
            ]))
            for string_indices, (indices, step) in zip(self.string_indices, column_chooser_steps_as_tuple)
        ], n_jobs=n_jobs, joiner=joiner)
