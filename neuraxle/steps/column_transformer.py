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
from typing import List, Tuple, Union

import numpy as np

from neuraxle.base import BaseStep, NonFittableMixin, MetaStepMixin
from neuraxle.pipeline import Pipeline
from neuraxle.steps.loop import ForEachDataInput
from neuraxle.union import FeatureUnion

ColumnSelectionType = Union[Tuple[int, BaseStep], Tuple[List[int], BaseStep], Tuple[slice, BaseStep]]
ColumnChooserTupleList = List[ColumnSelectionType]


class ColumnSelector2D(NonFittableMixin, BaseStep):
    """
    A ColumnSelector2D selects column in a sequence.
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


class ColumnsSelectorND(MetaStepMixin, BaseStep):
    """
    ColumnSelectorND wraps a ColumnSelector2D by as many ForEachDataInput step
    as needed to select the last dimension.
    """

    def __init__(self, columns_selection, n_dimension=3):
        BaseStep.__init__(self)

        col_selector = ColumnSelector2D(columns_selection=columns_selection)
        for _ in range(min(0, n_dimension - 2)):
            col_selector = ForEachDataInput(col_selector)

        MetaStepMixin.__init__(self, col_selector)
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

    def __init__(self, column_chooser_steps_as_tuple: ColumnChooserTupleList, n_dimension: int = 3):
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
        ])
