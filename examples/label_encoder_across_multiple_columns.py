import numpy as np
import pandas
from sklearn.preprocessing import LabelEncoder

from neuraxle.steps.column_transformer import ColumnTransformer
from neuraxle.steps.loop import FlattenForEach

# Discussion:
# https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
df = pandas.DataFrame({
    'pets': ['cat', 'dog', 'cat', 'monkey', 'dog', 'dog'],
    'owner': ['Champ', 'Ron', 'Brick', 'Champ', 'Veronica', 'Ron'],
    'location': ['San_Diego', 'New_York', 'New_York', 'San_Diego', 'San_Diego', 'New_York']
})


def _apply_same_encoder_to_all_columns():
    """
    One shared LabelEncoder will be applied on all the data to encode it.
    """
    p = FlattenForEach(LabelEncoder(), then_unflatten=True)

    p, predicted_output = p.fit_transform(df.values)

    expected_output = np.array([
        [6, 7, 6, 8, 7, 7],
        [1, 3, 0, 1, 5, 3],
        [4, 2, 2, 4, 4, 2]
    ]).transpose()
    assert np.array_equal(predicted_output, expected_output)


def _apply_different_encoders_to_columns():
    """
    One standalone LabelEncoder will be applied on the pets,
    and another one will be shared for the columns owner and location.
    """
    p = ColumnTransformer([
        # A different encoder will be used for column 0 with name "pets":
        (0, FlattenForEach(LabelEncoder(), then_unflatten=True)),
        # A shared encoder will be used for column 1 and 2, "owner" and "location":
        ([1, 2], FlattenForEach(LabelEncoder(), then_unflatten=True)),
    ], n_dimension=2)

    p, predicted_output = p.fit_transform(df.values)

    expected_output = np.array([
        [0, 1, 0, 2, 1, 1],
        [1, 3, 0, 1, 5, 3],
        [4, 2, 2, 4, 4, 2]
    ]).transpose()
    assert np.array_equal(predicted_output, expected_output)


def main():
    _apply_same_encoder_to_all_columns()
    _apply_different_encoders_to_columns()


if __name__ == "__main__":
    main()
