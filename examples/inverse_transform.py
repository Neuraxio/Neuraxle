import numpy as np

from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import MultiplyByN


def main():
    p = Pipeline([MultiplyByN(multiply_by=2)])

    _in = np.array([1, 2])

    _out = p.transform(_in)

    assert np.array_equal(_out, np.array([2, 4]))

    _regenerated_in = reversed(p).transform(_out)

    assert np.array_equal(_regenerated_in, _in)
