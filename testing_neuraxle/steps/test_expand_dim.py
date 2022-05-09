from typing import List, Iterable

import numpy as np

from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline
from neuraxle.steps.flow import ExpandDim
from neuraxle.steps.misc import HandleCallbackStep, TapeCallbackFunction


def test_expand_dim_transform():
    di = np.array(range(10))
    eo = [None] * 10
    handle_fit_callback = TapeCallbackFunction()
    handle_transform_callback = TapeCallbackFunction()
    handle_fit_transform_callback = TapeCallbackFunction()
    p = Pipeline([
        ExpandDim(
            HandleCallbackStep(
                handle_fit_callback,
                handle_transform_callback,
                handle_fit_transform_callback
            )
        )
    ])

    outputs = p.transform(di)

    assert np.array_equal(outputs, di)
    assert handle_fit_callback.data == []
    assert np.array_equal(
        np.array(handle_transform_callback.data[0][0].di),
        np.array([di])
    )
    assert np.array_equal(
        np.array(handle_transform_callback.data[0][0].eo),
        np.array([eo])
    )
    assert handle_fit_transform_callback.data == []


def test_expand_dim_fit():
    handle_fit_callback = TapeCallbackFunction()
    handle_transform_callback = TapeCallbackFunction()
    handle_fit_transform_callback = TapeCallbackFunction()
    p = Pipeline([
        ExpandDim(
            HandleCallbackStep(
                handle_fit_callback,
                handle_transform_callback,
                handle_fit_transform_callback
            )
        )
    ])

    p = p.fit(np.array(range(10)), np.array(range(10)))

    assert handle_transform_callback.data == []
    assert handle_fit_transform_callback.data == []
    assert np.array_equal(
        np.array(handle_fit_callback.data[0][0].data_inputs),
        np.array([np.array(range(10))])
    )
    assert np.array_equal(
        np.array(handle_fit_callback.data[0][0].expected_outputs),
        np.array([np.array(range(10))])
    )


def test_expand_dim_fit_transform():
    handle_fit_callback = TapeCallbackFunction()
    handle_transform_callback = TapeCallbackFunction()
    handle_fit_transform_callback = TapeCallbackFunction()
    p = Pipeline([
        ExpandDim(
            HandleCallbackStep(
                handle_fit_callback,
                handle_transform_callback,
                handle_fit_transform_callback
            )
        )
    ])

    p, outputs = p.fit_transform(np.array(range(10)), np.array(range(10)))

    assert np.array_equal(outputs, np.array(range(10)))
    assert handle_transform_callback.data == []
    assert handle_fit_callback.data == []
    assert np.array_equal(
        np.array(handle_fit_transform_callback.data[0][0].data_inputs),
        np.array([np.array(range(10))])
    )
    assert np.array_equal(
        np.array(handle_fit_transform_callback.data[0][0].expected_outputs),
        np.array([np.array(range(10))])
    )
