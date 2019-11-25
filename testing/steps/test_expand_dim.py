from typing import List, Iterable

import numpy as np

from neuraxle.base import BaseHasher
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline
from neuraxle.steps.flow import ExpandDim
from neuraxle.steps.misc import HandleCallbackStep, TapeCallbackFunction

SUMMARY_ID = 'b79cdac314fb78f2cd38f23e74c5fc66'


class SomeSummaryHasher(BaseHasher):
    def __init__(self, fake_summary_id):
        self.fake_summary_id = fake_summary_id

    def single_hash(self, current_id: str, hyperparameters: HyperparameterSamples) -> List[str]:
        return self.fake_summary_id

    def hash(self, current_ids: List[str], hyperparameters: HyperparameterSamples, data_inputs: Iterable) -> List[str]:
        return [self.fake_summary_id]


def test_expand_dim_transform():
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
    p['ExpandDim'].hashers = [SomeSummaryHasher(fake_summary_id=SUMMARY_ID)]

    outputs = p.transform(np.array(range(10)))

    assert np.array_equal(outputs, np.array(range(10)))
    assert handle_fit_callback.data == []
    assert handle_transform_callback.data[0][0].current_ids == [SUMMARY_ID]
    assert np.array_equal(
        np.array(handle_transform_callback.data[0][0].data_inputs),
        np.array([np.array(range(10))])
    )
    assert np.array_equal(
        np.array(handle_transform_callback.data[0][0].expected_outputs),
        np.array([[None] * 10])
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
    p['ExpandDim'].hashers = [SomeSummaryHasher(fake_summary_id=SUMMARY_ID)]

    p = p.fit(np.array(range(10)), np.array(range(10)))

    assert handle_transform_callback.data == []
    assert handle_fit_transform_callback.data == []
    assert handle_fit_callback.data[0][0].current_ids == [SUMMARY_ID]
    assert handle_fit_callback.data[0][0].summary_id == SUMMARY_ID
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
    p['ExpandDim'].hashers = [SomeSummaryHasher(fake_summary_id=SUMMARY_ID)]

    p, outputs = p.fit_transform(np.array(range(10)), np.array(range(10)))

    assert np.array_equal(outputs, np.array(range(10)))
    assert handle_transform_callback.data == []
    assert handle_fit_callback.data == []
    assert handle_fit_transform_callback.data[0][0].current_ids == [SUMMARY_ID]
    assert handle_fit_transform_callback.data[0][0].summary_id == SUMMARY_ID
    assert np.array_equal(
        np.array(handle_fit_transform_callback.data[0][0].data_inputs),
        np.array([np.array(range(10))])
    )
    assert np.array_equal(
        np.array(handle_fit_transform_callback.data[0][0].expected_outputs),
        np.array([np.array(range(10))])
    )
