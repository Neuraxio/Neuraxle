import unittest
import numpy as np
from neuraxle.data_container import StripAbsentValues

from neuraxle.pipeline import MiniBatchSequentialPipeline, Joiner, Pipeline
from neuraxle.steps.misc import TransformCallbackStep, TapeCallbackFunction, FitTransformCallbackStep
from neuraxle.steps.numpy import MultiplyByN, ToList


class MultiplyBy2TransformCallbackStep(Pipeline):
    def __init__(self, tape, names):
        Pipeline.__init__(self, [
            TransformCallbackStep(tape, names),
            MultiplyByN(2),
            ToList(),
        ])


class MultiplyBy2FitTransformCallbackStep(Pipeline):
    def __init__(self, tape_transform, tape_fit, names):
        Pipeline.__init__(self, [
            FitTransformCallbackStep(tape_transform, tape_fit, names),
            MultiplyByN(2),
            ToList(),
        ])


def test_mini_batch_sequential_pipeline_should_transform_steps_sequentially_for_each_barrier_for_each_batch():
    # Given
    tapes = [TapeCallbackFunction() for _ in range(4)]
    p = MiniBatchSequentialPipeline([
        MultiplyBy2TransformCallbackStep(tapes[0], ["0"]),
        MultiplyBy2TransformCallbackStep(tapes[1], ["1"]),
        Joiner(batch_size=10),
        MultiplyBy2TransformCallbackStep(tapes[2], ["2"]),
        MultiplyBy2TransformCallbackStep(tapes[3], ["3"]),
        Joiner(batch_size=10)
    ])
    minibatch_1 = list(range(10))
    minibatch_2 = list(range(10, 20))
    full_batch = minibatch_1 + minibatch_2

    # When
    outputs = p.transform(full_batch)

    # Then
    for t, tape in enumerate(tapes):
        assert tape.name_tape == [str(t), str(t)]
        assert (tape.data[0] == MultiplyByN(2**t).transform(minibatch_1)).all()
        assert (tape.data[1] == MultiplyByN(2**t).transform(minibatch_2)).all()
    assert (outputs == MultiplyByN(2**4).transform(full_batch)).all()


def test_mini_batch_sequential_pipeline_should_fit_transform_steps_sequentially_for_each_barrier_for_each_batch():
    # Given
    tapes_trs = [TapeCallbackFunction() for _ in range(4)]
    tapes_fit = [TapeCallbackFunction() for _ in range(4)]
    p = MiniBatchSequentialPipeline([
        MultiplyBy2FitTransformCallbackStep(tapes_trs[0], tapes_fit[0], ["0"]),
        MultiplyBy2FitTransformCallbackStep(tapes_trs[1], tapes_fit[1], ["1"]),
        Joiner(batch_size=10),
        MultiplyBy2FitTransformCallbackStep(tapes_trs[2], tapes_fit[2], ["2"]),
        MultiplyBy2FitTransformCallbackStep(tapes_trs[3], tapes_fit[3], ["3"]),
        Joiner(batch_size=10)
    ])
    minibatch_1 = list(range(10))
    minibatch_2 = list(range(10, 20))
    full_batch = minibatch_1 + minibatch_2

    # When
    p, outputs = p.fit_transform(full_batch, full_batch)

    # Then
    for t, tape in enumerate(tapes_trs):
        assert tape.name_tape == [str(t), str(t)]
        assert (tape.data[0] == MultiplyByN(2**t).transform(minibatch_1)).all()
        assert (tape.data[1] == MultiplyByN(2**t).transform(minibatch_2)).all()
    for t, tape in enumerate(tapes_fit):
        assert tape.name_tape == [str(t), str(t)]
        # DI:
        assert (tape.data[0][0] == MultiplyByN(2**t).transform(minibatch_1)).all()
        assert (tape.data[1][0] == MultiplyByN(2**t).transform(minibatch_2)).all()
        # EO:
        assert (tape.data[0][1] == minibatch_1)
        assert (tape.data[1][1] == minibatch_2)
    assert (outputs == MultiplyByN(2**4).transform(full_batch)).all()


def test_minibatch_sequential_pipeline_change_batch_size_works():
    tape1 = TapeCallbackFunction()
    tape1_fit = TapeCallbackFunction()
    tape2 = TapeCallbackFunction()
    tape2_fit = TapeCallbackFunction()

    p = MiniBatchSequentialPipeline([
        MultiplyBy2FitTransformCallbackStep(tape1, tape1_fit, ["1"]),
        Joiner(batch_size=10),
        MultiplyBy2FitTransformCallbackStep(tape2, tape2_fit, ["2"]),
        Joiner(batch_size=10)
    ])

    # When
    p, outputs = p.fit_transform(list(range(20)), list(range(20)))
    p.set_batch_size(5)
    p, outputs = p.fit_transform(list(range(20, 30)), list(range(20, 30)))

    # Then

    assert tape1.data == [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]
    ]
    assert tape1_fit.data == [
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
        ([20, 21, 22, 23, 24], [20, 21, 22, 23, 24]),
        ([25, 26, 27, 28, 29], [25, 26, 27, 28, 29])
    ]
    assert tape1.name_tape == ["1", "1", "1", "1"]

    assert tape2.data == [
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18], [20, 22, 24, 26, 28, 30, 32, 34, 36, 38],
        [40, 42, 44, 46, 48], [50, 52, 54, 56, 58]
    ]
    assert tape2_fit.data == [
        ([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ([20, 22, 24, 26, 28, 30, 32, 34, 36, 38], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
        ([40, 42, 44, 46, 48], [20, 21, 22, 23, 24]),
        ([50, 52, 54, 56, 58], [25, 26, 27, 28, 29])
    ]
    assert tape2.name_tape == ["2", "2", "2", "2"]


class TestMiniBatchSequentialPipelineBatchingBehavior(unittest.TestCase):

    def setUp(self):
        self.data = list(range(10))
        self.tape_trs = TapeCallbackFunction()
        self.tape_fit = TapeCallbackFunction()
        self.tape_step = FitTransformCallbackStep(self.tape_trs, self.tape_fit, ["0"])

    def test_batch_size_2(self):
        pipeline = MiniBatchSequentialPipeline([
            self.tape_step
        ], batch_size=2)

        pipeline.transform(self.data)

        assert self.tape_trs.data == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    def test_batch_size_3_discard_incomplete(self):
        pipeline = MiniBatchSequentialPipeline([
            self.tape_step
        ], batch_size=3, keep_incomplete_batch=False)

        pipeline.transform(self.data)

        assert self.tape_trs.data == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    def test_batch_size_3_keep_incomplete_using_nones(self):
        pipeline = MiniBatchSequentialPipeline(
            [self.tape_step],
            batch_size=3,
            keep_incomplete_batch=True,
            default_value_data_inputs=None,
            default_value_expected_outputs=None
        )

        pipeline.transform(self.data)

        assert self.tape_trs.data == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, None, None]]

    def test_batch_size_3_keep_incomplete_using_nones_and_absent_incomplete_batch_values(self):
        pipeline = MiniBatchSequentialPipeline(
            [self.tape_step],
            batch_size=3,
            keep_incomplete_batch=True,
            default_value_data_inputs=StripAbsentValues()
        )

        pipeline.transform(self.data)

        assert self.tape_trs.data == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    def test_batch_size_2_using_joiner(self):
        pipeline = MiniBatchSequentialPipeline([
            self.tape_step,
            Joiner(batch_size=2)
        ])

        pipeline.transform(self.data)

        assert self.tape_trs.data == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    def test_batch_size_3_using_joiner(self):
        pipeline = MiniBatchSequentialPipeline([
            self.tape_step,
            Joiner(batch_size=3, keep_incomplete_batch=False)
        ])

        pipeline.transform(self.data)

        assert self.tape_trs.data == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    def test_batch_size_3_using_joiner_keep_incomplete_using_nones(self):
        pipeline = MiniBatchSequentialPipeline([
            self.tape_step,
            Joiner(
                batch_size=3,
                keep_incomplete_batch=True,
                default_value_data_inputs=None,
                default_value_expected_outputs=None
            )
        ])

        pipeline.transform(self.data)

        assert self.tape_trs.data == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, None, None]]

    def test_batch_size_3_using_joiner_keep_incomplete_using_absent_incomplete_batch_values(self):
        pipeline = MiniBatchSequentialPipeline([
            self.tape_step,
            Joiner(
                batch_size=3,
                keep_incomplete_batch=True,
                default_value_data_inputs=StripAbsentValues()
            )
        ])

        pipeline.transform(self.data)

        assert self.tape_trs.data == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    def test_fit_with_keep_incomplete_batch_using_nones(self):
        pipeline = MiniBatchSequentialPipeline([
            self.tape_step,
            Joiner(
                batch_size=3,
                keep_incomplete_batch=True,
                default_value_data_inputs=None,
                default_value_expected_outputs=None
            )
        ])

        pipeline.fit(self.data, self.data)

        assert self.tape_fit.data == [
            ([0, 1, 2], [0, 1, 2]),
            ([3, 4, 5], [3, 4, 5]),
            ([6, 7, 8], [6, 7, 8]),
            ([9, None, None], [9, None, None]),
        ]

    def test_fit_with_keep_incomplete_batch_using_partially_nones_in_inputs(self):
        pipeline = MiniBatchSequentialPipeline([
            self.tape_step,
            Joiner(
                batch_size=3,
                keep_incomplete_batch=True,
                default_value_data_inputs=None,
                default_value_expected_outputs=StripAbsentValues()
            )
        ])

        pipeline.fit(self.data, self.data)

        assert self.tape_fit.data == [
            ([0, 1, 2], [0, 1, 2]),
            ([3, 4, 5], [3, 4, 5]),
            ([6, 7, 8], [6, 7, 8]),
            ([9, None, None], [9]),
        ]

    def test_fit_with_keep_incomplete_batch_using_partially_nones_in_outputs(self):
        pipeline = MiniBatchSequentialPipeline([
            self.tape_step,
            Joiner(
                batch_size=3,
                keep_incomplete_batch=True,
                default_value_data_inputs=StripAbsentValues(),
                default_value_expected_outputs=None
            )
        ])

        pipeline.fit(self.data, self.data)

        assert self.tape_fit.data == [
            ([0, 1, 2], [0, 1, 2]),
            ([3, 4, 5], [3, 4, 5]),
            ([6, 7, 8], [6, 7, 8]),
            ([9], [9, None, None]),
        ]

    def test_fit_with_keep_incomplete_batch_using_absent_incomplete_batch_values(self):
        pipeline = MiniBatchSequentialPipeline([
            self.tape_step,
            Joiner(
                batch_size=3,
                keep_incomplete_batch=True,
                default_value_data_inputs=StripAbsentValues(),
                default_value_expected_outputs=StripAbsentValues()
            )
        ])

        pipeline.fit(self.data, self.data)

        assert self.tape_fit.data == [
            ([0, 1, 2], [0, 1, 2]),
            ([3, 4, 5], [3, 4, 5]),
            ([6, 7, 8], [6, 7, 8]),
            ([9], [9]),
        ]
