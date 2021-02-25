import numpy as np

from neuraxle.pipeline import MiniBatchSequentialPipeline, Joiner
from neuraxle.steps.misc import TransformCallbackStep, TapeCallbackFunction, FitTransformCallbackStep


class MultiplyBy2TransformCallbackStep(TransformCallbackStep):
    def transform(self, data_inputs):
        TransformCallbackStep.transform(self, data_inputs)

        return list(np.array(data_inputs) * 2)


def test_mini_batch_sequential_pipeline_should_transform_steps_sequentially_for_each_barrier_for_each_batch():
    # Given
    tape1 = TapeCallbackFunction()
    tape2 = TapeCallbackFunction()
    tape3 = TapeCallbackFunction()
    tape4 = TapeCallbackFunction()
    p = MiniBatchSequentialPipeline([
        MultiplyBy2TransformCallbackStep(tape1, ["1"]),
        MultiplyBy2TransformCallbackStep(tape2, ["2"]),
        Joiner(batch_size=10),
        MultiplyBy2TransformCallbackStep(tape3, ["3"]),
        MultiplyBy2TransformCallbackStep(tape4, ["4"]),
        Joiner(batch_size=10)
    ])

    # When
    outputs = p.transform(list(range(20)))

    # Then
    assert outputs == [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304]

    assert tape1.data == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
    assert tape1.name_tape == ["1", "1"]

    assert tape2.data == [[0, 2, 4, 6, 8, 10, 12, 14, 16, 18], [20, 22, 24, 26, 28, 30, 32, 34, 36, 38]]
    assert tape2.name_tape == ["2", "2"]

    assert tape3.data == [[0, 4, 8, 12, 16, 20, 24, 28, 32, 36], [40, 44, 48, 52, 56, 60, 64, 68, 72, 76]]
    assert tape3.name_tape == ["3", "3"]

    assert tape4.data == [[0, 8, 16, 24, 32, 40, 48, 56, 64, 72], [80, 88, 96, 104, 112, 120, 128, 136, 144, 152]]
    assert tape4.name_tape == ["4", "4"]


class MultiplyBy2FitTransformCallbackStep(FitTransformCallbackStep):
    def fit_transform(self, data_inputs, expected_outputs=None):
        FitTransformCallbackStep.fit_transform(self, data_inputs, expected_outputs)

        return self, list(np.array(data_inputs) * 2)


def test_mini_batch_sequential_pipeline_should_fit_transform_steps_sequentially_for_each_barrier_for_each_batch():
    # Given
    tape1 = TapeCallbackFunction()
    tape1_fit = TapeCallbackFunction()
    tape2 = TapeCallbackFunction()
    tape2_fit = TapeCallbackFunction()
    tape3 = TapeCallbackFunction()
    tape3_fit = TapeCallbackFunction()
    tape4 = TapeCallbackFunction()
    tape4_fit = TapeCallbackFunction()
    p = MiniBatchSequentialPipeline([
        MultiplyBy2FitTransformCallbackStep(tape1, tape1_fit, ["1"]),
        MultiplyBy2FitTransformCallbackStep(tape2, tape2_fit, ["2"]),
        Joiner(batch_size=10),
        MultiplyBy2FitTransformCallbackStep(tape3, tape3_fit, ["3"]),
        MultiplyBy2FitTransformCallbackStep(tape4, tape4_fit, ["4"]),
        Joiner(batch_size=10)
    ])

    # When
    p, outputs = p.fit_transform(list(range(20)), list(range(20)))

    # Then
    assert outputs == [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304]

    assert tape1.data == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
    assert tape1_fit.data == [([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                              ([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])]
    assert tape1.name_tape == ["1", "1"]

    assert tape2.data == [[0, 2, 4, 6, 8, 10, 12, 14, 16, 18], [20, 22, 24, 26, 28, 30, 32, 34, 36, 38]]
    assert tape2_fit.data == [([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                              ([20, 22, 24, 26, 28, 30, 32, 34, 36, 38], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])]
    assert tape2.name_tape == ["2", "2"]

    assert tape3.data == [[0, 4, 8, 12, 16, 20, 24, 28, 32, 36], [40, 44, 48, 52, 56, 60, 64, 68, 72, 76]]
    assert tape3_fit.data == [([0, 4, 8, 12, 16, 20, 24, 28, 32, 36], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                              ([40, 44, 48, 52, 56, 60, 64, 68, 72, 76], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])]
    assert tape3.name_tape == ["3", "3"]

    assert tape4.data == [[0, 8, 16, 24, 32, 40, 48, 56, 64, 72], [80, 88, 96, 104, 112, 120, 128, 136, 144, 152]]
    assert tape4_fit.data == [([0, 8, 16, 24, 32, 40, 48, 56, 64, 72], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                              ([80, 88, 96, 104, 112, 120, 128, 136, 144, 152], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])]
    assert tape4.name_tape == ["4", "4"]

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
    p, outputs = p.fit_transform(list(range(20,30)), list(range(20,30)))

    # Then

    assert tape1.data == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                          [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
    assert tape1_fit.data == [([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                              ([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                              ([20, 21, 22, 23, 24], [20, 21, 22, 23, 24]),
                              ([25, 26, 27, 28, 29], [25, 26, 27, 28, 29])]
    assert tape1.name_tape == ["1", "1", "1", "1"]

    assert tape2.data == [[0, 2, 4, 6, 8, 10, 12, 14, 16, 18], [20, 22, 24, 26, 28, 30, 32, 34, 36, 38],
                          [40, 42, 44, 46, 48], [50, 52, 54, 56, 58]]
    assert tape2_fit.data == [([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                              ([20, 22, 24, 26, 28, 30, 32, 34, 36, 38], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                              ([40, 42, 44, 46, 48], [20, 21, 22, 23, 24]),
                              ([50, 52, 54, 56, 58], [25, 26, 27, 28, 29])]
    assert tape2.name_tape == ["2", "2", "2", "2"]


