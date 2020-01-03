from neuraxle.distributed.streaming import QueuedPipeline
from neuraxle.steps.misc import TapeCallbackFunction
from testing.test_minibatch_sequential_pipeline import MultiplyBy2FitTransformCallbackStep


def test_queued_pipeline():
    # Given
    tape1 = TapeCallbackFunction()
    tape1_fit = TapeCallbackFunction()
    tape2 = TapeCallbackFunction()
    tape2_fit = TapeCallbackFunction()
    tape3 = TapeCallbackFunction()
    tape3_fit = TapeCallbackFunction()
    tape4 = TapeCallbackFunction()
    tape4_fit = TapeCallbackFunction()

    p = QueuedPipeline([
        MultiplyBy2FitTransformCallbackStep(tape1, tape1_fit, ["1"]),
        MultiplyBy2FitTransformCallbackStep(tape2, tape2_fit, ["2"]),
        MultiplyBy2FitTransformCallbackStep(tape3, tape3_fit, ["3"]),
        MultiplyBy2FitTransformCallbackStep(tape4, tape4_fit, ["4"]),
    ], max_batches=10, batch_size=10)

    # When
    p, outputs = p.fit_transform(range(100), range(100))

    assert len(outputs) == 100
