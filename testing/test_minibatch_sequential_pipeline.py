from neuraxle.pipeline import MiniBatchSequentialPipeline, Barrier
from testing.test_pipeline import SomeStep


def test_streaming_pipeline_should_transform_steps_sequentially_for_each_batch():
    p = MiniBatchSequentialPipeline([
        SomeStep(),
        SomeStep(),
        Barrier(),
        SomeStep(),
        SomeStep(),
        Barrier()
    ], batch_size=10)

    p.transform()
