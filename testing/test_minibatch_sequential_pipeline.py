from neuraxle.pipeline import MiniBatchSequentialPipeline, Barrier
from testing.test_pipeline import SomeStep


def test_mini_batch_sequential_pipeline_should_transform_steps_sequentially_for_each_barrier_for_each_batch():
    p = MiniBatchSequentialPipeline([
        SomeStep(),
        SomeStep(),
        Barrier(),
        SomeStep(),
        SomeStep(),
        Barrier()
    ], batch_size=10)

    outputs = p.transform(range(100))

    # assert steps have received the right batches
    # assert steps have transformed the data correctly


def test_mini_batch_sequential_pipeline_should_fit_transform_steps_sequentially_for_each_barrier_for_each_batch():
    p = MiniBatchSequentialPipeline([
        SomeStep(),
        SomeStep(),
        Barrier(),
        SomeStep(),
        SomeStep(),
        Barrier()
    ], batch_size=10)

    outputs = p.transform(range(100))

    # assert steps have received the right batches
    # assert steps have transformed the data correctly
    # assert steps have been fitted with the right batches in the right order


def test_mini_batch_sequential_pipeline_joiner():
    p = MiniBatchSequentialPipeline([
        SomeStep(),
        SomeStep(),
        Barrier(),
        SomeStep(),
        SomeStep(),
        Barrier()
    ], batch_size=10)

    # TODO: joiner ????????

