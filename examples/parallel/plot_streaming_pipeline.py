"""
Parallel processing in Neuraxle
===================================================================

This demonstrates how to stream data in parallel in a Neuraxle pipeline.
The pipeline steps' parallelism here will be obvious.

The pipeline has two steps:
1. Preprocessing: the step that process the data simply sleeps.
2. Model: the model simply multiplies the data by two.

This can be used with scikit-learn as well to transform things in parallel,
and any other library such as tensorflow.

Pipelines benchmarked:
1. We first use a classical pipeline and evaluate the time.
2. Then we use a minibatched pipeline and we evaluate the time.
3. Then we use a parallel pipeline and we evaluate the time.

We expect the parallel pipeline to be faster due to having more workers
in parallel, as well as starting the model's transformations at the same
time that other batches are being preprocessed, using queues.


..
    Copyright 2021, Neuraxio Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.


"""
import time
import numpy as np
from neuraxle.base import ExecutionContext

from neuraxle.distributed.streaming import SequentialQueuedPipeline
from neuraxle.pipeline import BasePipeline, Pipeline, MiniBatchSequentialPipeline
from neuraxle.steps.loop import ForEach
from neuraxle.steps.misc import Sleep
from neuraxle.steps.numpy import MultiplyByN


def eval_run_time(pipeline: BasePipeline):
    pipeline.setup(ExecutionContext())
    a = time.time()
    output = pipeline.transform(list(range(100)))
    b = time.time()
    seconds = b - a
    return seconds, output


def main():
    """
    The task is to sleep 0.02 seconds for each data input and then multiply by 2.
    """
    sleep_time = 0.02
    preprocessing_and_model_steps = [ForEach(Sleep(sleep_time=sleep_time)), MultiplyByN(2)]

    # Classical pipeline - all at once with one big batch:
    p = Pipeline(preprocessing_and_model_steps)
    time_vanilla_pipeline, output_classical = eval_run_time(p)
    print(f"Classical 'Pipeline' execution time: {time_vanilla_pipeline} seconds.")

    # Classical minibatch pipeline - minibatch size 25:
    p = MiniBatchSequentialPipeline(preprocessing_and_model_steps,
                                    batch_size=25)
    time_minibatch_pipeline, output_minibatch = eval_run_time(p)
    print(f"Minibatched 'MiniBatchSequentialPipeline' execution time: {time_minibatch_pipeline} seconds.")

    # Parallel pipeline - minibatch size 25 with 4 parallel workers per step that
    # have a max queue size of 10 batches between preprocessing and the model:
    p = SequentialQueuedPipeline(preprocessing_and_model_steps,
                                 n_workers_per_step=4, max_queue_size=10, batch_size=25)
    time_parallel_pipeline, output_parallel = eval_run_time(p)
    print(f"Parallel 'SequentialQueuedPipeline' execution time: {time_parallel_pipeline} seconds.")

    assert time_parallel_pipeline < time_minibatch_pipeline, str((time_parallel_pipeline, time_vanilla_pipeline))
    assert np.array_equal(output_classical, output_minibatch)
    assert np.array_equal(output_classical, output_parallel)


if __name__ == '__main__':
    main()
