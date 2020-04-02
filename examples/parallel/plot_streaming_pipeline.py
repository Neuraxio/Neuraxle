"""
Parallel processing in Neuraxle
===================================================================

This demonstrates how to stream data in parallel in a Neuraxle pipeline.

..
    Copyright 2019, Neuraxio Inc.

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

from neuraxle.distributed.streaming import SequentialQueuedPipeline
from neuraxle.pipeline import Pipeline
from neuraxle.steps.loop import ForEachDataInput
from neuraxle.steps.misc import Sleep
from testing.test_streaming import MultiplyBy


def main():
    sleep_time = 0.02
    p = SequentialQueuedPipeline([
        Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)]),
        Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)]),
        Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)]),
        Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)])
    ], n_workers_per_step=8, max_size=10, batch_size=10)

    a = time.time()
    outputs_streaming = p.transform(list(range(100)))
    b = time.time()
    time_queued_pipeline = b - a
    print('SequentialQueuedPipeline')
    print('execution time: {} seconds'.format(time_queued_pipeline))

    p = Pipeline([
        Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)]),
        Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)]),
        Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)]),
        Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)])
    ])

    a = time.time()
    outputs_vanilla = p.transform(list(range(100)))
    b = time.time()
    time_vanilla_pipeline = b - a

    print('VanillaPipeline')
    print('execution time: {} seconds'.format(time_vanilla_pipeline))

    assert time_queued_pipeline < time_vanilla_pipeline
    assert np.array_equal(outputs_streaming, outputs_vanilla)


if __name__ == '__main__':
    main()
