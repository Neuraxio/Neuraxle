"""
Usage of ValueCachingWrapper in Neuraxle.
=============================================================

This demonstrates how you can use value caching in a Neuraxle pipeline.

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
import os
import shutil
import time

import numpy as np

from neuraxle.pipeline import Pipeline
from neuraxle.steps.caching import JoblibTransformDIValueCachingWrapper
from neuraxle.steps.loop import ForEach
from neuraxle.steps.misc import Sleep
from neuraxle.steps.numpy import MultiplyByN


def main():
    value_caching_folder = 'value_caching'
    if not os.path.exists(value_caching_folder):
        os.makedirs(value_caching_folder)

    data_inputs = list(range(100))

    sleep_time = 0.001
    a = time.time()
    for i in range(5):
        p = Pipeline([
            JoblibTransformDIValueCachingWrapper(
                ForEach(Pipeline([Sleep(sleep_time=sleep_time), MultiplyByN(2)])),
                cache_folder=value_caching_folder
            )
        ])
        outputs_value_caching = p.transform(data_inputs)
    b = time.time()
    time_value_caching_pipeline = b - a
    print('Pipeline with ValueCachingWrapper')
    print('execution time: {} seconds'.format(time_value_caching_pipeline))

    a = time.time()
    for i in range(5):
        p = Pipeline([
            ForEach(Pipeline([Sleep(sleep_time=sleep_time), MultiplyByN(2)])),
        ])

        outputs_vanilla = p.transform(data_inputs)
    b = time.time()
    time_vanilla_pipeline = b - a
    print('Pipeline without value caching')
    print('execution time: {} seconds'.format(time_vanilla_pipeline))

    shutil.rmtree(value_caching_folder)

    assert np.array_equal(outputs_value_caching, outputs_vanilla)
    assert time_value_caching_pipeline < time_vanilla_pipeline


if __name__ == '__main__':
    main()
