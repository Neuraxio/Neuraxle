"""
Replace a method by another one.
=========================================================================

This demonstrates how to replace a method in a Neuraxle Pipeline.

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

from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import MultiplyByN


def main():
    p = Pipeline([
        MultiplyByN(2),
        MultiplyByN(4)
    ])

    outputs = p.transform(list(range(10)))
    print('transform: {}'.format(outputs))

    p = p.mutate(new_method='inverse_transform', method_to_assign_to='transform')

    outputs = p.transform(list(range(10)))
    print('inverse_transform: {}'.format(outputs))


if __name__ == '__main__':
    main()
