"""
Apply recursive operations to a pipeline
===========================================================

This demonstrates how to apply a method to each pipeline step.

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
import json

from scipy.stats import randint

from neuraxle.base import Identity
from neuraxle.hyperparams.space import RecursiveDict, HyperparameterSamples, HyperparameterSpace
from neuraxle.pipeline import Pipeline


class IdentityWithRvs(Identity):
    def _rvs(self):
        return HyperparameterSamples(self.hyperparams_space.rvs())


def rvs(step) -> RecursiveDict:
    return HyperparameterSamples(step.hyperparams_space.rvs())


def main():
    p = Pipeline([
        IdentityWithRvs().set_hyperparams_space(HyperparameterSpace({
            'a': randint(low=2, high=5)
        })),
        IdentityWithRvs().set_hyperparams_space(HyperparameterSpace({
            'b': randint(low=100, high=400)
        }))
    ])

    samples: HyperparameterSamples = p.apply(rvs)
    print('p.apply(rvs) ==>')
    print(json.dumps(samples, indent=4))

    # or equivalently:

    samples: HyperparameterSamples = p.apply('_rvs')
    print('p.apply(\'_rvs\') ==>')
    print(json.dumps(samples, indent=4))


if __name__ == '__main__':
    main()
