"""
Create Pipeline Steps that require implementing only handler methods
========================================================================================================================

If a pipeline step only needs to implement handler methods, then you can inherit from the ForceHandleMixin as demonstrated here.
Handler methods are useful when :

    - You need to change the shape of the data container passed to the following steps, or the wrapped steps.
    - You want to apply side effects based on the data container, and the execution context.
    - You want to change the pipeline execution flow based on the data container, and the execution context.

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

..
    Thanks to Umaneo Technologies Inc. for their contributions to this Machine Learning
    project, visit https://www.umaneo.com/ for more information on Umaneo Technologies Inc.

"""
import numpy as np

from neuraxle.base import BaseStep, ForceAlwaysHandleMixin, DataContainer, ExecutionContext
from neuraxle.pipeline import Pipeline


class ForceAlwaysAlwaysHandleMixinStep(ForceAlwaysHandleMixin, BaseStep):
    """
    Please make your steps inherit from ForceHandleMixin when they only implement handle_methods, but also
    when you want to make impossible the use of regular fit, transform, and fit_transform methods
    Also, make sure that BaseStep is the last step you inherit from.
    """

    def __init__(self):
        ForceAlwaysHandleMixin.__init__(self)
        BaseStep.__init__(self)

    def handle_fit(self, data_container: DataContainer, context: ExecutionContext):
        """
        Change the shape of the data container.
        and/or
        Apply any side effects based on the data container
        And/or
        Change the execution flow of the pipeline
        """
        data_container = self.handle_after_any(data_container)
        return self, data_container

    def handle_transform(self, data_container: DataContainer, context: ExecutionContext):
        """
        Change the shape of the data container.
        and/or
        Apply any side effects based on the data container
        And/or
        Change the execution flow of the pipeline
        """
        data_container = self.handle_after_any(data_container)
        return data_container

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext):
        """
        Change the shape of the data container.
        and/or
        Apply any side effects based on the data container
        And/or
        Change the execution flow of the pipeline
        """
        data_container = self.handle_after_any(data_container)
        return self, data_container


def main():
    p = Pipeline([
        ForceAlwaysAlwaysHandleMixinStep(),
    ])

    p = p.fit(np.array([0, 1]), np.array([0, 1]))
    p = p.transform(np.array([0, 1]))
    
    # Note that calling `ForceAlwaysAlwaysHandleMixinStep().fit(...)`
    # on some data would have crashed. We would need to instead do
    # `ForceAlwaysAlwaysHandleMixinStep().handle_fit(...)` with a handle.


if __name__ == "__main__":
    main()
