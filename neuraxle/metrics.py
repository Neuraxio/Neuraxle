"""
Neuraxle's metrics classes
=================================================
The neuraxle classes to track metrics results.

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
from typing import Dict

from neuraxle.base import MetaStepMixin, BaseStep, ExecutionContext
from neuraxle.data_container import DataContainer


class MetricsWrapper(MetaStepMixin, BaseStep):
    """
    Add metrics calculation to a step. Calculates metrics after each fit, fit_transform, or even transform if there is an expected outputs.

    Example usage :

    .. code-block:: python

        wrapped = MetricsWrapper(wrapped=wrapped, metrics=self.batch_metrics, name=BATCH_METRICS_STEP_NAME)
        wrapped = MiniBatchSequentialPipeline(
            [wrapped],
            batch_size=self.batch_size,
            enabled=True
        )

        # toggle metrics on, and off
        wrapped.apply('toggle_metrics')


    .. seealso::
        :class:`~neuraxle.api.DeepLearningPipeline`,
        :class:`~neuraxle.base.MetaStepMixin`,
        :class:`~neuraxle.base.BaseStep`
    """

    def __init__(
            self,
            wrapped: BaseStep,
            metrics: Dict,
            name: str = None,
            print_metrics=False,
            print_fun=print
    ):
        BaseStep.__init__(self, name=name)
        MetaStepMixin.__init__(self, wrapped)

        self.metrics: Dict = metrics
        self._initialize_metrics(metrics)

        self.print_metrics = print_metrics
        self.print_fun = print_fun
        self.enabled = True

    def _initialize_metrics(self, metrics):
        """
        Initialize metrics results dict for train, and validation using the metrics function dict.

        :param metrics: metrics function dict
        :type metrics: dict
        :return:
        """
        self.metrics_results_train = {}
        self.metrics_results_validation = {}

        for m in metrics:
            self.metrics_results_train[m] = []
            self.metrics_results_validation[m] = []

    def _did_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        return self._did_fit_transform_or_transform(data_container, context)

    def _did_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        return self._did_fit_transform_or_transform(data_container, context)

    def _did_fit_transform_or_transform(self, data_container: DataContainer, context: ExecutionContext):
        """
        Calculate metrics results after fit, or transform if there is an expected outputs in the data container.
        Also, calculate validation metrics if there is a sub data container named validation in the data container.
        Please refer to :class:`~neuraxle.data_container.DataContainer` for more information about sub data containers.

        :param data_container: data container to calculate metrics for
        :type data_container: DataContainer
        :return: data container
        :rtype: DataContainer
        """
        if data_container.expected_outputs is None or len(data_container.expected_outputs) == 0:
            return data_container

        self._calculate_metrics_results(data_container)

        if 'validation' in data_container and self.enabled:
            self.set_train(False)

            self.apply('disable_metrics')
            validation_data_container = self._transform_data_container(data_container['validation'].copy(), context)
            self.apply('enable_metrics')
            self._calculate_metrics_results(validation_data_container)

            self.set_train(True)

        return data_container

    def _calculate_metrics_results(self, data_container: DataContainer):
        """
        Calculate metrics results using the transformed data container, and the metrics function dict.

        :param data_container: transformed data container
        :type data_container: DataContainer
        :return:
        """
        if not self.enabled:
            return

        result = {}
        for metric_name, metric_function in self.metrics.items():
            result_metric = metric_function(data_container.data_inputs, data_container.expected_outputs)
            result[metric_name] = result_metric

            if self.is_train:
                self.metrics_results_train[metric_name].append(result_metric)
            else:
                self.metrics_results_validation[metric_name].append(result_metric)

        if self.print_metrics:
            self.print_fun(result)

    def get_metrics(self) -> Dict:
        """
        Get all metrics results using the transformed data container, and the metrics function dict.
        To be used with :func:`neuraxle.base.BaseStep.apply` method.

        :return: dict with the step name as key, and all of the training, and validation metrics as values
        """
        return {
            'train': self.metrics_results_train,
            'validation': self.metrics_results_validation
        }

    def toggle_metrics(self):
        """
        Toggle metrics wrapper on and off to temporarily disable metrics if needed..

        :return:
        """
        self.enabled = not self.enabled

    def disable_metrics(self):
        """
        Disable metrics wrapper metrics if needed..

        :return:
        """
        self.enabled = False

    def enable_metrics(self):
        """
        Enable metrics wrapper metrics if needed..

        :return:
        """
        self.enabled = True
