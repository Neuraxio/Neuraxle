from typing import Dict

from neuraxle.base import MetaStepMixin, BaseStep, ExecutionContext
from neuraxle.data_container import DataContainer


class MetricsWrapper(MetaStepMixin, BaseStep):
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
        self.print_metrics = print_metrics
        self.print_fun = print_fun
        self.metrics_results_train = []
        self.metrics_results_validation = []
        self.enabled = True

    def _did_process(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        if data_container.expected_outputs is not None and len(data_container.expected_outputs) > 0:
            self._calculate_metrics_results(data_container)

        return data_container

    def _calculate_metrics_results(self, data_container):
        if not self.enabled:
            return

        result = {}
        for metric_name, metric_fun in self.metrics.items():
            result[metric_name] = metric_fun(data_container.data_inputs, data_container.expected_outputs)

        if self.is_train:
            self.metrics_results_train.append(result)
        else:
            self.metrics_results_validation.append(result)

        if self.print_metrics:
            self.print_fun(result)

    def get_metric_validation(self, metric_name):
        results = []
        for m in self.metrics_results_validation:
            results.append(m[metric_name])

        return results

    def get_metric_train(self, metric_name):
        results = []
        for m in self.metrics_results_train:
            results.append(m[metric_name])

        return results

    def toggle_metrics(self):
        self.enabled = not self.enabled
