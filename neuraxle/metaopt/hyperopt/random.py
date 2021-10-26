from abc import ABC
from neuraxle.hyperparams.space import HyperparameterSamples


class BaseHyperparameterSelectionStrategy(ABC):
    @abstractmethod
    def find_next_best_hyperparams(self, auto_ml_container: 'AutoMLContainer') -> HyperparameterSamples:
        """
        Find the next best hyperparams using previous trials.

        :param auto_ml_container: trials data container
        :return: next best hyperparams
        """
        raise NotImplementedError()
