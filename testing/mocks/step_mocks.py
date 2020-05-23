from neuraxle.base import BaseStep, TruncableSteps, MetaStep, BaseTransformer
from neuraxle.hyperparams.distributions import LogUniform, Quantized, RandInt, Boolean
from neuraxle.hyperparams.space import HyperparameterSpace, HyperparameterSamples

HYPERPARAMETERS_SPACE = HyperparameterSpace({
    'learning_rate': LogUniform(0.0001, 0.1),
    'l2_weight_reg': LogUniform(0.0001, 0.1),
    'momentum': LogUniform(0.01, 1.0),
    'hidden_size': Quantized(LogUniform(16, 512)),
    'num_layers': RandInt(1, 4),
    'num_lstm_layers': RandInt(1, 2),
    'use_xavier_init': Boolean(),
    'use_max_pool_else_avg_pool': Boolean(),
    'dropout_drop_proba': LogUniform(0.3, 0.7)
})

HYPERPARAMETERS = HyperparameterSamples({
    'learning_rate': 0.1,
    'l2_weight_reg': 0.001,
    'hidden_size': 32,
    'num_layers': 3,
    'num_lstm_layers': 1,
    'use_xavier_init': True,
    'use_max_pool_else_avg_pool': True,
    'dropout_drop_proba': 0.5,
    'momentum': 0.1
})

AN_INPUT = "I am an input"
AN_EXPECTED_OUTPUT = "I am an expected output"


class SomeStep(BaseTransformer):
    def __init__(self, hyperparams_space: HyperparameterSpace = None, output=AN_EXPECTED_OUTPUT):
        super().__init__(hyperparams=None, hyperparams_space=hyperparams_space)
        self.output = output

    def transform(self, data_inputs):
        return [self.output] * len(data_inputs)


class SomeStepWithHyperparams(BaseStep):
    def __init__(self):
        super().__init__(
            hyperparams=HYPERPARAMETERS,
            hyperparams_space=HYPERPARAMETERS_SPACE,
            name="MockStep"
        )

    def transform(self, data_inputs):
        pass

    def fit(self, data_inputs, expected_outputs=None):
        pass


class SomeMetaStepWithHyperparams(MetaStep):
    def __init__(self):
        MetaStep.__init__(self, wrapped=SomeStepWithHyperparams())

    def transform(self, data_inputs):
        pass

    def fit(self, data_inputs, expected_outputs=None):
        pass


class SomeTruncableStep(TruncableSteps):
    def __init__(self):
        TruncableSteps.__init__(self,
                                hyperparams=HYPERPARAMETERS,
                                hyperparams_space=HYPERPARAMETERS_SPACE,
                                steps_as_tuple=(SomeStepWithHyperparams(), SomeStepWithHyperparams())
                                )

    def transform(self, data_inputs):
        pass

    def fit(self, data_inputs, expected_outputs=None):
        pass


class SomeSplitStep(BaseStep):
    def transform(self, data_inputs):
        pass
