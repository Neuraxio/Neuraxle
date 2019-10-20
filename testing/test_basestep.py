from testing.mocks.step_mocks import SomeStepWithHyperparams

EXPECTED_STR_OUTPUT = """SomeStepWithHyperparams(
	name=MockStep,
	hyperparameters=HyperparameterSamples([('learning_rate', 0.1),
                       ('l2_weight_reg', 0.001),
                       ('hidden_size', 32),
                       ('num_layers', 3),
                       ('num_lstm_layers', 1),
                       ('use_xavier_init', True),
                       ('use_max_pool_else_avg_pool', True),
                       ('dropout_drop_proba', 0.5),
                       ('momentum', 0.1)])
)"""


def test_basestep_representation_works_correctly():
    output = str(SomeStepWithHyperparams())
    assert output == EXPECTED_STR_OUTPUT
