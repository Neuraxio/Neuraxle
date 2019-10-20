from neuraxle.pipeline import Pipeline
from testing.mocks.step_mocks import SomeSplitStep, SomeStep, SomeTruncableStep

EXPECTED_STR_OUTPUT = """SomeTruncableStep
(
	SomeTruncableStep(
	name=SomeTruncableStep,
	hyperparameters=HyperparameterSamples([('learning_rate', 0.1),
                       ('l2_weight_reg', 0.001),
                       ('hidden_size', 32),
                       ('num_layers', 3),
                       ('num_lstm_layers', 1),
                       ('use_xavier_init', True),
                       ('use_max_pool_else_avg_pool', True),
                       ('dropout_drop_proba', 0.5),
                       ('momentum', 0.1)])
)(
		[('MockStep',
  SomeStepWithHyperparams(
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
)),
 ('MockStep1',
  SomeStepWithHyperparams(
	name=MockStep1,
	hyperparameters=HyperparameterSamples([('learning_rate', 0.1),
                       ('l2_weight_reg', 0.001),
                       ('hidden_size', 32),
                       ('num_layers', 3),
                       ('num_lstm_layers', 1),
                       ('use_xavier_init', True),
                       ('use_max_pool_else_avg_pool', True),
                       ('dropout_drop_proba', 0.5),
                       ('momentum', 0.1)])
))]	
)
)"""

def test_truncable_steps_should_split_by_type():
    pipeline = Pipeline([
        SomeStep(),
        SomeStep(),
        SomeSplitStep(),
        SomeStep(),
        SomeStep(),
        SomeSplitStep(),
        SomeStep(),
    ])

    sub_pipelines = pipeline.split(SomeSplitStep)

    assert 'SomeStep' in sub_pipelines[0]
    assert 'SomeStep1' in sub_pipelines[0]
    assert 'SomeSplitStep' in sub_pipelines[0]
    assert 'SomeStep2' in sub_pipelines[1]
    assert 'SomeStep3' in sub_pipelines[1]
    assert 'SomeSplitStep1' in sub_pipelines[1]
    assert 'SomeStep4' in sub_pipelines[2]


def test_set_train_should_set_train_to_false():
    pipeline = Pipeline([
        SomeStep(),
        SomeStep(),
        Pipeline([
            SomeStep(),
        ])
    ])

    pipeline.set_train(False)

    assert not pipeline.is_train
    assert not pipeline[0].is_train
    assert not pipeline[1].is_train
    assert not pipeline[2].is_train
    assert not pipeline[2][0].is_train


def test_set_train_should_set_train_to_true():
    pipeline = Pipeline([
        SomeStep(),
        SomeStep(),
        Pipeline([
            SomeStep(),
        ])
    ])

    assert pipeline.is_train
    assert pipeline[0].is_train
    assert pipeline[1].is_train
    assert pipeline[2].is_train
    assert pipeline[2][0].is_train


def test_basestep_representation_works_correctly():
    output = str(SomeTruncableStep())
    assert output == EXPECTED_STR_OUTPUT
