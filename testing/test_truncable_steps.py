from neuraxle.pipeline import Pipeline
from testing.mocks.step_mocks import SomeSplitStep, SomeStep, SomeTruncableStep

EXPECTED_STR_OUTPUT = """main step : 
step name : SomeTruncableStep
hyperparameters : HyperparameterSamples([('MockStep__learning_rate', 0.1),
                       ('MockStep__l2_weight_reg', 0.001),
                       ('MockStep__hidden_size', 32),
                       ('MockStep__num_layers', 3),
                       ('MockStep__num_lstm_layers', 1),
                       ('MockStep__use_xavier_init', True),
                       ('MockStep__use_max_pool_else_avg_pool', True),
                       ('MockStep__dropout_drop_proba', 0.5),
                       ('MockStep__momentum', 0.1),
                       ('MockStep1__learning_rate', 0.1),
                       ('MockStep1__l2_weight_reg', 0.001),
                       ('MockStep1__hidden_size', 32),
                       ('MockStep1__num_layers', 3),
                       ('MockStep1__num_lstm_layers', 1),
                       ('MockStep1__use_xavier_init', True),
                       ('MockStep1__use_max_pool_else_avg_pool', True),
                       ('MockStep1__dropout_drop_proba', 0.5),
                       ('MockStep1__momentum', 0.1)])
intermediate steps : 
[('MockStep',
  step name : MockStep
hyperparameters : HyperparameterSamples([('learning_rate', 0.1),
                       ('l2_weight_reg', 0.001),
                       ('hidden_size', 32),
                       ('num_layers', 3),
                       ('num_lstm_layers', 1),
                       ('use_xavier_init', True),
                       ('use_max_pool_else_avg_pool', True),
                       ('dropout_drop_proba', 0.5),
                       ('momentum', 0.1)])),
 ('MockStep1',
  step name : MockStep1
hyperparameters : HyperparameterSamples([('learning_rate', 0.1),
                       ('l2_weight_reg', 0.001),
                       ('hidden_size', 32),
                       ('num_layers', 3),
                       ('num_lstm_layers', 1),
                       ('use_xavier_init', True),
                       ('use_max_pool_else_avg_pool', True),
                       ('dropout_drop_proba', 0.5),
                       ('momentum', 0.1)]))]"""

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
