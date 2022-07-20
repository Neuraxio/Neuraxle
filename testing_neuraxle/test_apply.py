import pytest
from neuraxle.base import BaseStep, Identity, NonFittableMixin
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline
from neuraxle.steps.flow import ExpandDim
from neuraxle.steps.loop import ForEach
from neuraxle.steps.numpy import MultiplyByN
from neuraxle.steps.output_handlers import OutputTransformerWrapper


def test_apply_on_pipeline_with_positional_argument_should_call_method_on_each_steps():
    pipeline = Pipeline([MultiplyByN(1), MultiplyByN(1)])

    pipeline.apply('_set_hyperparams', hyperparams=HyperparameterSamples({
        'multiply_by': 2,
        'MultiplyByN__multiply_by': 3,
        'MultiplyByN1__multiply_by': 4
    }))

    assert pipeline.get_hyperparams()['multiply_by'] == 2
    assert pipeline['MultiplyByN'].get_hyperparams()['multiply_by'] == 3
    assert pipeline['MultiplyByN1'].get_hyperparams()['multiply_by'] == 4


def test_apply_method_on_pipeline_should_call_method_on_each_steps():
    pipeline = Pipeline([MultiplyByN(1), MultiplyByN(1)])

    pipeline.apply(
        lambda step: step._set_hyperparams(HyperparameterSamples({'multiply_by': 2}))
    )

    assert pipeline.get_hyperparams()['multiply_by'] == 2
    assert pipeline['MultiplyByN'].get_hyperparams()['multiply_by'] == 2
    assert pipeline['MultiplyByN1'].get_hyperparams()['multiply_by'] == 2


def test_apply_on_pipeline_with_meta_step_and_positional_argument():
    pipeline = Pipeline([OutputTransformerWrapper(MultiplyByN(1)), MultiplyByN(1)])

    pipeline.apply('_set_hyperparams', hyperparams=HyperparameterSamples({
        'multiply_by': 2,
        'OutputTransformerWrapper__multiply_by': 3,
        'OutputTransformerWrapper__MultiplyByN__multiply_by': 4,
        'MultiplyByN__multiply_by': 5
    }))

    assert pipeline.get_hyperparams()['multiply_by'] == 2
    assert pipeline['OutputTransformerWrapper'].get_hyperparams()['multiply_by'] == 3
    assert pipeline['OutputTransformerWrapper'].wrapped.get_hyperparams()['multiply_by'] == 4
    assert pipeline['MultiplyByN'].get_hyperparams()['multiply_by'] == 5


def test_apply_method_on_pipeline_with_meta_step_should_call_method_on_each_steps():
    pipeline = Pipeline([OutputTransformerWrapper(MultiplyByN(1)), MultiplyByN(1)])

    pipeline.apply(
        lambda step: step._set_hyperparams(HyperparameterSamples({'multiply_by': 2}))
    )

    assert pipeline.get_hyperparams()['multiply_by'] == 2
    assert pipeline['OutputTransformerWrapper'].get_hyperparams()['multiply_by'] == 2
    assert pipeline['OutputTransformerWrapper'].wrapped.get_hyperparams()['multiply_by'] == 2
    assert pipeline['MultiplyByN'].get_hyperparams()['multiply_by'] == 2


def test_has_children_mixin_apply_should_apply_method_to_direct_childrends():
    p = Pipeline([
        ('a', Identity()),
        ('b', Identity()),
        Pipeline([
            ('c', Identity()),
            ('d', Identity())
        ]),
    ])

    p.apply('_set_hyperparams', ra=None, hyperparams=HyperparameterSamples({
        'a__hp': 0,
        'b__hp': 1,
        'Pipeline__hp': 2
    }))

    assert p['a'].hyperparams.to_flat_dict()['hp'] == 0
    assert p['b'].hyperparams.to_flat_dict()['hp'] == 1
    assert p['Pipeline'].hyperparams.to_flat_dict()['hp'] == 2


def test_has_children_mixin_apply_should_apply_method_to_recursive_childrends():
    p = Pipeline([
        ('a', Identity()),
        ('b', Identity()),
        Pipeline([
            ('c', Identity()),
            ('d', Identity())
        ]),
    ])

    p.apply('_set_hyperparams', ra=None, hyperparams=HyperparameterSamples({
        'Pipeline__c__hp': 3,
        'Pipeline__d__hp': 4
    }))

    assert p['Pipeline']['c'].hyperparams.to_flat_dict()['hp'] == 3
    assert p['Pipeline']['d'].hyperparams.to_flat_dict()['hp'] == 4


def test_has_children_mixin_apply_should_return_recursive_dict_to_direct_childrends():
    p = Pipeline([
        ('a', Identity().set_hyperparams(HyperparameterSamples({'hp': 0}))),
        ('b', Identity().set_hyperparams(HyperparameterSamples({'hp': 1})))
    ])

    results = p.apply('_get_hyperparams', ra=None)

    assert results.to_flat_dict()['a__hp'] == 0
    assert results.to_flat_dict()['b__hp'] == 1


def test_has_children_mixin_apply_should_return_recursive_dict_to_recursive_childrends():
    p = Pipeline([
        Pipeline([
            ('c', Identity().set_hyperparams(HyperparameterSamples({'hp': 3}))),
            ('d', Identity().set_hyperparams(HyperparameterSamples({'hp': 4})))
        ]).set_hyperparams(HyperparameterSamples({'hp': 2})),
    ])

    results = p.apply('_get_hyperparams', ra=None)

    assert results['Pipeline__hp'] == 2
    assert results['Pipeline__c__hp'] == 3
    assert results['Pipeline__d__hp'] == 4


class Mutating2TransformsStep(NonFittableMixin, BaseStep):

    def __init__(self):
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)

    def transform(self, data_input, expected_outputs):
        raise AssertionError("Mutate failed. Should have entered in other methods.")

    def transform_a(self, data_input, expected_outputs):
        return ['a']

    def transform_b(self, data_input, expected_outputs):
        return ['b']


@pytest.mark.parametrize("step", [
    Mutating2TransformsStep(),
    ExpandDim(ForEach(Mutating2TransformsStep())),
    Pipeline([Mutating2TransformsStep()]),
])
def test_mutate(step: BaseStep):
    step = step.mutate(new_method="transform_a", method_to_assign_to="transform")
    _a = step.transform([0])
    assert _a == 'a'

    step = step.mutate(new_method="transform_b", method_to_assign_to="transform")
    _b = step.transform([0])
    assert _b == 'b'
