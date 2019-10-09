from typing import Tuple, Any

from neuraxle.base import DataContainer, BaseStep, NonFittableMixin, ExecutionContext
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace
from neuraxle.pipeline import Pipeline
from neuraxle.steps.util import OutputTransformerMixin


class MultiplyBy2OutputTransformer(NonFittableMixin, OutputTransformerMixin, BaseStep):
    def __init__(
            self,
            hyperparams: HyperparameterSamples = None,
            hyperparams_space: HyperparameterSpace = None,
            name: str = None
    ):
        NonFittableMixin.__init__(self)
        BaseStep.__init__(self, hyperparams, hyperparams_space, name)
        OutputTransformerMixin.__init__(self)

    def transform(self, data_inputs) -> Tuple[Any, Any]:
        dis, eos = data_inputs

        new_dis = []
        new_eos = []
        for di, eo in zip(dis, eos):
            new_dis.append(di * 2)
            new_eos.append(eo * 2)

        return new_dis, new_eos


def test_output_transformer_should_zip_data_input_and_expected_output_in_the_transformed_output(tmpdir):
    pipeline = Pipeline([
        MultiplyBy2OutputTransformer()
    ])

    pipeline, new_data_container = pipeline.handle_fit_transform(
        DataContainer(
            current_ids=[0, 1, 2],
            data_inputs=[1, 2, 3],
            expected_outputs=[2, 3, 4]
        ),
        ExecutionContext.from_root(pipeline, tmpdir)
    )

    assert new_data_container.data_inputs == [2, 4, 6]
    assert new_data_container.expected_outputs == [4, 6, 8]
