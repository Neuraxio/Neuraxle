from typing import Tuple, Any

from py._path.local import LocalPath

from neuraxle.base import BaseStep, ExecutionContext, ExecutionMode, TransformerStep
from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace
from neuraxle.pipeline import Pipeline
from neuraxle.steps.output_handlers import InputAndOutputTransformerMixin


class MultiplyBy2OutputTransformer(InputAndOutputTransformerMixin, TransformerStep):
    def __init__(
            self,
            hyperparams: HyperparameterSamples = None,
            hyperparams_space: HyperparameterSpace = None,
            name: str = None
    ):
        TransformerStep.__init__(self, hyperparams, hyperparams_space, name)
        InputAndOutputTransformerMixin.__init__(self)

    def transform(self, data_inputs) -> Tuple[Any, Any]:
        dis, eos = data_inputs

        new_dis = []
        new_eos = []
        for di, eo in zip(dis, eos):
            new_dis.append(di * 2)
            new_eos.append(eo * 2)

        return new_dis, new_eos


def test_output_transformer_should_zip_data_input_and_expected_output_in_the_transformed_output(tmpdir: LocalPath):
    pipeline = Pipeline([
        MultiplyBy2OutputTransformer()
    ])

    pipeline, new_data_container = pipeline.handle_fit_transform(
        DataContainer(data_inputs=[1, 2, 3], current_ids=[0, 1, 2], expected_outputs=[2, 3, 4]),
        ExecutionContext(tmpdir)
    )

    assert new_data_container.data_inputs == [2, 4, 6]
    assert new_data_container.expected_outputs == [4, 6, 8]
