from typing import Tuple, Any

from neuraxle.base import DataContainer
from neuraxle.pipeline import Pipeline
from neuraxle.steps.util import OutputTransformerMixin, OutputTransformerWrapper
from testing.test_pipeline import SomeStep


class MultiplyBy2OutputTransformer(OutputTransformerMixin, SomeStep):
    def transform_input_output(self, data_inputs, expected_outputs=None) -> Tuple[Any, Any]:
        dis = []
        eos = []
        for di, eo in zip(data_inputs, expected_outputs):
            dis.append(di*2)
            eos.append(eo*2)

        return dis, eos


def test_output_transformer_should_zip_data_input_and_expected_output_in_the_transformed_output():
    pipeline = Pipeline([
        OutputTransformerWrapper(
            MultiplyBy2OutputTransformer()
        )
    ])

    pipeline, new_data_container = pipeline.handle_fit_transform(
        DataContainer(current_ids=[0, 1, 2], data_inputs=[1, 2, 3], expected_outputs=[2, 3, 4])
    )

    assert new_data_container.data_inputs == [2, 4, 6]
    assert new_data_container.expected_outputs == [4, 6, 8]
