import pytest

from neuraxle.data_container import DataContainer, AbsentValuesNullObject
import numpy as np


class LoadableItem:
    def __init__(self):
        self.loaded = False

    def load(self) -> 'LoadableItem':
        self.loaded = True
        return self

    def is_loaded(self):
        return self.loaded


class SomeLazyLoadableCollection:
    def __init__(self, inner_list):
        self.inner_list = inner_list
        self.iterations = 0

    def __iter__(self):
        for item in self.inner_list:
            yield item.load()

    def __getitem__(self, item):
        return SomeLazyLoadableCollection([
            item.load()
            for item in self.inner_list[item]
        ])

    def __len__(self):
        return len(self.inner_list)


def test_data_container_minibatch_should_be_lazy_and_use_getitem_when_data_is_lazy_loadable():
    items = [LoadableItem() for _ in range(10)]
    data_inputs = SomeLazyLoadableCollection(items)
    expected_outputs = SomeLazyLoadableCollection([LoadableItem() for _ in range(10)])
    data_container = DataContainer(
        data_inputs=data_inputs,
        expected_outputs=expected_outputs
    )

    i = 0
    batch_size = 2
    for batch in data_container.minibatches(batch_size=batch_size):
        assert len(batch) == batch_size
        assert all(item.is_loaded() for item in data_inputs.inner_list[:(i * batch_size)])
        for y in range((i + 1) * batch_size, len(data_inputs)):
            assert not items[y].is_loaded()
        i += 1

@pytest.mark.parametrize('batch_size,include_incomplete_pass,default_value,expected_data_containers', [
    (3, False, None, [
        DataContainer(ids=[0, 1, 2], data_inputs=[0, 1, 2], expected_outputs=[10, 11, 12]),
        DataContainer(ids=[3, 4, 5], data_inputs=[3, 4, 5], expected_outputs=[13, 14, 15]),
        DataContainer(ids=[6, 7, 8], data_inputs=[6, 7, 8], expected_outputs=[16, 17, 18]),
    ]),
    (3, True, 0, [
        DataContainer(ids=[0, 1, 2], data_inputs=[0, 1, 2], expected_outputs=[10, 11, 12]),
        DataContainer(ids=[3, 4, 5], data_inputs=[3, 4, 5], expected_outputs=[13, 14, 15]),
        DataContainer(ids=[6, 7, 8], data_inputs=[6, 7, 8], expected_outputs=[16, 17, 18]),
        DataContainer(ids=[0, 1, 2], data_inputs=[9, 0, 0], expected_outputs=[19, 0, 0])
    ]),
    (3, True, AbsentValuesNullObject(), [
        DataContainer(ids=[0, 1, 2], data_inputs=[0, 1, 2], expected_outputs=[10, 11, 12]),
        DataContainer(ids=[3, 4, 5], data_inputs=[3, 4, 5], expected_outputs=[13, 14, 15]),
        DataContainer(ids=[6, 7, 8], data_inputs=[6, 7, 8], expected_outputs=[16, 17, 18]),
        DataContainer(ids=[9], data_inputs=[9], expected_outputs=[19])
    ])
])
def test_data_container_batching(batch_size, include_incomplete_pass, default_value, expected_data_containers):
    data_container = DataContainer(
        ids=[str(i) for i in range(10)],
        data_inputs=np.array(list(range(10))),
        expected_outputs=np.array(list(range(10, 20)))
    )

    # When
    data_containers = []
    for dc in data_container.minibatches(
        batch_size=batch_size,
        keep_incomplete_batch=include_incomplete_pass,
        default_value_data_inputs=default_value
    ):
        data_containers.append(dc)

    # Then
    assert len(expected_data_containers) == len(data_containers)
    for expected_data_container, actual_data_container in zip(expected_data_containers, data_containers):
        np.array_equal(expected_data_container.ids, actual_data_container.ids)
        np.array_equal(expected_data_container.data_inputs, actual_data_container.data_inputs)
        np.array_equal(expected_data_container.expected_outputs, actual_data_container.expected_outputs)
