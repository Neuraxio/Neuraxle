from neuraxle.data_container import DataContainer


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


def test_data_container_convolve1d_should_be_lazy_and_use_getitem_when_data_is_lazy_loadable():
    data_inputs = SomeLazyLoadableCollection([LoadableItem() for _ in range(10)])
    expected_outputs = SomeLazyLoadableCollection([LoadableItem() for _ in range(10)])
    data_container = DataContainer(
        data_inputs=data_inputs,
        expected_outputs=expected_outputs
    )

    i = 0
    batch_size = 2
    for batch in data_container.batch(batch_size=batch_size):
        assert len(batch) == batch_size
        assert all(item.is_loaded() for item in data_inputs.inner_list[:i + 1])
        assert all(not item.is_loaded() for item in data_inputs.inner_list[i + 1:])
        i += 1
