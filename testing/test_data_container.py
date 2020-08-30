import numpy as np
import pytest

from neuraxle.data_container import DataContainer, ListDataContainer, AbsentValuesNullObject


def test_data_container_iter_method_should_iterate_with_none_current_ids():
    data_container = DataContainer(data_inputs=np.array(list(range(100))),
                                   expected_outputs=np.array(list(range(100, 200)))).set_current_ids(None)

    for i, (current_id, data_input, expected_outputs) in enumerate(data_container):
        assert current_id is None
        assert data_input == i
        assert expected_outputs == i + 100


def test_data_container_iter_method_should_iterate_with_none_expected_outputs():
    data_container = DataContainer(current_ids=[str(i) for i in range(100)], data_inputs=np.array(list(range(100))),
                                   expected_outputs=None)

    for i, (current_id, data_input, expected_outputs) in enumerate(data_container):
        assert data_input == i
        assert expected_outputs is None


def test_data_container_len_method_should_return_data_inputs_len():
    data_container = DataContainer(current_ids=None, data_inputs=np.array(list(range(100))), expected_outputs=None)

    assert len(data_container) == 100


def test_data_container_should_iterate_through_batches_using_convolved():
    data_container = DataContainer(current_ids=[str(i) for i in range(100)], data_inputs=np.array(list(range(100))),
                                   expected_outputs=np.array(list(range(100, 200))))

    batches = []
    for b in data_container.convolved_1d(stride=10, kernel_size=10):
        batches.append(b)

    for i, batch in enumerate(batches):
        assert np.array_equal(np.array(batch.data_inputs), np.array(list(range(i * 10, (i * 10) + 10))))
        assert np.array_equal(
            np.array(batch.expected_outputs),
            np.array(list(range((i * 10) + 100, (i * 10) + 100 + 10)))
        )


def test_list_data_container_concat():
    # Given
    data_container = ListDataContainer(
        current_ids=[str(i) for i in range(100)],
        data_inputs=np.array(list(range(100))),
        expected_outputs=np.array(list(range(100, 200)))
    )

    # When
    data_container.concat(DataContainer(
        current_ids=[str(i) for i in range(100, 200)],
        data_inputs=np.array(list(range(100, 200))),
        expected_outputs=np.array(list(range(200, 300)))
    ))

    # Then
    assert np.array_equal(np.array(data_container.current_ids), np.array(list(range(0, 200))).astype(np.str))

    expected_data_inputs = np.array(list(range(0, 200))).astype(np.int)
    actual_data_inputs = np.array(data_container.data_inputs).astype(np.int)
    assert np.array_equal(actual_data_inputs, expected_data_inputs)

    expected_expected_outputs = np.array(list(range(100, 300))).astype(np.int)
    assert np.array_equal(np.array(data_container.expected_outputs).astype(np.int), expected_expected_outputs)


@pytest.mark.parametrize('batch_size,include_incomplete_pass,default_value,expected_data_containers', [
    (3, False, None, [
        DataContainer(current_ids=[0, 1, 2], data_inputs=[0, 1, 2], expected_outputs=[10, 11, 12]),
        DataContainer(current_ids=[3, 4, 5], data_inputs=[3, 4, 5], expected_outputs=[13, 14, 15]),
        DataContainer(current_ids=[6, 7, 8], data_inputs=[6, 7, 8], expected_outputs=[16, 17, 18]),
    ]),
    (3, True, 0, [
        DataContainer(current_ids=[0, 1, 2], data_inputs=[0, 1, 2], expected_outputs=[10, 11, 12]),
        DataContainer(current_ids=[3, 4, 5], data_inputs=[3, 4, 5], expected_outputs=[13, 14, 15]),
        DataContainer(current_ids=[6, 7, 8], data_inputs=[6, 7, 8], expected_outputs=[16, 17, 18]),
        DataContainer(current_ids=[0, 1, 2], data_inputs=[9, 0, 0], expected_outputs=[19, 0, 0])
    ]),
    (3, True, AbsentValuesNullObject(), [
        DataContainer(current_ids=[0, 1, 2], data_inputs=[0, 1, 2], expected_outputs=[10, 11, 12]),
        DataContainer(current_ids=[3, 4, 5], data_inputs=[3, 4, 5], expected_outputs=[13, 14, 15]),
        DataContainer(current_ids=[6, 7, 8], data_inputs=[6, 7, 8], expected_outputs=[16, 17, 18]),
        DataContainer(current_ids=[9], data_inputs=[9], expected_outputs=[19])
    ])
])
def test_convolved(batch_size, include_incomplete_pass, default_value, expected_data_containers):
    data_container = DataContainer(
        current_ids=[str(i) for i in range(10)],
        data_inputs=np.array(list(range(10))),
        expected_outputs=np.array(list(range(10, 20)))
    )

    # When
    data_containers = []
    for dc in data_container.convolved_1d(
        stride=batch_size,
        kernel_size=batch_size,
        include_incomplete_pass=include_incomplete_pass,
        default_value=default_value
    ):
        data_containers.append(dc)

    # Then
    assert len(expected_data_containers) == len(data_containers)
    for expected_data_container, actual_data_container in zip(expected_data_containers, data_containers):
        np.array_equal(expected_data_container.current_ids, actual_data_container.current_ids)
        np.array_equal(expected_data_container.data_inputs, actual_data_container.data_inputs)
        np.array_equal(expected_data_container.expected_outputs, actual_data_container.expected_outputs)
