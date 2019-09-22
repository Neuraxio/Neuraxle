from neuraxle.base import DataContainer


def test_data_container_should_get_first_n_items_data_container():
    data_container = DataContainer(
        current_ids=[0, 1, 2, 3],
        data_inputs=[4, 5, 6, 7],
        expected_outputs=[8, 9, 10, 11],
    )

    sliced_data_container = data_container[:2]

    assert sliced_data_container.current_ids == [0, 1]
    assert sliced_data_container.data_inputs == [4, 5]
    assert sliced_data_container.expected_outputs == [8, 9]


def test_data_container_should_get_last_n_items_data_container():
    data_container = DataContainer(
        current_ids=[0, 1, 2, 3],
        data_inputs=[4, 5, 6, 7],
        expected_outputs=[8, 9, 10, 11],
    )

    sliced_data_container = data_container[2:]

    assert sliced_data_container.current_ids == [2, 3]
    assert sliced_data_container.data_inputs == [6, 7]
    assert sliced_data_container.expected_outputs == [10, 11]


def test_data_container_should_get_range_of_items_data_container():
    data_container = DataContainer(
        current_ids=[0, 1, 2, 3],
        data_inputs=[4, 5, 6, 7],
        expected_outputs=[8, 9, 10, 11],
    )

    sliced_data_container = data_container[1:3]

    assert sliced_data_container.current_ids == [1, 2]
    assert sliced_data_container.data_inputs == [5, 6]
    assert sliced_data_container.expected_outputs == [9, 10]


def test_data_container_should_get_single_item_data():
    data_container = DataContainer(
        current_ids=[0, 1, 2, 3],
        data_inputs=[4, 5, 6, 7],
        expected_outputs=[8, 9, 10, 11],
    )

    current_id, data_input, expected_output = data_container[1]

    assert current_id == 1
    assert data_input == 5
    assert expected_output == 9
