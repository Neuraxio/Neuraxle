"""
Neuraxle's Checkpoint Classes
====================================
The checkpoint classes used by the checkpoint pipeline runner

..
    Copyright 2019, Neuraxio Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""

import os
import pickle
import warnings
from abc import abstractmethod, ABC
from enum import Enum
from typing import List, Tuple, Iterable, Any

from neuraxle.base import ResumableMixin, BaseStep, DataContainer, ListDataContainer, ExecutionContext, \
    ExecutionMode, NonTransformableMixin, NonFittableMixin


class BaseCheckpointer(ResumableMixin):
    """
    Base class to implement a step checkpoint or data container checkpoint.

    Checkpoint step uses many BaseCheckpointer to checkpoint both data container checkpoints, and step checkpoints.

    BaseCheckpointer has an execution mode so there could be different checkpoints for each execution mode (fit, fit_transform or transform).
    """

    def __init__(
            self,
            execution_mode: ExecutionMode
    ):
        self.execution_mode = execution_mode

    def is_for_execution_mode(self, execution_mode) -> bool:
        """
        Returns true if the checkpointer should be used with the given execution mode.

        :param execution_mode: execution mode (fit, fit_transform, or transform)
        :return: if the checkpointer should be used
        :rtype: bool
        """
        if execution_mode == ExecutionMode.FIT:
            return self.execution_mode == ExecutionMode.FIT or \
                   self.execution_mode == ExecutionMode.FIT_OR_FIT_TRANSFORM

        if execution_mode == ExecutionMode.FIT_TRANSFORM:
            return self.execution_mode == ExecutionMode.FIT_TRANSFORM or \
                   self.execution_mode == ExecutionMode.FIT_OR_FIT_TRANSFORM

        if execution_mode == ExecutionMode.TRANSFORM:
            return self.execution_mode == ExecutionMode.TRANSFORM

    @abstractmethod
    def save_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Save the data container or fitted step checkpoint with the given data container, and context.
        Returns the data container checkpoint, or latest data container.

        :param data_container: data container to save data container or fitted steps
        :param context: context to save data container or fitted steps
        :return: saved data container
        :rtype: DataContainer
        """
        raise NotImplementedError()

    @abstractmethod
    def read_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Read the data container checkpoint with self.data_checkpointer.
        Returns a new data container loaded with all the data inputs,
        and expected outputs for each current id in the given data container.

        :param data_container: data container containing the current_ids to read checkpoint for
        :param context: context to read checkpoint for
        :return: the data container checkpoint
        :rtype: DataContainer
        """
        raise NotImplementedError()


class StepSavingCheckpointer(BaseCheckpointer):
    """
    StepCheckpointer is used by the Checkpoint step to save the fitted steps contained in the context of type ExecutionContext.

    By default, StepCheckpointer saves the fitted steps when the execution mode is either FIT, or FIT_TRANSFORM :
    ```
    StepCheckpointer(ExecutionMode.FIT_OR_FIT_TRANSFORM)

    # is equivalent to :

    StepCheckpointer()
    ```
    """

    def __init__(
            self,
            execution_mode: ExecutionMode = ExecutionMode.FIT_OR_FIT_TRANSFORM,
    ):
        BaseCheckpointer.__init__(self, execution_mode=execution_mode)

    def save_checkpoint(
            self,
            data_container: DataContainer,
            context: ExecutionContext
    ) -> DataContainer:
        # TODO: save the context by execution mode AND data container ids / summary
        context.save_all_unsaved()
        return data_container

    def should_resume(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        # TODO: change this when we support multiple execution modes and data container ids / summary
        return True


class BaseDataSaver(ABC):
    @abstractmethod
    def get_checkpoint_path(self, checkpoint_path: str, current_id) -> str:
        """
        Returns the checkpoint path for a data input id

        :param checkpoint_path:
        :param current_id:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def save_checkpoint_for_current_id(self, checkpoint_path: str, current_id, data_input, expected_output) -> bool:
        """
        Save the given current id, data input, and expected output.

        :param checkpoint_path:
        :param current_id:
        :param data_input:
        :param expected_output:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def read_checkpoint_for_current_id(self, checkpoint_path: str, current_id) -> Tuple:
        """
        Read the data inputs, and expected outputs for the given current id.

        :param checkpoint_path:
        :param current_id:
        :return: tuple(current_id, checkpoint_data_input, checkpoint_expected_output)
        :rtype: tuple(str, Iterable, Iterable)
        """
        raise NotImplementedError()


class PickleDataSaver(BaseDataSaver):
    """
    PickleDataContainerSaver is used by DataContainerCheckpointer so save,
    and load the checkpoint files for all of the current ids inside a data container.
    """

    def __init__(self, execution_mode: ExecutionMode):
        if execution_mode == ExecutionMode.FIT:
            self.checkpoint_file_name_suffix = ExecutionMode.FIT.value

        if execution_mode == ExecutionMode.TRANSFORM:
            self.checkpoint_file_name_suffix = ExecutionMode.TRANSFORM.value

        if execution_mode == ExecutionMode.FIT_TRANSFORM:
            self.checkpoint_file_name_suffix = ExecutionMode.FIT_TRANSFORM.value

    def save_checkpoint_for_current_id(self, checkpoint_path: str, current_id, data_input, expected_output) -> bool:
        """
        Save the given current id, data input, and expected output using pickle.dump.

        :param checkpoint_path:
        :param current_id:
        :param data_input:
        :param expected_output:
        :return:
        """
        with open(self.get_checkpoint_path(checkpoint_path, current_id), 'wb') as file:
            pickle.dump((current_id, data_input, expected_output), file)

        return True

    def read_checkpoint_for_current_id(self, checkpoint_path: str, current_id) -> Tuple[str, Iterable, Iterable]:
        """
        Read the data inputs, and expected outputs for the given current id using pickle.load.

        :param checkpoint_path:
        :param current_id:
        :return: tuple(current_id, checkpoint_data_input, checkpoint_expected_output)
        :rtype: tuple(str, Iterable, Iterable)
        """
        with open(self.get_checkpoint_path(checkpoint_path, current_id), 'rb') as file:
            (checkpoint_current_id, checkpoint_data_input, checkpoint_expected_output) = \
                pickle.load(file)
            return checkpoint_current_id, checkpoint_data_input, checkpoint_expected_output

    def get_checkpoint_path(self, checkpoint_path: str, current_id) -> str:
        """
        Get the checkpoint file path for a data input id

        :param checkpoint_path:
        :param current_id:
        :return: path
        :rtype: str
        """
        return os.path.join(
            checkpoint_path,
            '{0}_{1}.pickle'.format(current_id, self.checkpoint_file_name_suffix)
        )


class DataContainerCheckpointer(BaseCheckpointer):
    """
    Class used to checkpoint a data container for it's given execution mode.
    This class uses a BaseDataContainerSaver to actually save the data checkpoint.

    DataContainerCheckpointer loops through a data container to save, or load the data inputs,
    and expected outputs using the current_ids inside the data container.
    """

    def __init__(
            self,
            data_container_saver: BaseDataSaver,
            execution_mode: ExecutionMode
    ):
        BaseCheckpointer.__init__(self, execution_mode)

        self.execution_mode: ExecutionMode = execution_mode
        self.data_checkpointer: BaseDataSaver = data_container_saver

    def save_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Save the given data container with self.data_checkpointer.

        :param data_container: data container to checkpoint
        :param context: context to checkpoint from
        :return:
        """
        context.mkdir()
        for current_id, data_input, expected_output in data_container:
            self.data_checkpointer.save_checkpoint_for_current_id(
                context.get_path(),
                current_id,
                data_input,
                expected_output
            )

        return data_container

    def read_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Read the data container checkpoint with self.data_checkpointer.
        Returns a new data container loaded with all the data inputs,
        and expected outputs for each current id in the given data container.

        :param data_container: data container containing the current_ids to read checkpoint for
        :param context: context to read checkpoint for
        :return:
        """
        list_data_container = ListDataContainer.empty()

        for current_id, data_input, expected_output in data_container:
            checkpoint_current_id, checkpoint_data_input, checkpoint_expected_outputs = \
                self.data_checkpointer.read_checkpoint_for_current_id(
                    context.get_path(),
                    current_id
                )

            list_data_container.append(
                checkpoint_current_id,
                checkpoint_data_input,
                checkpoint_expected_outputs
            )

        return list_data_container

    def should_resume(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        """
        Returns True if the checkpoint exists for each data inputs.

        :param context: execution context
        :param data_container:
        :return:
        """
        for current_id in data_container.current_ids:
            if not os.path.exists(
                    self.data_checkpointer.get_checkpoint_path(
                        context.get_path(),
                        current_id
                    )
            ):
                return False

        return True


class Checkpoint(NonFittableMixin, NonTransformableMixin, ResumableMixin, BaseStep):
    """
    Resumable Checkpoint Step to load, and save both data checkpoints, and step checkpoints.
    Checkpoint uses a list of step checkpointers(List[StepCheckpointer]), and data checkpointers(List[BaseCheckpointer]).

    Data Checkpoints save the state of the data container (transformed data inputs, and expected outputs)
    for the current execution mode (fit, fit_transform, or transform).

    Step Checkpoints save the state of the fitted steps before the checkpoint
    for the current execution mode (fit or fit_transform).

    By default(no arguments specified), the Checkpoint step saves the step checkpoints for any fit or fit transform,
    and saves a different data checkpoint with pickle data container checkpointers :

    ```
    Checkpoint(
        step_checkpointers=[
            StepCheckpointer(ExecutionMode.FIT_OR_FIT_TRANSFORM)
        ]
        data_checkpointers=[
            DataContainerCheckpointer.create_pickle_checkpointer(ExecutionMode.TRANSFORM),
            DataContainerCheckpointer.create_pickle_checkpointer(ExecutionMode.FIT),
            DataContainerCheckpointer.create_pickle_checkpointer(ExecutionMode.FIT_TRANSFORM),
        ]
    )

    # this is equivalent to :

    Checkpoint()
    ```
    """

    def __init__(
            self,
            step_checkpointer: StepSavingCheckpointer = None,
            data_checkpointers: List[BaseCheckpointer] = None
    ):
        BaseStep.__init__(self)
        if step_checkpointer is None:
            warnings.warn('Checkpoint Step Initialized without Step checkpointers: {0}.'.format(self.name))
        self.step_checkpointer = step_checkpointer

        if data_checkpointers is None:
            data_checkpointers = []

        self.data_checkpointers: List[BaseCheckpointer] = data_checkpointers

    def handle_fit(self, data_container: DataContainer, context: ExecutionContext) -> Tuple[
        'Checkpoint', DataContainer]:
        """
        Saves step, and data checkpointers for the FIT execution mode.

        :param data_container: data container for creating the data checkpoint
        :param context: context for creating the step checkpoint
        :return: saved data container
        :rtype: DataContainer
        """
        self.save_checkpoint(data_container, context)
        return self, data_container

    def handle_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Saves step, and data checkpointers for the TRANSORM execution mode.

        :param data_container: data container for creating the data checkpoint
        :param context: context for creating the step checkpoint
        :return: saved data container
        :rtype: DataContainer
        """
        return self.save_checkpoint(data_container, context)

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> Tuple[
        'Checkpoint', DataContainer]:
        """
        Saves step, and data checkpointers for the FIT_TRANSORM execution mode.

        :param data_container: data container for creating the data checkpoint
        :param context: context for creating the step checkpoint
        :return: saved data container
        :rtype: DataContainer
        """
        return self, self.save_checkpoint(data_container, context)

    def save_checkpoint(self, data_container: DataContainer, context: ExecutionContext):
        """
        Saves step, and data checkpointers for the current execution mode.

        :param data_container: data container for creating the data checkpoint
        :param context: context for creating the step checkpoint
        :return: saved data container
        :rtype: DataContainer
        """
        if self.step_checkpointer is not None and self.step_checkpointer.is_for_execution_mode(
                context.get_execution_mode()):
            self.step_checkpointer.save_checkpoint(data_container, context)

        for checkpointer in self.data_checkpointers:
            if checkpointer.is_for_execution_mode(context.get_execution_mode()):
                checkpointer.save_checkpoint(data_container, context)

        return data_container

    def read_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Read data checkpoint for the current execution mode using self.data_checkpointers.

        :param data_container: data container to load checkpoint from
        :param context: execution mode to load checkpoint from
        :return: loaded data container checkpoint
        :rtype: DataContainer
        """
        for checkpointer in self.data_checkpointers:
            if checkpointer.is_for_execution_mode(context.get_execution_mode()):
                data_container = checkpointer.read_checkpoint(data_container, context)

        return data_container

    def should_resume(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        """
        Returns True if all of the execution mode data checkpointers can be resumed.

        :param context: execution context
        :param data_container: data to resume
        :return: if we can resume the checkpoint
        :rtype: bool
        """
        for checkpointer in self.data_checkpointers:
            if checkpointer.is_for_execution_mode(context.get_execution_mode()):
                if not checkpointer.should_resume(data_container, context):
                    return False

        return True


class BigCheckpoint(Checkpoint):
    def __init__(self):
        Checkpoint.__init__(
            self,
            step_checkpointer=StepSavingCheckpointer(ExecutionMode.FIT_OR_FIT_TRANSFORM),
            data_checkpointers=[
                DataContainerCheckpointer(
                    PickleDataSaver(ExecutionMode.TRANSFORM),
                    ExecutionMode.TRANSFORM
                ),
                DataContainerCheckpointer(
                    PickleDataSaver(ExecutionMode.FIT),
                    ExecutionMode.FIT
                ),
                DataContainerCheckpointer(
                    PickleDataSaver(ExecutionMode.FIT_TRANSFORM),
                    ExecutionMode.FIT_TRANSFORM
                )
            ]
        )


class BaseMiniDataCheckpointer(ABC):
    @abstractmethod
    def save(self, path, ids, data):
        raise NotImplementedError()

    @abstractmethod
    def read(self, path, ids) -> Tuple:
        raise NotImplementedError()

class MiniDataCheckpointSuffix(Enum):
    DATA_INPUT = 'di'
    EXPECTED_OUTPUT = 'eo'

class PickleMiniDataCheckpointer(BaseMiniDataCheckpointer):
    def __init__(self, file_name_suffix):
        self.file_name_suffix = file_name_suffix

    def save(self, checkpoint_path: str, current_id, data) -> bool:
        """
        Save the given current id, data input, and expected output using pickle.dump.

        :param checkpoint_path:
        :param current_id:
        :param data:
        :return:
        """
        with open(self.get_checkpoint_path(checkpoint_path, current_id), 'wb') as file:
            pickle.dump((current_id, data), file)

        return True

    def read(self, checkpoint_path: str, current_id) -> Tuple[str, Any]:
        """
        Read the data inputs, and expected outputs for the given current id using pickle.load.

        :param checkpoint_path:
        :param current_id:
        :return: tuple(current_id, checkpoint_data_input, checkpoint_expected_output)
        :rtype: tuple(str, Iterable, Iterable)
        """
        with open(self.get_checkpoint_path(checkpoint_path, current_id), 'rb') as file:
            (checkpoint_current_id, checkpoint_data) = pickle.load(file)
            return checkpoint_current_id, checkpoint_data

    def get_checkpoint_path(self, checkpoint_path: str, current_id) -> str:
        """
        Get the checkpoint file path for a data input id

        :param checkpoint_path:
        :param current_id:
        :return: path
        :rtype: str
        """
        return os.path.join( checkpoint_path, '{0}_{1}.pickle'.format(current_id, self.file_name_suffix))


class MiniDataCheckpointerWrapper(BaseCheckpointer):
    def __init__(
            self,
            data_input_checkpointer: BaseMiniDataCheckpointer,
            expected_output_checkpointer: BaseMiniDataCheckpointer = None
    ):
        execution_mode = ExecutionMode.FIT_OR_FIT_TRANSFORM_OR_TRANSFORM # TODO: analyse if we need this or not ?
        BaseCheckpointer.__init__(self, execution_mode)

        self.data_input_checkpointer = data_input_checkpointer
        self.expected_output_checkpointer = expected_output_checkpointer

    def save_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        for current_id, data_input, _ in data_container:
            self.data_input_checkpointer.save(ids=current_id, data=data_input, path=context.get_path())

        for current_id, _, expected_output in data_container:
            self.expected_output_checkpointer.save(ids=current_id, data=expected_output, path=context.get_path())

        return data_container

    def read_checkpoint(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        for current_id, data_input, _ in data_container:
            checkpoint = self.data_input_checkpointer.read(ids=data_container.current_ids, path=context.get_path())
            data_container.set_data_inputs(checkpoint)

        expected_outputs = []
        for current_id, _, expected_output in data_container:
            checkpoint = self.expected_output_checkpointer.read(ids=current_id, path=context.get_path())
            expected_outputs.append(checkpoint)

        return data_container


class MiniCheckpoint(Checkpoint):
    def __init__(self):
        Checkpoint.__init__(
            self,
            step_checkpointer=StepSavingCheckpointer(),
            data_checkpointers=[
                MiniDataCheckpointerWrapper(
                    data_input_checkpointer=PickleMiniDataCheckpointer(MiniDataCheckpointSuffix.DATA_INPUT),
                    expected_output_checkpointer=PickleMiniDataCheckpointer(MiniDataCheckpointSuffix.EXPECTED_OUTPUT)
                )
            ]
        )
