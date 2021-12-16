"""
Pipeline Steps For Caching
=====================================

..
    Copyright 2021, Neuraxio Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

..
    Thanks to Umaneo Technologies Inc. for their contributions to this Machine Learning
    project, visit https://www.umaneo.com/ for more information on Umaneo Technologies Inc.

"""
import hashlib
import os
import shutil
from abc import ABC, abstractmethod
from typing import Any, Iterable, Tuple

import joblib
from neuraxle.base import BaseStep, ExecutionContext, MetaStep
from neuraxle.data_container import DataContainer


class BaseValueHasher(ABC):
    @abstractmethod
    def hash(self, data_input):
        raise NotImplementedError()


class StrValueHasher(BaseValueHasher):
    """
    Hashes the data inputs by converting them to a string
    and then using hashlib.blake2s (:func:`hashlib.blake2s`).
    """

    def hash(self, data_input):
        m = hashlib.blake2s()
        m.update(str.encode(str(data_input)))

        return m.hexdigest()


class TransformDIValueCachingWrapper(MetaStep):
    """
    Value caching wrapper wraps a step to cache the values.

    This class only operates on calls to transform.
    It does not check ids of the data inputs, nor the expected_outpts:
    only the data inputs are checked.

    It also flushes cache when the wrapped step is fitted.
    """

    def __init__(
            self,
            wrapped: BaseStep,
            cache_folder: str = None,
            value_hasher: BaseValueHasher = None,
    ):
        MetaStep.__init__(self, wrapped)
        self.value_hasher = value_hasher

        if self.value_hasher is None:
            self.value_hasher = StrValueHasher()

        if cache_folder is None:
            cache_folder = os.path.join(ExecutionContext().push(self).get_path(), "cache")
        self.value_caching_folder = cache_folder

    def _fit_transform_data_container(
        self, data_container: DataContainer, context: ExecutionContext
    ) -> Tuple[BaseStep, DataContainer]:
        """
        Fit transform data container.

        :param context: execution context
        :param data_container: the data container to transform
        :type data_container: neuraxle.data_container.DataContainer

        :return: tuple(fitted pipeline, data_container)
        """
        self.create_checkpoint_path()
        self.flush_cache()

        self.wrapped = self.wrapped.fit(data_container.data_inputs, data_container.expected_outputs)
        outputs = self._transform_with_cache(data_container)
        data_container.set_data_inputs(outputs)

        return self, data_container

    def _transform_data_container(self, data_container, context):
        """
        Transform data container.

        :param context: execution context
        :param data_container: the data container to transform
        :type data_container: neuraxle.data_container.DataContainer

        :return: transformed data container
        """
        self.create_checkpoint_path()
        outputs = self._transform_with_cache(data_container)
        data_container.set_data_inputs(outputs)

        return data_container

    def _hash_value(self, data_input):
        return self.value_hasher.hash(data_input)

    def _transform_with_cache(self, data_container: DataContainer) -> Iterable:
        """
        Transform data container using value caching.

        :param data_container: the data container to transform
        :type data_container: neuraxle.data_container.DataContainer

        :return: iterable
        """
        outputs = []
        for _, data_input, _ in data_container:
            if self.contains_cache_for(data_input):
                outputs.extend(self.read_cache(data_input))
            else:
                output = self.wrapped.transform([data_input])
                self.write_cache(data_input, output)
                outputs.extend(output)
        return outputs

    @abstractmethod
    def create_checkpoint_path(self) -> str:
        """
        Create checkpoint path only if it does not exists.

        :return: checkpoint path
        """
        raise NotImplementedError()

    @abstractmethod
    def flush_cache(self):
        """
        Flush all cached values
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def read_cache(self, data_input) -> Any:
        """
        Read cache for a given data input.

        :param data_input: data input to get cache for
        :type data_input: Any

        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def write_cache(self, data_input, output):
        """
        Write cache for a given data input and output.

        :param data_input: data input to write cache for
        :type data_input: Any

        :param output: output to write cache for
        :type output: Any

        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def contains_cache_for(self, data_input) -> bool:
        """
        Returns true if the data input transform output is cached.

        :param data_input: to get cache from
        :return: boolean to indicate if a cache is present for the given data input
        """
        raise NotImplementedError()

    @abstractmethod
    def get_cache_path_for(self, data_input) -> str:
        """
        Get the cache path for the given data input.

        :param data_input: data input to get cache path for
        :return: str for cache path
        """
        raise NotImplementedError()


class JoblibTransformDIValueCachingWrapper(TransformDIValueCachingWrapper):
    """
    Value Caching Wrapper class that caches the wrapped step transformed data inputs using python ``joblib`` library.

    It only saves and loads to cache the data inputs upon a handled transformation call.
    It also flushes cache when the wrapped step is fitted.

    .. seealso::
        :class:`TransformDIValueCachingWrapper`
    """

    def create_checkpoint_path(self) -> str:
        """
        Create checkpoint path only if it does not exists.
        """
        if not os.path.exists(self.value_caching_folder):
            os.makedirs(self.value_caching_folder)

        return self.value_caching_folder

    def flush_cache(self):
        shutil.rmtree(self.value_caching_folder)
        os.mkdir(self.value_caching_folder)

    def read_cache(self, data_input):
        return joblib.load(self.get_cache_path_for(data_input))

    def write_cache(self, data_input, output):
        return joblib.dump(output, self.get_cache_path_for(data_input))

    def contains_cache_for(self, data_input) -> bool:
        return os.path.exists(self.get_cache_path_for(data_input))

    def get_cache_path_for(self, data_input):
        hash_value = self._hash_value(data_input)
        return os.path.join(self.value_caching_folder, '{0}.joblib'.format(hash_value))
