"""
Neuraxle's Observable Classes
===================================================================
Base observable classes

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
from abc import abstractmethod
from typing import TypeVar, Generic

T = TypeVar('T')


class _Observable(Generic[T]):
    def __init__(self):
        self._observers = set()

    def subscribe(self, observer: '_Observer[T]'):
        self._observers.add(observer)

    def unsubscribe(self, observer: '_Observer[T]'):
        self._observers.discard(observer)

    def on_next(self, value: T):
        for observer in self._observers:
            observer.on_next(value)

    def on_complete(self, value: T):
        for observer in self._observers:
            observer.on_complete(value)

    def on_error(self, value: Exception):
        for observer in self._observers:
            observer.on_error(value)


class _Observer(Generic[T]):
    @abstractmethod
    def on_next(self, value: T):
        pass

    def on_complete(self, value: T):
        pass

    def on_error(self, value: Exception):
        pass
