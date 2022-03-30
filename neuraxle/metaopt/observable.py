"""
Neuraxle's Observable Classes
===================================================================
Base observable classes, implementing the Observer pattern.
Some of them are used to track the evolution of the optimization process.

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
from typing import Set, TypeVar, Generic


BaseDataclassT = TypeVar('BaseDataclassT')


class _ObservableRepo(Generic[BaseDataclassT]):
    """
    This class is used to implement the Observer design pattern.
    The _Observable class is a subject that is being observed by the _Observer class.
    The type T is the type of the value that will be observed.

    There are methods that the observer can define to send the notification:
    - notify_next(value: T),
    - notify_complete(value: T).

    The _Observable class is a generic class, so it can be used with any type T.

    It is possible to subscribe and unsubscribe observers.
    A subscription is a call to the subscribe method of the _Observable class.
    A unsubscription is a call to the unsubscribe method of the _Observable class.
    Thus, the subsibers receive the notifications of the _Observable class.

    A notification is a call to one of the notify_* methods of the _Observable class.
    A notification is a call to one of the update_* methods of the _Observer class.
    """

    def __init__(self):
        self._observers: Set[_ObserverOfRepo[BaseDataclassT]] = set()

    def subscribe(self, observer: '_ObserverOfRepo[BaseDataclassT]'):
        self._observers.add(observer)

    def unsubscribe(self, observer: '_ObserverOfRepo[BaseDataclassT]'):
        self._observers.discard(observer)

    def notify_next(self, value: BaseDataclassT):
        for observer in self._observers:
            observer.update_next(value)

    def notify_complete(self, value: BaseDataclassT):
        for observer in self._observers:
            observer.update_complete(value)


class _ObserverOfRepo(Generic[BaseDataclassT]):
    """
    This class is used to implement the Observer design pattern.
    The _Observer class is an observer that is being notified by the _Observable class.
    The type T is the type of the value that will be observed.

    Upon receiving a notification, the _Observer class can define these methods:
    - update_next(value: T),
    - update_complete(value: T)

    These methods are called by the _Observable class observing the observer.
    """
    @abstractmethod
    def update_next(self, value: BaseDataclassT):
        pass

    def update_complete(self, value: BaseDataclassT):
        pass
