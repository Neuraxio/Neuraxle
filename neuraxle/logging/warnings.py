"""
Neuraxle's Deprecation Warnings
====================================
Code evolves through time. When updating Neuraxle, you may find
that some old arguments you were using or some classes you were
using changed. Warnings will be printed using the methods here.

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

import warnings


SILENCE_DEPRECATION_WARNING = False


def silence_all_deprecation_warnings():
    """
    Turn off all of the neuraxle.logging.warnings for deprecations.
    """
    global SILENCE_DEPRECATION_WARNING
    SILENCE_DEPRECATION_WARNING = True


def warn_deprecated_class(self, replacement_class: type = None):
    global SILENCE_DEPRECATION_WARNING
    if not SILENCE_DEPRECATION_WARNING and replacement_class is not None:
        warnings.warn(
            _deprecated_class_msg(replacement_class) + _deact_msg_instructions()
        )
    return self


class RaiseDeprecatedClass:
    def __init__(self, replacement_class: type = None) -> None:
        raise_deprecated_class(replacement_class)


def raise_deprecated_class(replacement_class: type = None):
    raise DeprecationWarning(_deprecated_class_msg(replacement_class))


def _deprecated_class_msg(self, replacement_class: type = None) -> str:
    return (
        f"The class `{self.__class__.__name__}` is deprecated."
        f" Please consider using the class `{replacement_class.__name__}` instead. " if (
            hasattr(replacement_class, "__name__") and replacement_class.__name__ is not None) else ""
    )


def warn_deprecated_arg(self, arg_name, default_value, value, replacement_argument_name, replacement_class: type = None):
    global SILENCE_DEPRECATION_WARNING
    if not SILENCE_DEPRECATION_WARNING and default_value != value:
        if type(replacement_class) is type:
            replacement_class = replacement_class.__name__
        warnings.warn(
            f"Argument `{arg_name}={value}` for class `{self.__class__.__name__}` is deprecated. "
            f"Please consider using `{replacement_argument_name}` "
            f"or the class `{replacement_class}` " if replacement_class is not None else ""
            f"instead. "
            f"{_deact_msg_instructions()}"
        )
    return self


def _deact_msg_instructions() -> str:
    return (
        " If you want to disable these warnings,"
        " call `neuraxle.logging.warnings.silence_all_deprecation_warnings()`."
    )
