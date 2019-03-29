from collections import OrderedDict
from typing import List, Tuple, Dict, Any, Union

NamedTupleList = List[Tuple[str, 'BaseStep']]

MaybeOrderedDict = Union[Dict, OrderedDict]
FlatHyperparams = MaybeOrderedDict[str, Any]  # Any except Dict.
DictHyperparams = MaybeOrderedDict[str, Union[Dict, Any]]
