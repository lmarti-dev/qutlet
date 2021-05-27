from numbers import Complex
from typing import Any, Dict
import json


def decode_custom(dct: Dict) -> Any:
    if JSONEncoder.FLAG in dct:
        dtype = dct[JSONEncoder.FLAG]

        if dtype == JSONEncoder.FLAG_TYPE_COMPLEX:
            return complex(dct[JSONEncoder.COMPLEX_REAL], dct[JSONEncoder.COMPLEX_IMAG])

    return dct


class JSONEncoder(json.JSONEncoder):
    FLAG = "$$"
    FLAG_TYPE_COMPLEX = "c"
    COMPLEX_REAL = "r"
    COMPLEX_IMAG = "i"

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Complex):
            return {
                JSONEncoder.FLAG: JSONEncoder.FLAG_TYPE_COMPLEX,
                JSONEncoder.COMPLEX_REAL: obj.real,
                JSONEncoder.COMPLEX_IMAG: obj.imag,
            }

        return json.JSONEncoder.default(self, obj)


def dumps(*args, **kwargs) -> str:
    kwargs.setdefault("cls", JSONEncoder)
    return json.dumps(*args, **kwargs)


def dump(*args, **kwargs) -> None:
    kwargs.setdefault("cls", JSONEncoder)
    return json.dump(*args, **kwargs)


def load(*args, **kwargs) -> Any:
    kwargs.setdefault("object_hook", decode_custom)

    return json.load(*args, **kwargs)


def loads(*args, **kwargs) -> Any:
    kwargs.setdefault("object_hook", decode_custom)

    return json.loads(*args, **kwargs)
