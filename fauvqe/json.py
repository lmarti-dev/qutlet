from numbers import Complex
from typing import Any, Dict
import json

import cirq
import numpy as np
import sympy

import fauvqe
from fauvqe.restorable import Restorable


def decode_custom(dct: Dict) -> Any:
    if JSONEncoder.FLAG in dct:
        dtype = dct[JSONEncoder.FLAG]

        if dtype == JSONEncoder.FLAG_COMPLEX:
            return JSONEncoder.decode_complex(dct)

        if dtype == JSONEncoder.FLAG_RESTORABLE:
            return JSONEncoder.decode_restorable(dct)

        if dtype == JSONEncoder.FLAG_NUMPY:
            return JSONEncoder.decode_numpy_array(dct)

        if dtype == JSONEncoder.FLAG_SYMPY:
            return JSONEncoder.decode_sympy(dct)

        if dtype == JSONEncoder.FLAG_CIRQ:
            return JSONEncoder.decode_cirq(dct)

    return dct


class JSONEncoder(json.JSONEncoder):
    FLAG = "$$"

    FLAG_COMPLEX = "c"
    COMPLEX_REAL = "r"
    COMPLEX_IMAG = "i"

    FLAG_RESTORABLE = "restorable"
    RESTORABLE_DATA = "data"
    RESTORABLE_TYPE = "type"

    FLAG_NUMPY = "numpy"
    NUMPY_ARRAY = "array"
    NUMPY_DTYPE = "dtype"

    FLAG_SYMPY = "sympy"
    SYMPY_NAME = "name"

    FLAG_CIRQ = "cirq"
    CIRQ_CIRCUIT = "circuit"

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Complex):
            return JSONEncoder.encode_complex(obj)

        if isinstance(obj, Restorable):
            return JSONEncoder.encode_restorable(obj)

        if isinstance(obj, np.ndarray):
            return JSONEncoder.encode_numpy_array(obj)

        if isinstance(obj, sympy.Symbol):
            return JSONEncoder.encode_sympy(obj)

        if isinstance(obj, cirq.Circuit):
            return JSONEncoder.encode_cirq(obj)

        return json.JSONEncoder.default(self, obj)  # pragma: no cover (python internal)

    @staticmethod
    def encode_sympy(obj: sympy.Symbol) -> Dict:
        return {
            JSONEncoder.FLAG: JSONEncoder.FLAG_SYMPY,
            JSONEncoder.SYMPY_NAME: str(obj),
        }

    @staticmethod
    def decode_sympy(dct: Dict) -> sympy.Symbol:
        return sympy.Symbol(dct[JSONEncoder.SYMPY_NAME])

    @staticmethod
    def encode_cirq(obj: cirq.Circuit) -> Dict:
        return {
            JSONEncoder.FLAG: JSONEncoder.FLAG_CIRQ,
            JSONEncoder.CIRQ_CIRCUIT: cirq.to_json(obj, indent=None),
        }

    @staticmethod
    def decode_cirq(dct: Dict) -> cirq.Circuit:
        return cirq.read_json(json_text=dct[JSONEncoder.CIRQ_CIRCUIT])

    @staticmethod
    def encode_complex(obj: Complex) -> Dict:
        return {
            JSONEncoder.FLAG: JSONEncoder.FLAG_COMPLEX,
            JSONEncoder.COMPLEX_REAL: obj.real,
            JSONEncoder.COMPLEX_IMAG: obj.imag,
        }

    @staticmethod
    def decode_complex(dct: Dict) -> complex:
        return complex(dct[JSONEncoder.COMPLEX_REAL], dct[JSONEncoder.COMPLEX_IMAG])

    @staticmethod
    def encode_restorable(obj: Restorable) -> Dict:
        return {
            JSONEncoder.FLAG: JSONEncoder.FLAG_RESTORABLE,
            JSONEncoder.RESTORABLE_TYPE: type(obj).__name__,
            JSONEncoder.RESTORABLE_DATA: obj.to_json_dict(),
        }

    @staticmethod
    def decode_restorable(dct: Dict) -> Restorable:
        available_modules = {
            export[0]: export[1]
            for export in fauvqe.__dict__.items()
            if not (export[0].startswith("__") and export[0].endswith("__"))
        }

        restore_type_name = dct[JSONEncoder.RESTORABLE_TYPE]
        if restore_type_name not in available_modules.keys():
            raise NotImplementedError("Unknown type {}".format(restore_type_name))
        restore_type = available_modules[restore_type_name]

        if issubclass(restore_type, Restorable):
            # Use restore method
            return restore_type.from_json_dict(dct[JSONEncoder.RESTORABLE_DATA])

        # Attempt to create instance directly
        return restore_type(**dct[JSONEncoder.RESTORABLE_DATA]["constructor_params"])

    @staticmethod
    def encode_numpy_array(obj: np.ndarray) -> Dict:
        return {
            JSONEncoder.FLAG: JSONEncoder.FLAG_NUMPY,
            JSONEncoder.NUMPY_DTYPE: str(obj.dtype),
            JSONEncoder.NUMPY_ARRAY: obj.tolist(),
        }

    @staticmethod
    def decode_numpy_array(dct: Dict) -> np.ndarray:
        return np.array(dct[JSONEncoder.NUMPY_ARRAY], dtype=dct[JSONEncoder.NUMPY_DTYPE])


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
