import abc
from typing import Dict

import fauvqe


class Restorable(abc.ABC):
    @abc.abstractmethod
    def to_json_dict(self) -> Dict:
        raise NotImplementedError()  # pragma: no cover

    @classmethod
    @abc.abstractmethod
    def from_json_dict(cls, params: Dict):
        raise NotImplementedError()  # pragma: no cover

    @staticmethod
    def restore(dct: Dict):
        available_modules = {
            export[0]: export[1]
            for export in fauvqe.__dict__.items()
            if not (export[0].startswith("__") and export[0].endswith("__"))
        }

        restore_type_name = dct["type"]

        if restore_type_name not in available_modules.keys():
            raise NotImplementedError("Unknown type {}".format(restore_type_name))

        for param, value in dct["constructor_params"].items():
            if isinstance(value, dict) and "type" in value:
                dct["constructor_params"][param] = Restorable.restore(value)

        restore_type = available_modules[restore_type_name]

        if issubclass(restore_type, Restorable):
            # Use restore method
            return restore_type.from_json_dict(dct)

        # Attempt to create instance directly
        return restore_type(**dct["constructor_params"])
