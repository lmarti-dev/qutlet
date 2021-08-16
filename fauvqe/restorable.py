import abc
from typing import Dict
import numpy as np


class Restorable(abc.ABC):
    @abc.abstractmethod
    def to_json_dict(self) -> Dict:
        raise NotImplementedError()  # pragma: no cover

    @classmethod
    @abc.abstractmethod
    def from_json_dict(cls, params: Dict):
        raise NotImplementedError()  # pragma: no cover

    def __eq__(self, other): 
        if not isinstance(other, self.__class__):
            # don't attempt to compare against unrelated types
            return False

        #Most general: avoid to define Attributes
        temp_bools = []
        for key in self.__dict__.keys():
            #print(key)
            if isinstance(getattr(self, key), np.ndarray):
                if isinstance(getattr(other, key), np.ndarray):
                    if len(getattr(self, key)) != 0 and len(getattr(other, key)) != 0:
                        #print("key: \t{}\n(getattr(self, key): \n{}\ngetattr(other, key): \n{}\n".format(key, getattr(self, key), getattr(other, key)))
                        temp_bools.append((getattr(self, key) == getattr(other, key)).all())
                    else:
                        temp_bools.append(len(getattr(self, key)) == len(getattr(other, key))) 
                else:
                    return False
            else:
                if key != 'simulator':
                    #print("key: \t{}\ngetattr(self, key): \n{}\ngetattr(other, key): \n{}\n".format(key, getattr(self, key), getattr(other, key)))
                    temp_bools.append(getattr(self, key) == getattr(other, key))
                else:
                    temp_bools.append(getattr(self, key).__class__ == getattr(other, key).__class__)
        #print(temp_bools)
        return all(temp_bools)