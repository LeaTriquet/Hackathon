import random
from dataclasses import dataclass


@dataclass
class HyperParameter:
    name: str       # name of the HyperParameter
    _min: float     # min of the HyperParameter
    _max: float     # max of the HyperParameter
    
    def __call__(self) -> float:
        """ Get number btw _min and _max """
        return random.uniform(self._min, self._max)
    
    def __str__(self) -> str:
        """ Get name """
        if self._min == self._max:
            return self.name + f'{self._min}'
        return self.name + f'{self._min};{self._max}'