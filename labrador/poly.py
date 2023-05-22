from . import params
import secrets
import numpy as np
from typing import List

class poly:
    def __init__(self, data: List[int] = None):
        if data is None:
            self.data = np.zeros(params.DEGREE, dtype=int)
            return
        assert len(data) == params.DEGREE, f"at {data}"
        self.data = [d % params.Q for d in data]
    
    @staticmethod
    def random_poly() -> 'poly':
        return poly([secrets.randbelow(params.Q) for _ in range(params.DEGREE)])
    
    @staticmethod
    def test_poly() -> 'poly':
        return poly([i + 1 for i in range(params.DEGREE)])

    def conjugate(self) -> 'poly':
        res = [0 for _ in range(params.DEGREE)]
        res[0] = self.data[0]
        i = params.DEGREE - 1
        for pi in self.data[1:]:
            res[i] = params.Q - pi
            i -= 1
        return poly(res)

    def __add__(self, other) -> 'poly':
        coeffs = [(a_i + b_i) % params.Q for (a_i, b_i) in zip(self.data, other.data)]
        return poly(coeffs)
    
    def __mul__(self, other) -> 'poly':
        # multiplication with scalar
        if isinstance(other, int):
            coeffs = [(other * coeff) % params.Q for coeff in self.data]
            return poly(coeffs)
        if isinstance(other, float):
            coeffs = [int((other * coeff)) % params.Q for coeff in self.data]
            return poly(coeffs)
        elif isinstance(other, poly):
            # poly multiplying and reducing (X^n + 1)
            c = np.polymul(self.data, other.data)
            j = 0
            for i in range(params.DEGREE, len(c)):
                c[j] -= c[i]
                c[j] %= params.Q
                j += 1
            c = c[:params.DEGREE]
            if len(c) != params.DEGREE:
                c += [0] * (params.DEGREE - len(c))
            return poly()
        else:
            raise Exception("Undefined operation")
        
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value

    def __str__(self) -> str:
        s = "<"
        for i in range(len(self.data) - 1):
            s += f"{self.data[i]} X^{i} + "
        s += f"{self.data[-1]}*X^{len(self.data) - 1}>"
        return s

    def __repr__(self) -> str:
        return str(self)