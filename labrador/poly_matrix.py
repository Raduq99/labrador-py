from .poly import poly
from .poly_vector import poly_vector
from . import params
import numpy as np
from typing import List

class poly_matrix:
    def __init__(self, data: List[List[poly]] = None, rowSize = 0, columnSize = 0):
        if data is None:
            self.data = [[poly() for _ in range(columnSize)] for _ in range(rowSize)]
            return
        # check that m is a matrix (list of lists)
        assert(len(data[0]) >= 0)
        self.data = data
    
    @staticmethod
    def random_matrix(rowSize: int, columnSize: int):
        return poly_matrix([[poly.random_poly() for _ in range(columnSize)] for _ in range(rowSize)])
    
    @staticmethod
    def test_matrix(rowSize: int, columnSize: int):
        return poly_matrix([[poly.test_poly() for _ in range(columnSize)] for _ in range(rowSize)])

    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value

    def __mul__(self, other: poly_vector) -> poly_vector:
        if isinstance(other, poly_vector):
            da = np.matmul(self.data, other.data)
            return poly_vector(da)
        raise Exception("Unknown multiplication operands")

    def __str__(self) -> str:
        s = "[\n"
        for v in self.data:
            s += "["
            for vi in v:
                s += str(vi) + "\n"
            s += "]\n"
        s += "]"
        return s

    def __repr__(self) -> str:
        return str(self)