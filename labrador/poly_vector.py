from .poly import poly
from . import params
from typing import List

class poly_vector:
    def __init__(self, data: List[poly] = None, size = params.N):
        if data is None:
            self.data = [poly() for _ in range(size)]
            return
        self.data = data

    @staticmethod
    def random_vector(size = params.N) -> 'poly_vector':
        return poly_vector([poly.random_poly() for _ in range(size)])
    
    @staticmethod
    def test_vector(size = params.N) -> 'poly_vector':
        return poly_vector([poly.test_poly() for _ in range(size)])
    
    def conjugate(self) -> 'poly_vector':
        return poly_vector([p.conjugate() for p in self.data])

    def get_coeff_vector(self) -> List[int]:
        return [coeff for polynomial in self.data for coeff in polynomial.data]

    # inner product
    @staticmethod
    def inner_product(first, second) -> poly:
        return sum([a * b for (a, b) in zip(first.data, second.data)], poly())

    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value

    def __mul__(self, other) -> 'poly_vector':
        if isinstance(other, int):
            return poly_vector([p * other for p in self.data])
        elif isinstance(other, poly):
            return poly_vector([p * other for p in self.data])
        else:
            raise Exception("Undefined operation")

    def __add__(self, other) -> 'poly_vector':
        return poly_vector([a + b for (a, b) in zip(self.data, other.data)])

    def __str__(self) -> str:
        str = "[\n"
        for p in self.data:
            str += f"{p},\n"
        str += "]"
        return str

    def __repr__(self) -> str:
        return str(self)