from .poly_matrix import poly_matrix
from .poly_vector import poly_vector
from .poly import poly
from . import params
from typing import List
import numpy as np
import secrets

class Verifier:
    def __init__(self, 
                 beta: int = None, 
                 a: List[poly_matrix] = None, 
                 phi: List[poly_vector] = None, 
                 b: List[poly] = None, 
                 a_prime: List[poly_matrix] = None, 
                 phi_prime: List[poly_vector] = None, 
                 b_prime: List[poly] = None):
        # norm bound
        self.beta = beta
        # array of K poly matrices(r x r)
        self.a = a
        # array of K poly vectors(size n)
        self.phi = phi
        # array of K poly (size n)
        self.b = b
        # array of L poly matrices(r x r)
        self.a_prime = a_prime
        # array of L poly vectors(size n)
        self.phi_prime = phi_prime
        # array of L poly (size n)
        self.b_prime = b_prime
    
    #################### STEP 2 #######################

    @staticmethod
    def sample_projection() -> int:
        r = secrets.randbelow(5)
        if r < 2:
            return -1
        elif r < 4:
            return 1
        else:
            return 0

    def generate_projections(self):
        # r matrices of size 256 x nd
        def gen_pi():
            pi = []
            for _ in range(256):
                pi.append(poly_vector(data=[poly([Verifier.sample_projection() for _ in range(params.DEGREE)]) for _ in range(params.N)]))
            return pi
        
        return [gen_pi() for _ in range(params.R)]
    
    def check_projection(self, projection: List[int]) -> bool:
        norm_squared = sum(pow(p, 2, params.Q) for p in projection) % params.Q
        return norm_squared <= (self.beta**2) * 128 
    
    #################### STEP 3 #######################

    def send_int_challenges(self):
        def generate_challenges(count: int) -> List[List[int]]:
            # [128/log(q)] arrays of count challenges
            return [[secrets.randbelow(params.Q) for _ in range(count)] for _ in range(params.FUNC_COUNT)]
        
        psi = generate_challenges(params.L)
        omega = generate_challenges(256)
        return (psi, omega)
    
    def check_b(self, b_second: List[poly], psi: List[List[int]], omega: List[List[int]], projection: List[int]):
        for k in range(params.FUNC_COUNT):
            res = sum([(c1 * c2) % params.Q for (c1, c2) in zip(omega[k], projection)]) % params.Q
            for l in range(params.L):
                res += psi[k][l] * self.b_prime[l][0] % params.Q
            res %= params.Q
            if b_second[k][0] != res:
                return False
        return True

    #################### STEP 4 #######################
    
    def send_poly_challenges(self):
        alpha = poly_vector.random_vector(params.K)
        beta = poly_vector.random_vector(params.FUNC_COUNT)
        return alpha, beta
    
    #################### STEP 4 #######################

    def __sample_challenge(self) -> poly:
        signs = [-1, 1]
        coeff_freq = [23, 31, 10]
        f = poly()
        for i in range(params.DEGREE):
            coeff = secrets.randbelow(3)
            while coeff_freq[coeff] == 0:
                coeff = secrets.randbelow(3)
            coeff_freq[coeff] -= 1
            sign = secrets.choice(signs)
            f[i] = sign * coeff
        return f

    def send_c_challenges(self):
        # TODO: operator norm check???
        return [self.__sample_challenge() for _ in range(params.R)]