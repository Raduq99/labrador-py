from .poly_matrix import poly_matrix
from .poly_vector import poly_vector
from .poly import poly
from . import params
from typing import List
import numpy as np

class Prover:
    def __init__(self, 
                 s: List[poly_vector] = None, 
                 beta: int = None, 
                 a: List[poly_matrix] = None, 
                 phi: List[poly_vector] = None, 
                 a_prime: List[List[poly_matrix]] = None, 
                 phi_prime: List[List[poly_vector]] = None):
        # s - array of r poly vectors(size n)
        self.s = s
        # beta - norm bound
        self.beta = beta
        # a - array of K poly matrices(r x r)
        self.a = a
        # phi - array of K arrays of R poly vectors(size n)
        self.phi = phi
        # a' - array of L poly matrices(r x r)
        self.a_prime = a_prime
        # phi' - array of L arrays of R poly vectors(size n)
        self.phi_prime = phi_prime

    def compute_constraints(self):
        self.b = []
        for k in range(params.K):
            b = poly()
            for i in range(params.R):
                for j in range(params.R):
                    b += self.a[k][i][j] * poly_vector.inner_product(self.s[i], self.s[j])
            for i in range(params.R):
                b += poly_vector.inner_product(self.phi[k][i], self.s[i])
            self.b.append(b)
        self.b_prime = []
        for l in range(params.L):
            b_prime = poly()
            for i in range(params.R):
                for j in range(params.R):
                    b_prime += self.a_prime[l][i][j] * poly_vector.inner_product(self.s[i], self.s[j])
            for i in range(params.R):
                b_prime += poly_vector.inner_product(self.phi_prime[l][i], self.s[i])
            self.b_prime.append(b_prime)
        return self.b, self.b_prime
    
    #################### STEP 1 #######################

    @staticmethod
    def ajtai_commit(w: poly_vector, commit_size) -> poly_vector:
        A = poly_matrix.random_matrix(commit_size, len(w.data))
        return A * w
    
    @staticmethod
    def decompose(t, size: int, base: int):
        if isinstance(t, poly_vector):
            res = [poly_vector(None, params.k) for _ in range(size)]
            poly_index = 0
            coeff_index = 0
            base_index = 0
            for ti in t.data:
                coeff_index = 0
                for coeff in ti.data:
                    base_index = 0
                    while coeff != 0:
                        res[base_index].data[poly_index].data[coeff_index] = coeff % base
                        coeff //= base
                        base_index += 1
                    coeff_index += 1
                poly_index += 1
            return res
        elif isinstance(t, poly):
            res = poly_vector(None, size)
            coeff_index = 0
            base_index = 0
            for coeff in t.data:
                base_index = 0
                while coeff != 0:
                    res.data[base_index].data[coeff_index] = coeff % base
                    coeff //= base
                    base_index += 1
                coeff_index += 1
            return res
        else:
            raise Exception("Wrong input")

    def construct_G(self) -> poly_matrix:
        data = [
            [poly_vector.inner_product(si, sj) for si in self.s] for sj in self.s
        ]
        return poly_matrix(data)

    def commit_u1(self):
        witness_commits = []
        for si in self.s:
            ti = Prover.ajtai_commit(si, params.k)
            witness_commits.append(Prover.decompose(ti, params.T1, params.B1))
        G = self.construct_G()
        garbage = []
        for r in G.data:
            garbage.append([])
            for c in r:
                garbage[-1].append(Prover.decompose(c, params.T2, params.B2))
        u1 = poly_vector(size=params.k)
        for i in range(params.R):
            for k in range(params.T1):
                u1 += Prover.ajtai_commit(witness_commits[i][k], params.k1)
        for i in range(params.R):
            for j in range(i, params.R):
                for k in range(params.T2):
                    u1 += Prover.ajtai_commit(poly_vector([garbage[i][j][k]]), params.k2)
        return u1
    
    #################### STEP 2 #######################

    def apply_projections(self, projections: List[List[List[int]]]) -> List[int]:
        p = []
        for j in range(256):
            for i in range(params.R):
                s_coeff = self.s[i].get_coeff_vector()
                proj_coeff = projections[i][j].get_coeff_vector()
                proj = sum([(c1 * c2) % params.Q for (c1, c2) in zip(s_coeff, proj_coeff)]) % params.Q
                p.append(proj)
        return p
    
    #################### STEP 3 #######################

    def __compute_aggregations(self, psi: List[int], omega: List[int], pi):
        # a'' - poly matrix (r x r)
        a_second = poly_matrix(rowSize=params.R, columnSize=params.R)
        for i in range(params.R):
            for j in range(params.R):
                for l in range(params.L):
                    a_second.data[i][j] += self.a_prime[l][i][j] * psi[l] 
        a_second

        # phi'' - array of R poly vectors (size N)
        phi_second = [poly_vector(size=params.N) for _ in range(params.R)]
        for i in range(params.R):
            for l in range(params.L):
                phi_second[i] += self.phi_prime[l][i] * psi[l]
            for j in range(256):
                p = pi[i][j].conjugate()
                o = omega[j]
                po = p * o
                phi_second[i] += po

        # b'' - poly
        b_second = poly()
        for i in range(params.R):
            for j in range(params.R):
                b_second += a_second[i][j] * poly_vector.inner_product(self.s[i], self.s[j])
        for i in range(params.R):
            b_second += poly_vector.inner_product(phi_second[i], self.s[i])

        return a_second, phi_second, b_second

    def compute_aggregations(self, psi: List[List[int]], omega: List[List[int]], pi):
        a_second, phi_second, b_second = [], [], []
        for k in range(params.FUNC_COUNT):
            (asec, psec, bsec) = self.__compute_aggregations(psi[k], omega[k], pi)
            a_second.append(asec)
            phi_second.append(psec)
            b_second.append(bsec)
        return a_second, phi_second, b_second

    #################### STEP 4 #######################

    def compute_phi(self, alpha: poly_vector, beta: poly_vector, phi_second) -> List[poly_vector]:
        phi = [poly_vector() for _ in range(params.R)]
        for i in range(params.R):
            for k in range(params.K):
                phi[i] += self.phi[k][i] * alpha[k]
            for k in range(params.FUNC_COUNT):
                phi[i] +=  phi_second[k][i] * beta[k]
        return phi

    def construct_H(self, phi: List[poly_vector]) -> poly_matrix:
        data = []
        for i in range(params.R):
            data.append([])
            for j in range(params.R):
                h = (poly_vector.inner_product(phi[i], self.s[j]) + poly_vector.inner_product(phi[j], self.s[i])) * 0.5
                data[-1].append(h)
        return poly_matrix(data)

    def commit_u2(self, phi: List[poly_vector]):
        H = self.construct_H(phi)
        garbage = []
        for r in H:
            garbage.append([])
            for c in r:
                garbage[-1].append(Prover.decompose(c, params.T1, params.B1))
        u2 = poly_vector(size=params.k2)
        for i in range(params.R):
            for j in range(params.R):
                for k in range(params.T1):
                    u2 += Prover.ajtai_commit(poly_vector([garbage[i][j][k]]), params.k2)
        return u2

    #################### STEP 5 #######################

    def compute_z(self, c: List[poly]) -> poly_vector:
        z = poly_vector()
        for i in range(params.R):
            z += self.s[i] * c[i]
        return z
