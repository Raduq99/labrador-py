from labrador.poly import poly
from labrador.poly_vector import poly_vector
from labrador.poly_matrix import poly_matrix
from labrador.prover import Prover
from labrador.verifier import Verifier
import labrador.params as params
import numpy as np
import secrets

def main():
    # p1 = poly()
    # p2 = poly(np.random.randint(params.Q, size=params.DEGREE))
    # v1 = poly_vector.random_vector()
    # print(v1[1])
    # print([item for sublist in v1.data for item in sublist.data])
    # v2 = poly_vector.random_vector()
    # s = [poly_vector.random_vector() for _ in range(params.R)]
    # beta = 333
    # a = [poly_matrix.random_matrix(params.R, params.R) for _ in range(params.K)]
    # a_prime = [poly_matrix.random_matrix(params.R, params.R) for _ in range(params.L)]
    # phi = [[poly_vector.random_vector() for _ in range(params.R)] for _ in range(params.K)]
    # phi_prime = [[poly_vector.random_vector() for _ in range(params.R)] for _ in range(params.L)]

    s = [poly_vector.test_vector() for _ in range(params.R)]
    beta = 10
    a = [poly_matrix.test_matrix(params.R, params.R) for _ in range(params.K)]
    a_prime = [poly_matrix.test_matrix(params.R, params.R) for _ in range(params.L)]
    phi = [[poly_vector.test_vector() for _ in range(params.R)] for _ in range(params.K)]
    phi_prime = [[poly_vector.test_vector() for _ in range(params.R)] for _ in range(params.L)]

    p = Prover(s, beta, a, phi, a_prime, phi_prime)
    b, b_prime = p.compute_constraints()
    v = Verifier(beta, a, phi, a_prime, phi_prime, b, b_prime)

    u1 = p.commit_u1()

    pi = v.generate_projections()
    proj = p.apply_projections(pi)
    norm_check = v.check_projection(proj)

    psi, omega = v.send_int_challenges()
    a_second, phi_second, b_second = p.compute_aggregations(psi, omega, pi)
    aggr_check = v.check_b(b_second, psi, omega, proj)

    alpha, beta = v.send_poly_challenges()
    phi = p.compute_phi(alpha, beta, phi_second)
    u2 = p.commit_u2(phi)
    
    c = v.send_c_challenges()
    z = p.compute_z(c)

    print(norm_check)
    print(aggr_check) # not always true
    print(z)

def test():
    # s1 = [secrets.randbelow(4) for _ in range(6)]
    # s2 = [secrets.randbelow(4) for _ in range(6)]
    # for _ in range(100):
    #     print(np.polymul(s1, s2))
    A = poly_matrix(data=[[poly([4, 2])]])
    w = poly_vector(data=[poly([0, 0])])
    print(f"A={A}")
    print(f"w={w}")
    print(A*w)

if __name__ == "__main__":
    # for _ in range(100):
    main()
    # test()