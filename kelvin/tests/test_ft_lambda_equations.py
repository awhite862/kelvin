import unittest
import numpy

from cqcpy import test_utils
from cqcpy import spin_utils

from kelvin import ft_cc_equations
from kelvin import quadrature


class FTLambdaEquationsTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-11
        self.n = 5
        self.ng = 4
        self.beta = 2.0

    def test_ccsd_opt(self):
        ng = self.ng
        n = self.n
        beta = self.beta
        T1old, T2old = test_utils.make_random_ft_T(ng, n)
        L1old, L2old = test_utils.make_random_ft_T(ng, n)
        F, I = test_utils.make_random_integrals(n, n)
        D1, D2 = test_utils.make_random_ft_D(n)
        ti, g, G = quadrature.simpsons(ng, beta)

        L1sim, L2sim = ft_cc_equations.ccsd_lambda_simple(
            F, I, T1old, T2old, L1old, L2old, D1, D2, ti, ng, g, G, beta)
        L1opt, L2opt = ft_cc_equations.ccsd_lambda_opt(
            F, I, T1old, T2old, L1old, L2old, D1, D2, ti, ng, g, G, beta)

        diff1 = numpy.linalg.norm(L1opt - L1sim)
        diff2 = numpy.linalg.norm(L2opt - L2sim)
        s1 = diff1 < self.thresh*numpy.sqrt(L1opt.size)
        s2 = diff2 < self.thresh*numpy.sqrt(L2opt.size)
        e1 = "Error in optimized FT-CC L1"
        e2 = "Error in optimized FT-CC L2"
        self.assertTrue(s1, e1)
        self.assertTrue(s2, e2)

    def test_uccsd_opt(self):
        ng = self.ng
        na = self.n
        nb = self.n
        n = na + nb
        beta = self.beta

        # unrestricted integrals
        Fa = test_utils.make_random_F(na, na)
        Fb = test_utils.make_random_F(nb, nb)
        Ia = test_utils.make_random_I_anti(na, na)
        Ib = test_utils.make_random_I_anti(nb, nb)
        Iabab = test_utils.make_random_Ifull_gen(
            na, na, nb, nb, na, na, nb, nb)

        # Full antisymmetric spin-orbital tensor
        I = spin_utils.int_to_spin2(Ia, Ib, Iabab, na, na, nb, nb)
        F = spin_utils.F_to_spin(Fa, Fb, na, na, nb, nb)

        T1aold = numpy.zeros((ng, na, na))
        T1bold = numpy.zeros((ng, nb, nb))
        T1old = numpy.zeros((ng, n, n))
        T2old = numpy.zeros((ng, n, n, n, n))
        for i in range(ng):
            T1at, T1bt = test_utils.make_random_T1_spatial(na, na, nb, nb)
            T1aold[i] = T1at
            T1bold[i] = T1bt
        T2aaold = numpy.zeros((ng, na, na, na, na))
        T2abold = numpy.zeros((ng, na, nb, na, nb))
        T2bbold = numpy.zeros((ng, nb, nb, nb, nb))
        T1old = numpy.zeros((ng, n, n))
        T2old = numpy.zeros((ng, n, n, n, n))
        for i in range(ng):
            T2r = test_utils.make_random_T2_spatial(na, na, nb, nb)
            T2aaold[i] = T2r[0]
            T2abold[i] = T2r[1]
            T2bbold[i] = T2r[2]
        for i in range(ng):
            T1old[i] = spin_utils.T1_to_spin(
                T1aold[i], T1bold[i], na, na, nb, nb)
            T2old[i] = spin_utils.T2_to_spin(
                T2aaold[i], T2abold[i], T2bbold[i], na, na, nb, nb)

        L1aold = numpy.zeros((ng, na, na))
        L1bold = numpy.zeros((ng, nb, nb))
        L1old = numpy.zeros((ng, n, n))
        L2old = numpy.zeros((ng, n, n, n, n))
        for i in range(ng):
            L1at, L1bt = test_utils.make_random_T1_spatial(na, na, nb, nb)
            L1aold[i] = L1at
            L1bold[i] = L1bt
        L2aaold = numpy.zeros((ng, na, na, na, na))
        L2abold = numpy.zeros((ng, na, nb, na, nb))
        L2bbold = numpy.zeros((ng, nb, nb, nb, nb))
        L1old = numpy.zeros((ng, n, n))
        L2old = numpy.zeros((ng, n, n, n, n))
        for i in range(ng):
            L2r = test_utils.make_random_T2_spatial(na, na, nb, nb)
            L2aaold[i] = L2r[0]
            L2abold[i] = L2r[1]
            L2bbold[i] = L2r[2]
        for i in range(ng):
            L1old[i] = spin_utils.T1_to_spin(
                L1aold[i], L1bold[i], na, na, nb, nb)
            L2old[i] = spin_utils.T2_to_spin(
                L2aaold[i], L2abold[i], L2bbold[i], na, na, nb, nb)

        D1a, D2aa = test_utils.make_random_ft_D(na)
        D1b, D2bb = test_utils.make_random_ft_D(nb)
        D2ab = test_utils.make_random_ft_D2(na, nb)
        D1 = spin_utils.T1_to_spin(D1a, D1b, na, na, nb, nb)
        D2 = spin_utils.D2_to_spin(D2aa, D2ab, D2bb, na, na, nb, nb)

        ti, g, G = quadrature.simpsons(ng, beta)
        L1ref, L2ref = ft_cc_equations.ccsd_lambda_simple(
            F, I, T1old, T2old, L1old, L2old, D1, D2, ti, ng, g, G, beta)
        L1a, L1b, L2aa, L2ab, L2bb = ft_cc_equations.uccsd_lambda_opt(
            Fa, Fb, Ia, Ib, Iabab, T1aold, T1bold,
            T2aaold, T2abold, T2bbold, L1aold, L1bold,
            L2aaold, L2abold, L2bbold, D1a, D1b,
            D2aa, D2ab, D2bb, ti, ng, g, G, beta)

        L1out = numpy.zeros((ng, n, n))
        L2out = numpy.zeros((ng, n, n, n, n))
        for i in range(ng):
            L1out[i] = spin_utils.T1_to_spin(
                L1a[i], L1b[i], na, na, nb, nb)
            L2out[i] = spin_utils.T2_to_spin(
                L2aa[i], L2ab[i], L2bb[i], na, na, nb, nb)

        diff1 = numpy.linalg.norm(L1out - L1ref)
        diff2 = numpy.linalg.norm(L2out - L2ref)
        s1 = diff1 < self.thresh*numpy.sqrt(L1out.size)
        s2 = diff2 < self.thresh*numpy.sqrt(L2out.size)
        e1 = "Error in optimized FT-CC L1"
        e2 = "Error in optimized FT-CC L2"
        self.assertTrue(s1, e1)
        self.assertTrue(s2, e2)


if __name__ == '__main__':
    unittest.main()
