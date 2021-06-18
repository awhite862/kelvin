import unittest
import numpy

from cqcpy import test_utils
from cqcpy import spin_utils

from kelvin import ft_cc_equations
from kelvin import quadrature


class FTamplEquationsTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-12
        self.n = 5
        self.ng = 4
        self.beta = 2.0

    def test_ccsd_stanton(self):
        ng = self.ng
        n = self.n
        T1old, T2old = test_utils.make_random_ft_T(ng, n)
        L1old, L2old = test_utils.make_random_ft_T(ng, n)
        F, I = test_utils.make_random_integrals(n, n)
        D1, D2 = test_utils.make_random_ft_D(n)
        ti, g, G = quadrature.simpsons(ng, self.beta)

        T1sim, T2sim = ft_cc_equations.ccsd_simple(
            F, I, T1old, T2old, D1, D2, ti, ng, G)
        T1stn, T2stn = ft_cc_equations.ccsd_stanton(
            F, I, T1old, T2old, D1, D2, ti, ng, G)

        diff1 = numpy.linalg.norm(T1stn - T1sim)
        diff2 = numpy.linalg.norm(T2stn - T2sim)
        s1 = diff1 < self.thresh*numpy.sqrt(T1sim.size)
        s2 = diff2 < self.thresh*numpy.sqrt(T2sim.size)
        e1 = "Error in Stanton FT T1: {}".format(diff1)
        e2 = "Error in Stanton FT T2: {}".format(diff2)
        self.assertTrue(s1, e1)
        self.assertTrue(s2, e2)

    def test_uccsd(self):
        ng = self.ng
        na = self.n
        nb = self.n
        n = na + nb

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

        T1a = numpy.zeros((ng, na, na))
        T1b = numpy.zeros((ng, nb, nb))
        for i in range(ng):
            T1at, T1bt = test_utils.make_random_T1_spatial(na, na, nb, nb)
            T1a[i] = T1at
            T1b[i] = T1bt
        T2aa = numpy.zeros((ng, na, na, na, na))
        T2ab = numpy.zeros((ng, na, nb, na, nb))
        T2bb = numpy.zeros((ng, nb, nb, nb, nb))
        T1old = numpy.zeros((ng, n, n))
        T2old = numpy.zeros((ng, n, n, n, n))
        for i in range(ng):
            T2aat, T2abt, T2bbt = test_utils.make_random_T2_spatial(na, na, nb, nb)
            T2aa[i] = T2aat
            T2ab[i] = T2abt
            T2bb[i] = T2bbt
        for i in range(ng):
            T1old[i] = spin_utils.T1_to_spin(
                T1a[i], T1b[i], na, na, nb, nb)
            T2old[i] = spin_utils.T2_to_spin(
                T2aa[i], T2ab[i], T2bb[i], na, na, nb, nb)

        D1a, D2aa = test_utils.make_random_ft_D(na)
        D2ab = test_utils.make_random_ft_D2(na, nb)
        D1b, D2bb = test_utils.make_random_ft_D(nb)
        D1 = spin_utils.T1_to_spin(D1a, D1b, na, na, nb, nb)
        D2 = spin_utils.D2_to_spin(D2aa, D2ab, D2bb, na, na, nb, nb)

        ti, g, G = quadrature.simpsons(ng, self.beta)

        T1ref, T2ref = ft_cc_equations.ccsd_stanton(
            F, I, T1old, T2old, D1, D2, ti, ng, G)
        T1out, T2out = ft_cc_equations.uccsd_stanton(
            Fa, Fb, Ia, Ib, Iabab, T1a, T1b, T2aa, T2ab, T2bb,
            D1a, D1b, D2aa, D2ab, D2bb, ti, ng, G)

        T1 = numpy.zeros((ng, n, n))
        T2 = numpy.zeros((ng, n, n, n, n))
        for i in range(ng):
            T1[i] = spin_utils.T1_to_spin(
                T1out[0][i], T1out[1][i], na, na, nb, nb)
            T2[i] = spin_utils.T2_to_spin(
                T2out[0][i], T2out[1][i], T2out[2][i], na, na, nb, nb)

        diff1 = numpy.linalg.norm(T1ref - T1)
        diff2 = numpy.linalg.norm(T2ref - T2)
        s1 = diff1 < self.thresh*numpy.sqrt(T1ref.size)
        s2 = diff2 < self.thresh*numpy.sqrt(T2ref.size)
        e1 = "Error in unrestricted FT T1: {}".format(diff1)
        e2 = "Error in unrestricted FT T2: {}".format(diff2)
        self.assertTrue(s1, e1)
        self.assertTrue(s2, e2)


if __name__ == '__main__':
    unittest.main()
