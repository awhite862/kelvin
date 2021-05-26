import unittest
import numpy
from kelvin import cc_utils
from kelvin import ft_cc_energy
from kelvin import ft_cc_equations
from kelvin import quadrature
from kelvin.h2_field_system import h2_field_system
from kelvin.neq_ccsd import neq_ccsd


def evalL(T1f, T1b, T1i, T2f, T2b, T2i, L1f, L1b, L1i, L2f, L2b, L2i,
          Ff, Fb, F, I, D1, D2, tir, tii, gr, gi, Gr, Gi, beta):
    ngr = gr.shape[0]
    ngi = gi.shape[0]
    E = ft_cc_energy.ft_cc_energy_neq(
        T1f, T1b, T1i, T2f, T2b, T2i,
        Ff.ov, Fb.ov, F.ov, I.oovv, gr, gi, beta)
    T1f_, T1b_, T1i_, T2f_, T2b_, T2i_ =\
        ft_cc_equations.neq_ccsd_stanton(
            Ff, Fb, F, I, T1f, T1b, T1i, T2f, T2b, T2i,
            D1, D2, tir, tii, ngr, ngi, Gr, Gi)
    T1f_ -= T1f
    T1b_ -= T1b
    T1i_ -= T1i
    T2f_ -= T2f
    T2b_ -= T2b
    T2i_ -= T2i

    TEf = 0.25*numpy.einsum('yijab,yabij->y', L2f, T2f_)
    TEf += numpy.einsum('yia,yai->y', L1f, T1f_)
    TEb = 0.25*numpy.einsum('yijab,yabij->y', L2b, T2b_)
    TEb += numpy.einsum('yia,yai->y', L1b, T1b_)
    TEi = 0.25*numpy.einsum('yijab,yabij->y', L2i, T2i_)
    TEi += numpy.einsum('yia,yai->y', L1i, T1i_)

    Te = (1.j/beta)*numpy.einsum('y,y->', TEf, gr)
    Te -= (1.j/beta)*numpy.einsum('y,y->', TEb, gr)
    Te += (1.0/beta)*numpy.einsum('y,y->', TEi, gi)

    return E + Te


def test_L1(cc, thresh):
    T = cc.T
    beta = 1.0/T
    mu = cc.mu
    ngr = cc.ngr
    ngi = cc.ngi
    tii, gi, Gi = quadrature.simpsons(ngi, beta)
    tir, gr, Gr = quadrature.midpoint(ngr, cc.tmax)
    en = cc.sys.g_energies_tot()
    D1 = en[:,None] - en[None,:]
    D2 = en[:,None,None,None] + en[None,:,None,None] \
        - en[None,None,:,None] - en[None,None,None,:]
    F, Ff, Fb, I = cc_utils.get_ft_integrals_neq(cc.sys, en, beta, mu)
    n = cc.L2f.shape[1]
    for y in range(ngi):
        for i in range(n):
            for a in range(n):
                d = 1.e-4
                TP = cc.T1i.copy()
                TM = cc.T1i.copy()
                TP[y, a, i] += d
                TM[y, a, i] -= d
                LP = evalL(cc.T1f, cc.T1b, TP, cc.T2f, cc.T2b, cc.T2i,
                           cc.L1f, cc.L1b, cc.L1i, cc.L2f, cc.L2b, cc.L2i,
                           Ff, Fb, F, I, D1, D2, tir, tii, gr, gi, Gr, Gi, beta)
                LM = evalL(cc.T1f, cc.T1b, TM, cc.T2f, cc.T2b, cc.T2i,
                           cc.L1f, cc.L1b, cc.L1i, cc.L2f, cc.L2b, cc.L2i,
                           Ff, Fb, F, I, D1, D2, tir, tii, gr, gi, Gr, Gi, beta)
                diff = (LP - LM)/(2*d)
                if numpy.abs(diff) > thresh:
                    return ('I-- {} {} {}: {}'.format(y, i, a, diff), False)
    for y in range(ngr):
        for i in range(n):
            for a in range(n):
                d = 1.e-4
                TP = cc.T1b.copy()
                TM = cc.T1b.copy()
                TP[y, a, i] += d
                TM[y, a, i] -= d
                LP = evalL(cc.T1f, TP, cc.T1i, cc.T2f, cc.T2b, cc.T2i,
                           cc.L1f, cc.L1b, cc.L1i, cc.L2f, cc.L2b, cc.L2i,
                           Ff, Fb, F, I, D1, D2, tir, tii, gr, gi, Gr, Gi, beta)
                LM = evalL(cc.T1f, TM, cc.T1i, cc.T2f, cc.T2b, cc.T2i,
                           cc.L1f, cc.L1b, cc.L1i, cc.L2f, cc.L2b, cc.L2i,
                           Ff, Fb, F, I, D1, D2, tir, tii, gr, gi, Gr, Gi, beta)
                diff = (LP - LM)/(2*d)
                if numpy.abs(diff) > thresh:
                    return ('B-- {} {} {}: {}'.format(y, i, a, diff), False)
    for y in range(ngr):
        for i in range(n):
            for a in range(n):
                d = 1.e-4
                TP = cc.T1f.copy()
                TM = cc.T1f.copy()
                TP[y, a, i] += d
                TM[y, a, i] -= d
                LP = evalL(TP, cc.T1b, cc.T1i, cc.T2f, cc.T2b, cc.T2i,
                           cc.L1f, cc.L1b, cc.L1i, cc.L2f, cc.L2b, cc.L2i,
                           Ff, Fb, F, I, D1, D2, tir, tii, gr, gi, Gr, Gi, beta)
                LM = evalL(TM, cc.T1b, cc.T1i, cc.T2f, cc.T2b, cc.T2i,
                           cc.L1f, cc.L1b, cc.L1i, cc.L2f, cc.L2b, cc.L2i,
                           Ff, Fb, F, I, D1, D2, tir, tii, gr, gi, Gr, Gi, beta)
                diff = (LP - LM)/(2*d)
                if numpy.abs(diff) > thresh:
                    return ('F-- {} {} {}: {}'.format(y, i, a, diff), False)
    return ("pass",True)


class NEQLambdaTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-8

    def test_h2_null_field(self):
        beta = 1.0
        T = 1./beta
        mu = 0.0
        omega = 0.0
        ngi = 20
        ngr = 80

        deltat = 0.04
        tmax = (ngr)*deltat
        tir = numpy.asarray([deltat/2.0 + float(j)*deltat for j in range(ngr)])
        sys = h2_field_system(T, mu, omega, tir, O=None, ot=None)
        cc = neq_ccsd(
            sys, T, mu=mu, tmax=tmax, econv=1e-12,
            max_iter=40, damp=0.0, ngr=ngr, ngi=ngi, iprint=0)
        E = cc.run()
        cc._neq_ccsd_lambda()

        # Check that L is zero
        beta = 1.0/T
        tii, gi, Gi = quadrature.simpsons(ngi, beta)
        tir, gr, Gr = quadrature.midpoint(ngr, tmax)
        en = cc.sys.g_energies_tot()
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
            - en[None,None,:,None] - en[None,None,None,:]
        F, Ff, Fb, I = cc_utils.get_ft_integrals_neq(cc.sys, en, beta, mu)
        L = evalL(cc.T1f, cc.T1b, cc.T1i, cc.T2f, cc.T2b, cc.T2i,
                  cc.L1f, cc.L1b, cc.L1i, cc.L2f, cc.L2b, cc.L2i,
                  Ff, Fb, F, I, D1, D2, tir, tii, gr, gi, Gr, Gi, beta)
        self.assertTrue(abs(L - E[1]) < self.thresh, "Lagrangian does not equal the Energy: {}".format(L))

        out = test_L1(cc, self.thresh)
        self.assertTrue(out[1], out[0])

    def test_h2_field(self):
        beta = 1.0
        T = 1./beta
        mu = 0.0
        omega = 0.5
        ngi = 14
        ngr = 80

        deltat = 0.04
        tmax = (ngr)*deltat
        beta = 1.0/T
        tii, gi, Gi = quadrature.simpsons(ngi, beta)
        tir, gr, Gr = quadrature.midpoint(ngr, tmax)
        sys = h2_field_system(T, mu, omega, tir, O=None, ot=None)
        ccsdT = neq_ccsd(
            sys, T, mu=mu, tmax=tmax, econv=1e-12,
            max_iter=40, damp=0.0, ngr=ngr, ngi=ngi, iprint=0)
        E = ccsdT.run()
        ccsdT._neq_ccsd_lambda()

        # Check that L is zero
        en = ccsdT.sys.g_energies_tot()
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
            - en[None,None,:,None] - en[None,None,None,:]
        cc = ccsdT
        F, Ff, Fb, I = cc_utils.get_ft_integrals_neq(cc.sys, en, beta, mu)
        L = evalL(cc.T1f, cc.T1b, cc.T1i, cc.T2f, cc.T2b, cc.T2i,
                  cc.L1f, cc.L1b, cc.L1i, cc.L2f, cc.L2b, cc.L2i,
                  Ff, Fb, F, I, D1, D2, tir, tii, gr, gi, Gr, Gi, beta)
        self.assertTrue(abs(L - E[1]) < self.thresh, "Lagrangian does not equal the Energy: {}  {}".format(L, E[1]))

        out = test_L1(ccsdT, self.thresh)
        self.assertTrue(out[1], out[0])


if __name__ == '__main__':
    unittest.main()
