import unittest
import numpy
from pyscf import gto, scf
from cqcpy import ft_utils
from cqcpy import utils
from cqcpy import spin_utils
from cqcpy import test_utils
from kelvin.ccsd import ccsd
from kelvin.scf_system import SCFSystem
from kelvin import cc_utils
from kelvin import ft_cc_energy
from kelvin import ft_cc_equations
from kelvin import quadrature


def fd_test_L1(cc, thresh):
    # test lambdas
    T = cc.T
    beta = 1.0/T
    mu = cc.mu
    ng = cc.ngrid
    ti,g,G = quadrature.simpsons(ng, beta)
    en = cc.sys.g_energies_tot()
    fo = ft_utils.ff(beta, en, mu)
    fv = ft_utils.ffv(beta, en, mu)
    D1 = utils.D1(en, en)
    D2 = utils.D2(en, en)
    if cc.athresh > 0.0:
        athresh = cc.athresh
        focc = [x for x in fo if x > athresh]
        fvir = [x for x in fv if x > athresh]
        iocc = [i for i, x in enumerate(fo) if x > athresh]
        ivir = [i for i, x in enumerate(fv) if x > athresh]
        F,I = cc_utils.ft_active_integrals(cc.sys, en, focc, fvir, iocc, ivir)

        D1 = D1[numpy.ix_(ivir, iocc)]
        D2 = D2[numpy.ix_(ivir, ivir, iocc, iocc)]
    else:
        F,I = cc_utils.ft_integrals(cc.sys, en, beta, mu)
    ng,no,nv = cc.L1.shape
    d = 1e-4
    for y in range(ng):
        for i in range(no):
            for a in range(nv):
                TF = cc.T1.copy()
                TB = cc.T1.copy()
                TF[y,a,i] += d
                TB[y,a,i] -= d
                EF = ft_cc_energy.ft_cc_energy(TF, cc.T2, F.ov, I.oovv, g, beta)
                EB = ft_cc_energy.ft_cc_energy(TB, cc.T2, F.ov, I.oovv, g, beta)
                TF1, TF2 = ft_cc_equations.ccsd_stanton(F, I, TF, cc.T2, D1, D2, ti, ng, G)
                TB1, TB2 = ft_cc_equations.ccsd_stanton(F, I, TB, cc.T2, D1, D2, ti, ng, G)
                TF2 -= cc.T2
                TB2 -= cc.T2
                TF1 -= TF
                TB1 -= TB
                TEf = 0.25*numpy.einsum('yijab,yabij->y', cc.L2, TF2)
                TEb = 0.25*numpy.einsum('yijab,yabij->y', cc.L2, TB2)
                TEf += numpy.einsum('yia,yai->y', cc.L1, TF1)
                TEb += numpy.einsum('yia,yai->y', cc.L1, TB1)
                Tef = (1.0/beta)*numpy.einsum('y,y->', TEf, g)
                Teb = (1.0/beta)*numpy.einsum('y,y->', TEb, g)
                fw = EF - Tef
                bw = EB - Teb
                diff = (fw - bw)/(2*d)
                if numpy.abs(diff) > thresh:
                    return ('{} {} {}: {}'.format(y, i, a, diff), False)
    return ("pass", True)


def fd_test_L2(cc, thresh):
    # test lambdas
    T = cc.T
    beta = 1.0/T
    mu = cc.mu
    ng = cc.ngrid
    ti,g,G = quadrature.simpsons(ng, beta)
    en = cc.sys.g_energies_tot()
    D1 = utils.D1(en, en)
    D2 = utils.D2(en, en)
    F,I = cc_utils.ft_integrals(cc.sys, en, beta, mu)
    n = cc.L2.shape[1]
    d = 1e-4
    for y in range(ng):
        for i in range(n):
            for j in range(n):
                for a in range(n):
                    for b in range(n):
                        TF = cc.T2.copy()
                        TB = cc.T2.copy()
                        TF[y,a,b,i,j] += d
                        TF[y,a,b,j,i] -= d
                        TF[y,b,a,i,j] -= d
                        TF[y,b,a,j,i] += d
                        TB[y,a,b,i,j] -= d
                        TB[y,a,b,j,i] += d
                        TB[y,b,a,i,j] += d
                        TB[y,b,a,j,i] -= d
                        EF = ft_cc_energy.ft_cc_energy(cc.T1, TF, F.ov, I.oovv, g, beta)
                        EB = ft_cc_energy.ft_cc_energy(cc.T1, TB, F.ov, I.oovv, g, beta)
                        TF1, TF2 = ft_cc_equations.ccsd_stanton(F, I, cc.T1, TF, D1, D2, ti, ng, G)
                        TB1, TB2 = ft_cc_equations.ccsd_stanton(F, I, cc.T1, TB, D1, D2, ti, ng, G)
                        TF2 -= TF
                        TB2 -= TB
                        TF1 -= cc.T1
                        TB1 -= cc.T1
                        TEf = 0.25*numpy.einsum('yijab,yabij->y', cc.L2, TF2)
                        TEb = 0.25*numpy.einsum('yijab,yabij->y', cc.L2, TB2)
                        TEf += numpy.einsum('yia,yai->y', cc.L1, TF1)
                        TEb += numpy.einsum('yia,yai->y', cc.L1, TB1)
                        Tef = (1.0/beta)*numpy.einsum('y,y->', TEf, g)
                        Teb = (1.0/beta)*numpy.einsum('y,y->', TEb, g)
                        fw = EF - Tef
                        bw = EB - Teb
                        diff = (fw - bw)/(2*d)
                        if numpy.abs(diff) > 1e-7:
                            return ('{} {} {} {} {}: {}'.format(y, i, j, a, b, diff), False)
    return ("pass", True)


class FTLambdaTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-8
        self.T = 2.0
        self.mu = 0.05

    def test_Be_sto3g_gen(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-13
        m.scf()
        sys = SCFSystem(m, self.T, self.mu, orbtype='g')
        ccsdT = ccsd(sys, T=self.T, mu=self.mu, iprint=0, max_iter=44, econv=1e-12)
        Etot, Ecc = ccsdT.run()
        ccsdT._ft_ccsd_lambda()
        out = fd_test_L1(ccsdT, self.thresh)
        self.assertTrue(out[1], out[0])

    def test_Be_sto3g_gen_active(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-13
        m.scf()
        T = 0.03
        mu = 0.0
        sys = SCFSystem(m, T, mu, orbtype='g')
        ccsdT = ccsd(sys, T=T, mu=mu, iprint=0, damp=0.45, max_iter=240, ngrid=20, econv=1e-10, tconv=1e-9, athresh=1e-20)
        Etot, Ecc = ccsdT.run()
        ccsdT._ft_ccsd_lambda()
        out = fd_test_L1(ccsdT, 1e-7)
        self.assertTrue(out[1], out[0])

    def test_Be_sto3g(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-13
        m.scf()
        ng = 7
        sys = SCFSystem(m, self.T, self.mu, orbtype='u')
        ccsdT = ccsd(sys, T=self.T, mu=self.mu, ngrid=ng, iprint=0, max_iter=44, econv=1e-12)
        Etot, Ecc = ccsdT.run()
        ccsdT._ft_uccsd_lambda()
        ea,eb = ccsdT.sys.u_energies_tot()
        na = ea.shape[0]
        nb = eb.shape[0]
        n = na + nb

        # convert to spin orbitals
        L1 = numpy.zeros((ng, n, n))
        L2 = numpy.zeros((ng, n, n, n, n))
        T1 = numpy.zeros((ng, n, n))
        T2 = numpy.zeros((ng, n, n, n, n))
        for y in range(ng):
            L1[y] = spin_utils.T1_to_spin(
                ccsdT.L1[0][y], ccsdT.L1[1][y], na, na, nb, nb)
            L2[y] = spin_utils.T2_to_spin(
                ccsdT.L2[0][y], ccsdT.L2[1][y], ccsdT.L2[2][y], na, na, nb, nb)
            T1[y] = spin_utils.T1_to_spin(
                ccsdT.T1[0][y], ccsdT.T1[1][y], na, na, nb, nb)
            T2[y] = spin_utils.T2_to_spin(
                ccsdT.T2[0][y], ccsdT.T2[1][y], ccsdT.T2[2][y], na, na, nb, nb)
        nccsdT = ccsd(sys, T=self.T, mu=self.mu, ngrid=ng, iprint=0, max_iter=44, econv=1e-12)
        nccsdT.L1 = L1
        nccsdT.L2 = L2
        nccsdT.T1 = T1
        nccsdT.T2 = T2
        out = fd_test_L1(nccsdT, self.thresh)
        self.assertTrue(out[1], out[0])

    def test_Be_deriv(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-13
        m.scf()
        ng = 7
        T = self.T
        mu = self.mu
        sys = SCFSystem(m, T, mu, orbtype='g')
        cc = ccsd(sys, T=T, mu=mu, iprint=0, max_iter=44, ngrid=ng, econv=1e-12)
        Etot, Ecc = cc.run()
        n = sys.g_energies_tot().shape[0]
        G = cc.G
        g = cc.g
        ti = cc.ti

        # random Lambda amplitudes
        L1, L2 = test_utils.make_random_ft_T(ng, n)

        # compute derivative from L
        T = cc.T
        beta = 1.0/T
        en = cc.sys.g_energies_tot()
        D1 = utils.D1(en, en)
        D2 = utils.D2(en, en)
        F,I = cc_utils.ft_integrals(sys, en, beta, mu)
        dT1 = numpy.zeros((ng, n, n))
        for y in range(ng):
            for i in range(n):
                for a in range(n):
                    d = 1.e-4
                    TF = cc.T1.copy()
                    TB = cc.T1.copy()
                    TF[y,a,i] += d
                    TB[y,a,i] -= d
                    EF = ft_cc_energy.ft_cc_energy(TF, cc.T2, F.ov, I.oovv, g, beta)
                    EB = ft_cc_energy.ft_cc_energy(TB, cc.T2, F.ov, I.oovv, g, beta)
                    TF1, TF2 = ft_cc_equations.ccsd_stanton(F, I, TF, cc.T2, D1, D2, ti, ng, G)
                    TB1, TB2 = ft_cc_equations.ccsd_stanton(F, I, TB, cc.T2, D1, D2, ti, ng, G)
                    TF2 -= cc.T2
                    TB2 -= cc.T2
                    TF1 -= TF
                    TB1 -= TB
                    TEf = 0.25*numpy.einsum('yijab,yabij->y', L2, TF2)
                    TEb = 0.25*numpy.einsum('yijab,yabij->y', L2, TB2)
                    TEf += numpy.einsum('yia,yai->y', L1, TF1)
                    TEb += numpy.einsum('yia,yai->y', L1, TB1)
                    Tef = (1.0/beta)*numpy.einsum('y,y->', TEf, g)
                    Teb = (1.0/beta)*numpy.einsum('y,y->', TEb, g)
                    fw = EF - Tef
                    bw = EB - Teb
                    dT1[y,a,i] = (fw - bw)/(2*d)

        # compute derivative from Lambda equations
        dT1L = numpy.zeros((ng, n, n))
        dT1L, L2n = ft_cc_equations.ccsd_lambda_simple(
            F, I, cc.T1, cc.T2, L1, L2, D1, D2, ti, ng, g, G, beta)
        dT1L = -(dT1L - L1)
        dT1L = dT1L.transpose((0, 2, 1))
        for y in range(ng):
            dT1L[y] *= g[y]/beta

        diff = numpy.linalg.norm(dT1L - dT1)/numpy.sqrt(dT1L.size)
        self.assertTrue(diff < 1e-8)


if __name__ == '__main__':
    unittest.main()
