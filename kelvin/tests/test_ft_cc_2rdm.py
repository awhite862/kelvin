import unittest
import numpy
from pyscf import gto, scf

from cqcpy import ft_utils
from cqcpy import utils

from kelvin.ccsd import ccsd
from kelvin.scf_system import SCFSystem
from kelvin import cc_utils
from numpy import einsum

try:
    from lattice.hubbard import Hubbard1D
    from kelvin.hubbard_system import HubbardSystem
    has_lattice = True
except ImportError:
    has_lattice = False


class FakeHubbardSystem(object):
    def __init__(self, sys, M=None):
        self.M = M
        self.sys = sys

    def has_g(self):
        return self.sys.has_g()

    def has_u(self):
        return self.sys.has_u()

    def has_r(self):
        return self.sys.has_r()

    def verify(self, T, mu):
        return self.sys.verify(T, mu)

    def const_energy(self):
        return self.sys.const_energy()

    def get_mp1(self):
        E1 = self.sys.get_mp1()
        beta = 1.0/self.sys.T
        mu = self.sys.mu
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        extra = 0.5*numpy.einsum('ijij,i,j->', self.M, fo, fo)
        return E1 + extra

    def g_energies_tot(self):
        return self.sys.g_energies_tot()

    def g_fock_tot(self):
        beta = 1.0/self.sys.T
        mu = self.sys.mu
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        return self.sys.g_fock_tot() + 0.5*einsum('piqi,i->pq', self.M, fo)

    def g_aint_tot(self):
        U = self.sys.g_aint_tot()
        return U + self.M


class FTCC2RDMTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-13

    @unittest.skipUnless(has_lattice, "Lattice module cannot be found")
    def test_hubbard(self):
        U = 1.0
        T = 1.0
        L = 2
        mu = 0.5
        Mg = numpy.zeros((2*L, 2*L, 2*L, 2*L))
        for i in range(L):
            Mg[i, L+i, i, L+i] = -2.0
            Mg[L+i, i, L+i, i] = -2.0
        Mg = Mg - Mg.transpose((0, 1, 3, 2))
        hub = Hubbard1D(2, 1.0, U)
        Pa = numpy.zeros((2, 2))
        Pb = numpy.zeros((2, 2))
        Pa[0, 0] = 1.0
        Pb[1, 1] = 1.0
        sys = HubbardSystem(T, hub, Pa, Pb, mu=mu, orbtype='g')
        cmat = utils.block_diag(sys.ua, sys.ub)
        Mg = sys._transform1(Mg, cmat)
        cc = ccsd(sys, T=T, mu=mu, iprint=0, max_iter=80, econv=1e-9)
        E, Ecc = cc.run()
        cc._ft_ccsd_lambda()
        cc._g_ft_1rdm()
        cc._g_ft_2rdm()
        P2tot = cc.full_2rdm()
        E2 = 0.25*numpy.einsum('pqrs,rspq->', P2tot, Mg)
        out = E2

        d = 5e-4
        sysf = FakeHubbardSystem(sys, M=d*Mg)
        ccf = ccsd(sysf, T=T, mu=mu, iprint=0, max_iter=80, econv=1e-9)
        Ef, Eccf = ccf.run()

        sysb = FakeHubbardSystem(sys, M=-d*Mg)
        ccb = ccsd(sysb, T=T, mu=mu, iprint=0, max_iter=80, econv=1e-9)
        Eb, Eccb = ccb.run()
        ref = (Ef - Eb)/(2*d)
        error = abs(ref - out) / abs(ref)
        self.assertTrue(error < 1e-6, "Error: {}".format(error))

    def test_Be_gen(self):
        T = 0.8
        beta = 1.0/T
        mu = 0.04
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        sys = SCFSystem(m, T, mu, orbtype='g')
        ccsdT = ccsd(sys, T=T, mu=mu, iprint=0)
        ccsdT.run()
        ccsdT._ft_ccsd_lambda()
        ccsdT._g_ft_2rdm()
        F, I = cc_utils.ft_integrals(sys, sys.g_energies_tot(), beta, mu)
        ref = (0.25/beta)*einsum('cdab,abcd->', ccsdT.P2[0], I.vvvv)
        ref += (0.5/beta)*einsum('ciab,abci->', ccsdT.P2[1], I.vvvo)
        ref += (0.5/beta)*einsum('bcai,aibc->', ccsdT.P2[2], I.vovv)
        ref += (0.25/beta)*einsum('ijab,abij->', ccsdT.P2[3], I.vvoo)
        ref += (1.0/beta)*einsum('bjai,aibj->', ccsdT.P2[4], I.vovo)
        ref += (0.25/beta)*einsum('abij,ijab->', ccsdT.P2[5], I.oovv)
        ref += (0.5/beta)*einsum('jkai,aijk->', ccsdT.P2[6], I.vooo)
        ref += (0.5/beta)*einsum('kaij,ijka->', ccsdT.P2[7], I.ooov)
        ref += (0.25/beta)*einsum('klij,ijkl->', ccsdT.P2[8], I.oooo)
        Inew = sys.g_aint_tot()
        out1 = (0.25)*einsum('pqrs,rspq->', ccsdT.n2rdm, Inew)
        Inew = sys.g_int_tot()
        out2 = (0.5)*einsum('pqrs,rspq->', ccsdT.n2rdm, Inew)
        diff1 = abs(ref - out1)
        diff2 = abs(ref - out2)
        self.assertTrue(diff1 < self.thresh, "Error in 2rdm: {}".format(diff1))
        self.assertTrue(diff2 < self.thresh, "Error in 2rdm: {}".format(diff2))

    def test_Be_gen_active(self):
        T = 0.05
        beta = 1.0/T
        mu = 0.04
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-11
        m.scf()
        sys = SCFSystem(m, T, mu, orbtype='g')
        athresh = 1e-20
        ccsdT = ccsd(sys, T=T, mu=mu, iprint=0, damp=0.16, tconv=1e-10,
                     athresh=athresh, ngrid=40, max_iter=80)
        ccsdT.run()
        ccsdT._ft_ccsd_lambda()
        ccsdT._g_ft_2rdm()
        en = sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)
        focc = [x for x in fo if x > athresh]
        fvir = [x for x in fv if x > athresh]
        iocc = [i for i, x in enumerate(fo) if x > athresh]
        ivir = [i for i, x in enumerate(fv) if x > athresh]
        F, I = cc_utils.ft_active_integrals(sys, en, focc, fvir, iocc, ivir)
        ref = (0.25/beta)*einsum('cdab,abcd->', ccsdT.P2[0], I.vvvv)
        ref += (0.5/beta)*einsum('ciab,abci->', ccsdT.P2[1], I.vvvo)
        ref += (0.5/beta)*einsum('bcai,aibc->', ccsdT.P2[2], I.vovv)
        ref += (0.25/beta)*einsum('ijab,abij->', ccsdT.P2[3], I.vvoo)
        ref += (1.0/beta)*einsum('bjai,aibj->', ccsdT.P2[4], I.vovo)
        ref += (0.25/beta)*einsum('abij,ijab->', ccsdT.P2[5], I.oovv)
        ref += (0.5/beta)*einsum('jkai,aijk->', ccsdT.P2[6], I.vooo)
        ref += (0.5/beta)*einsum('kaij,ijka->', ccsdT.P2[7], I.ooov)
        ref += (0.25/beta)*einsum('klij,ijkl->', ccsdT.P2[8], I.oooo)
        Inew = sys.g_aint_tot()
        out1 = (0.25)*einsum('pqrs,rspq->', ccsdT.n2rdm, Inew)
        Inew = sys.g_int_tot()
        out2 = (0.5)*einsum('pqrs,rspq->', ccsdT.n2rdm, Inew)
        diff1 = abs(ref - out1)/abs(ref)
        diff2 = abs(ref - out2)/abs(ref)
        self.assertTrue(diff1 < self.thresh, "Error in 2rdm: {}".format(diff1))
        self.assertTrue(diff2 < self.thresh, "Error in 2rdm: {}".format(diff2))

    def test_Be(self):
        T = 0.8
        mu = 0.04
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-11
        m.scf()
        sys = SCFSystem(m, T, mu, orbtype='g')
        ccsdT = ccsd(sys, T=T, mu=mu, iprint=0, tconv=1e-10)
        ccsdT.run()
        ccsdT._ft_ccsd_lambda()
        ccsdT._g_ft_2rdm()
        P2g = ccsdT.n2rdm

        sys = SCFSystem(m, T, mu, orbtype='u')
        na = sys.u_energies_tot()[0].shape[0]
        ccsdT = ccsd(sys, T=T, mu=mu, iprint=0, tconv=1e-10)
        ccsdT.run()
        ccsdT._ft_uccsd_lambda()
        ccsdT._u_ft_2rdm()
        P2u = ccsdT.n2rdm

        # aaaa block
        diff = numpy.linalg.norm(
            P2u[0] - P2g[:na, :na, :na, :na])/numpy.linalg.norm(P2u[0])
        self.assertTrue(
            diff < self.thresh, "Error in 2rdm(aaaa): {}".format(diff))

        # bbbb block
        diff = numpy.linalg.norm(
            P2u[1] - P2g[na:, na:, na:, na:])/numpy.linalg.norm(P2u[1])
        self.assertTrue(
            diff < self.thresh, "Error in 2rdm(bbbb): {}".format(diff))

        # abab block
        P2gab = P2g[:na, na:, :na, na:]
        diff = numpy.linalg.norm(
            P2u[2] - P2gab)/numpy.linalg.norm(P2u[2])
        self.assertTrue(
            diff < self.thresh, "Error in 2rdm(abab): {}".format(diff))

    def test_Be_active(self):
        T = 0.05
        mu = 0.04
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        athresh = 1e-20
        ng = 40
        m.conv_tol = 1e-11
        m.scf()
        sys = SCFSystem(m, T, mu, orbtype='g')
        ccsdT = ccsd(sys, T=T, mu=mu, iprint=0, tconv=1e-11,
                     athresh=athresh, ngrid=ng, damp=0.16, max_iter=80)
        ccsdT.run()
        ccsdT._ft_ccsd_lambda()
        ccsdT._g_ft_2rdm()
        P2g = ccsdT.n2rdm

        sys = SCFSystem(m, T, mu, orbtype='u')
        na = sys.u_energies_tot()[0].shape[0]
        ccsdT = ccsd(sys, T=T, mu=mu, iprint=0, tconv=1e-11,
                     athresh=athresh, ngrid=ng, damp=0.16, max_iter=80)
        ccsdT.run()
        ccsdT._ft_uccsd_lambda()
        ccsdT._u_ft_2rdm()
        P2u = ccsdT.n2rdm

        # aaaa block
        diff = numpy.linalg.norm(
            P2u[0] - P2g[:na, :na, :na, :na])/numpy.linalg.norm(P2u[0])
        self.assertTrue(diff < 1e-12, "Error in 2rdm(aaaa): {}".format(diff))

        # bbbb block
        diff = numpy.linalg.norm(
            P2u[1] - P2g[na:, na:, na:, na:])/numpy.linalg.norm(P2u[1])
        self.assertTrue(diff < 1e-12, "Error in 2rdm(bbbb): {}".format(diff))

        # abab block
        P2gab = P2g[:na, na:, :na, na:]
        diff = numpy.linalg.norm(
            P2u[2] - P2gab)/numpy.linalg.norm(P2u[2])
        self.assertTrue(diff < 1e-12, "Error in 2rdm(abab): {}".format(diff))

    def test_Be_active_full(self):
        T = 0.05
        mu = 0.04
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        athresh = 1e-20
        ng = 40
        m.conv_tol = 1e-12
        m.scf()
        sys = SCFSystem(m, T, mu, orbtype='g')
        ccsdT = ccsd(sys, T=T, mu=mu, iprint=0, tconv=1e-11,
                     athresh=athresh, ngrid=ng, damp=0.16, max_iter=80)
        ccsdT.run()
        P2g = ccsdT.full_2rdm()

        sys = SCFSystem(m, T, mu, orbtype='u')
        na = sys.u_energies_tot()[0].shape[0]
        ccsdT = ccsd(sys, T=T, mu=mu, iprint=0, tconv=1e-11,
                     athresh=athresh, ngrid=ng, damp=0.16, max_iter=80)
        ccsdT.run()
        P2u = ccsdT.full_2rdm()

        # aaaa block
        diff = numpy.linalg.norm(
            P2u[0] - P2g[:na, :na, :na, :na])/numpy.linalg.norm(P2u[0])
        self.assertTrue(diff < 1e-12, "Error in 2rdm(aaaa): {}".format(diff))

        # bbbb block
        diff = numpy.linalg.norm(
            P2u[1] - P2g[na:, na:, na:, na:])/numpy.linalg.norm(P2u[1])
        self.assertTrue(diff < 1e-12, "Error in 2rdm(bbbb): {}".format(diff))

        # abab block
        P2gab = P2g[:na, na:, :na, na:]
        diff = numpy.linalg.norm(
            P2u[2] - P2gab)/numpy.linalg.norm(P2u[2])
        self.assertTrue(diff < 1e-12, "Error in 2rdm(abab): {}".format(diff))


if __name__ == '__main__':
    unittest.main()
