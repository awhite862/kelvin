import unittest
import numpy
from pyscf import gto, scf
from kelvin.td_ccsd import TDCCSD
from kelvin.ccsd import ccsd
from kelvin.scf_system import SCFSystem


class TDCCSD2RDMTest(unittest.TestCase):
    def test_Be_rk4(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        T = 0.5
        mu = 0.0
        sys = SCFSystem(m, T, mu, orbtype='g')

        # compute normal-ordered 1-rdm
        ccsdT = ccsd(sys, T=T, mu=mu, ngrid=220, iprint=0)
        Eref, Eccref = ccsdT.run()
        ccsdT._g_ft_2rdm()

        # compute normal-order 1-rdm from propagation
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=320)
        Eout, Eccout = tdccsdT.run()
        Etmp, Ecctmp = tdccsdT._ccsd_lambda(rdm2=True)
        names = ["cdab", "ciab", "bcai", "ijab", "bjai", "abij", "jkai", "kaij", "klij"]
        diffs = [numpy.linalg.norm(r - o)/numpy.linalg.norm(r) for r,o in zip(ccsdT.P2, tdccsdT.P2)]
        errs = ["Difference in {}: {}".format(n, d) for n,d in zip(names, diffs)]
        for d,e in zip(diffs, errs):
            self.assertTrue(d < 5e-5, e)

    def test_Be_rk4_active(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        T = 0.05
        mu = 0.0
        sys = SCFSystem(m, T, mu, orbtype='g')

        # compute normal-ordered 1-rdm
        ccsdT = ccsd(sys, T=T, mu=mu, ngrid=220, iprint=0, athresh=1e-20)
        Eref, Eccref = ccsdT.run()
        ccsdT._g_ft_2rdm()

        # compute normal-order 1-rdm from propagation
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=320, athresh=1e-20, saveT=True)
        Eout, Eccout = tdccsdT.run()
        Etmp, Ecctmp = tdccsdT._ccsd_lambda(rdm2=True)
        names = ["cdab", "ciab", "bcai", "ijab", "bjai", "abij", "jkai", "kaij", "klij"]
        diffs = [numpy.linalg.norm(r - o)/numpy.linalg.norm(r) for r,o in zip(ccsdT.P2, tdccsdT.P2)]
        errs = ["Difference in {}: {}".format(n, d) for n,d in zip(names, diffs)]
        for d,e in zip(diffs, errs):
            self.assertTrue(d < 1e-4, e)

    def test_Be_u_vs_g(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        T = 0.5
        mu = 0.0
        # compute normal-order 1-rdm from propagation (g)
        sys = SCFSystem(m, T, mu, orbtype='g')
        prop = {"tprop": "rk4", "lprop": "rk4"}
        ccg = TDCCSD(sys, prop, T=T, mu=mu, ngrid=320)
        Eout, Eccout = ccg.run()
        Etmp, Ecctmp = ccg._ccsd_lambda(rdm2=True)
        sys = SCFSystem(m, T, mu, orbtype='u')
        prop = {"tprop": "rk4", "lprop": "rk4"}
        ccu = TDCCSD(sys, prop, T=T, mu=mu, ngrid=320)
        Eout, Eccout = ccu.run()
        Etmp, Ecctmp = ccu._uccsd_lambda(rdm2=True)
        na = 5
        den = float(numpy.sqrt(na*na*na*na))

        # cdab
        diff = numpy.linalg.norm(ccg.P2[0][:na, :na, :na, :na] - ccu.P2[0][0])/den
        self.assertTrue(diff < 1e-12, "Error in Pcdab: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[0][na:, na:, na:, na:] - ccu.P2[0][1])/den
        self.assertTrue(diff < 1e-12, "Error in PCDAB: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[0][:na, na:, :na, na:] - ccu.P2[0][2])/den
        self.assertTrue(diff < 1e-12, "Error in PcDaB: {}".format(diff))

        # ciab
        diff = numpy.linalg.norm(ccg.P2[1][:na, :na, :na, :na] - ccu.P2[1][0])/den
        self.assertTrue(diff < 1e-12, "Error in Pciab: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[1][na:, na:, na:, na:] - ccu.P2[1][1])/den
        self.assertTrue(diff < 1e-12, "Error in PCIAB: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[1][:na, na:, :na, na:] - ccu.P2[1][2])/den
        self.assertTrue(diff < 1e-12, "Error in PcIaB: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[1][na:, :na, na:, :na] - ccu.P2[1][3])/den
        self.assertTrue(diff < 1e-12, "Error in PCiAb: {}".format(diff))

        # bcai
        diff = numpy.linalg.norm(ccg.P2[2][:na, :na, :na, :na] - ccu.P2[2][0])/den
        self.assertTrue(diff < 1e-12, "Error in Pbcai: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[2][na:, na:, na:, na:] - ccu.P2[2][1])/den
        self.assertTrue(diff < 1e-12, "Error in PBCAI: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[2][:na, na:, :na, na:] - ccu.P2[2][2])/den
        self.assertTrue(diff < 1e-12, "Error in PbCaI: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[2][na:, :na, na:, :na] - ccu.P2[2][3])/den
        self.assertTrue(diff < 1e-12, "Error in PBcAi: {}".format(diff))

        # bjai
        diff = numpy.linalg.norm(ccg.P2[4][:na, :na, :na, :na] - ccu.P2[4][0])/den
        self.assertTrue(diff < 1e-12, "Error in Pbjai: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[4][na:, na:, na:, na:] - ccu.P2[4][1])/den
        self.assertTrue(diff < 1e-12, "Error in PBJAI: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[4][:na, na:, :na, na:] - ccu.P2[4][2])/den
        self.assertTrue(diff < 1e-12, "Error in PbJaI: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[4][:na, na:, na:, :na] - ccu.P2[4][3])/den
        self.assertTrue(diff < 1e-12, "Error in PbJAi: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[4][na:, :na, :na, na:] - ccu.P2[4][4])/den
        self.assertTrue(diff < 1e-12, "Error in PBjaI: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[4][na:, :na, na:, :na] - ccu.P2[4][5])/den
        self.assertTrue(diff < 1e-12, "Error in PBjAi: {}".format(diff))

        # abij
        diff = numpy.linalg.norm(ccg.P2[5][:na, :na, :na, :na] - ccu.P2[5][0])/den
        self.assertTrue(diff < 1e-12, "Error in Pabij: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[5][na:, na:, na:, na:] - ccu.P2[5][1])/den
        self.assertTrue(diff < 1e-12, "Error in PABIJ: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[5][:na, na:, :na, na:] - ccu.P2[5][2])/den
        self.assertTrue(diff < 1e-12, "Error in PaBiJ: {}".format(diff))

        # jkai
        diff = numpy.linalg.norm(ccg.P2[6][:na, :na, :na, :na] - ccu.P2[6][0])/den
        self.assertTrue(diff < 1e-12, "Error in Pjkai: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[6][na:, na:, na:, na:] - ccu.P2[6][1])/den
        self.assertTrue(diff < 1e-12, "Error in PJKAI: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[6][:na, na:, :na, na:] - ccu.P2[6][2])/den
        self.assertTrue(diff < 1e-12, "Error in PjKaI: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[6][na:, :na, na:, :na] - ccu.P2[6][3])/den
        self.assertTrue(diff < 1e-12, "Error in PJiAi: {}".format(diff))

        # kaij
        diff = numpy.linalg.norm(ccg.P2[7][:na, :na, :na, :na] - ccu.P2[7][0])/den
        self.assertTrue(diff < 1e-12, "Error in Pkaij: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[7][na:, na:, na:, na:] - ccu.P2[7][1])/den
        self.assertTrue(diff < 1e-12, "Error in PKAIJ: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[7][:na, na:, :na, na:] - ccu.P2[7][2])/den
        self.assertTrue(diff < 1e-12, "Error in PkAiJ: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[7][na:, :na, na:, :na] - ccu.P2[7][3])/den
        self.assertTrue(diff < 1e-12, "Error in PkAiJ: {}".format(diff))

        # klij
        diff = numpy.linalg.norm(ccg.P2[8][:na, :na, :na, :na] - ccu.P2[8][0])/den
        self.assertTrue(diff < 1e-12, "Error in Pklij: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[8][na:, na:, na:, na:] - ccu.P2[8][1])/den
        self.assertTrue(diff < 1e-12, "Error in PKLIJ: {}".format(diff))

        diff = numpy.linalg.norm(ccg.P2[8][:na, na:, :na, na:] - ccu.P2[8][2])/den
        self.assertTrue(diff < 1e-12, "Error in PkLiJ: {}".format(diff))

    def test_Be_r_vs_u(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        T = 0.5
        mu = 0.0
        # compute normal-order 1/n-rdm from propagation (u)
        sys = SCFSystem(m, T, mu, orbtype='u')
        prop = {"tprop": "rk4", "lprop": "rk4"}
        ccu = TDCCSD(sys, prop, T=T, mu=mu, ngrid=80)
        Eout, Eccout = ccu.run()
        Etmp, Ecctmp = ccu._uccsd_lambda(rdm2=True)
        sys = SCFSystem(m, T, mu, orbtype='r')
        prop = {"tprop": "rk4", "lprop": "rk4"}
        ccr = TDCCSD(sys, prop, T=T, mu=mu, ngrid=80)
        Eout, Eccout = ccr.run()
        Etmp, Ecctmp = ccr._rccsd_lambda(rdm2=True)
        na = 5
        den = float(numpy.sqrt(na*na*na*na))

        # cdab
        diff = numpy.linalg.norm(ccr.P2[0] - ccu.P2[0][2])/den
        self.assertTrue(diff < 1e-12, "Error in Pcdab: {}".format(diff))

        # ciab
        diff = numpy.linalg.norm(ccr.P2[1] - ccu.P2[1][2])/den
        self.assertTrue(diff < 1e-12, "Error in Pciab: {}".format(diff))

        # bcai
        diff = numpy.linalg.norm(ccr.P2[2] - ccu.P2[2][2])/den
        self.assertTrue(diff < 1e-12, "Error in Pbcai: {}".format(diff))

        # bjai
        diff = numpy.linalg.norm(ccr.P2[4] - ccu.P2[4][2])/den
        self.assertTrue(diff < 1e-12, "Error in Pbjai: {}".format(diff))

        diff = numpy.linalg.norm(ccr.P2[5] + ccu.P2[4][3].transpose((0, 1, 3, 2)))/den
        self.assertTrue(diff < 1e-12, "Error in Pbjia: {}".format(diff))

        # abij
        diff = numpy.linalg.norm(ccr.P2[6] - ccu.P2[5][2])/den
        self.assertTrue(diff < 1e-12, "Error in PaBiJ: {}".format(diff))

        # jkai
        diff = numpy.linalg.norm(ccr.P2[7] - ccu.P2[6][2])/den
        self.assertTrue(diff < 1e-12, "Error in PjKaI: {}".format(diff))

        # kaij
        diff = numpy.linalg.norm(ccr.P2[8] - ccu.P2[7][2])/den
        self.assertTrue(diff < 1e-12, "Error in PkAiJ: {}".format(diff))

        # klij
        diff = numpy.linalg.norm(ccr.P2[9] - ccu.P2[8][2])/den
        self.assertTrue(diff < 1e-12, "Error in Pklij: {}".format(diff))


if __name__ == '__main__':
    unittest.main()
