import unittest
import numpy
from pyscf import gto, scf
from kelvin.td_ccsd import TDCCSD
from kelvin.ccsd import ccsd
from kelvin.scf_system import SCFSystem


class TDCCSD1RDMTest(unittest.TestCase):
    def test_Be_rk1(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        T = 2.0
        mu = 0.0
        sys = SCFSystem(m, T, mu, orbtype='g')

        # compute normal-ordered 1-rdm
        ccsdT = ccsd(sys, T=T, mu=mu, ngrid=25, iprint=0)
        ccsdT.run()
        ccsdT._g_ft_1rdm()

        # compute normal-order 1-rdm from propagation
        prop = {"tprop": "rk1", "lprop": "rk1"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=150)
        tdccsdT.run()
        tdccsdT._ccsd_lambda()
        eia = numpy.linalg.norm(ccsdT.dia - tdccsdT.dia)
        eji = numpy.linalg.norm(ccsdT.dji - tdccsdT.dji)
        eba = numpy.linalg.norm(ccsdT.dba - tdccsdT.dba)
        eai = numpy.linalg.norm(ccsdT.dai - tdccsdT.dai)
        erroria = "Difference in pia: {}".format(eia)
        errorji = "Difference in pji: {}".format(eji)
        errorba = "Difference in pba: {}".format(eba)
        errorai = "Difference in pai: {}".format(eai)
        self.assertTrue(eia < 1e-3, erroria)
        self.assertTrue(eji < 1e-3, errorji)
        self.assertTrue(eba < 1e-3, errorba)
        self.assertTrue(eai < 1e-3, errorai)

    def test_Be_rk2(self):
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
        ccsdT = ccsd(sys, T=T, mu=mu, ngrid=30, iprint=0)
        ccsdT.run()
        ccsdT._g_ft_1rdm()

        # compute normal-order 1-rdm from propagation
        prop = {"tprop": "rk2", "lprop": "rk2"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=150)
        tdccsdT.run()
        tdccsdT._ccsd_lambda()
        eia = numpy.linalg.norm(ccsdT.dia - tdccsdT.dia)
        eji = numpy.linalg.norm(ccsdT.dji - tdccsdT.dji)
        eba = numpy.linalg.norm(ccsdT.dba - tdccsdT.dba)
        eai = numpy.linalg.norm(ccsdT.dai - tdccsdT.dai)
        erroria = "Difference in pia: {}".format(eia)
        errorji = "Difference in pji: {}".format(eji)
        errorba = "Difference in pba: {}".format(eba)
        errorai = "Difference in pai: {}".format(eai)
        self.assertTrue(eia < 1e-3, erroria)
        self.assertTrue(eji < 1e-3, errorji)
        self.assertTrue(eba < 1e-3, errorba)
        self.assertTrue(eai < 1e-3, errorai)

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
        ccsdT = ccsd(sys, T=T, mu=mu, ngrid=70, iprint=0)
        ccsdT.run()
        ccsdT._g_ft_1rdm()

        # compute normal-order 1-rdm from propagation
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=150)
        tdccsdT.run()
        tdccsdT._ccsd_lambda()
        eia = numpy.linalg.norm(ccsdT.dia - tdccsdT.dia)
        eji = numpy.linalg.norm(ccsdT.dji - tdccsdT.dji)
        eba = numpy.linalg.norm(ccsdT.dba - tdccsdT.dba)
        eai = numpy.linalg.norm(ccsdT.dai - tdccsdT.dai)
        erroria = "Difference in pia: {}".format(eia)
        errorji = "Difference in pji: {}".format(eji)
        errorba = "Difference in pba: {}".format(eba)
        errorai = "Difference in pai: {}".format(eai)
        self.assertTrue(eia < 1e-5, erroria)
        self.assertTrue(eji < 1e-5, errorji)
        self.assertTrue(eba < 1e-5, errorba)
        self.assertTrue(eai < 1e-5, errorai)

    def test_Be_active(self):
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
        ccsdT = ccsd(sys, T=T, mu=mu, ngrid=100, iprint=0, damp=0.4, athresh=1e-20)
        ccsdT.run()
        ccsdT._g_ft_1rdm()

        # compute normal-order 1-rdm from propagation
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=400, athresh=1e-20, saveT=True)
        tdccsdT.run()
        tdccsdT._ccsd_lambda()
        eia = numpy.linalg.norm(ccsdT.dia - tdccsdT.dia)
        eji = numpy.linalg.norm(ccsdT.dji - tdccsdT.dji)
        eba = numpy.linalg.norm(ccsdT.dba - tdccsdT.dba)
        eai = numpy.linalg.norm(ccsdT.dai - tdccsdT.dai)
        erroria = "Difference in pia: {}".format(eia)
        errorji = "Difference in pji: {}".format(eji)
        errorba = "Difference in pba: {}".format(eba)
        errorai = "Difference in pai: {}".format(eai)
        self.assertTrue(eia < 5e-4, erroria)
        self.assertTrue(eji < 5e-4, errorji)
        self.assertTrue(eba < 5e-4, errorba)
        self.assertTrue(eai < 5e-4, errorai)

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

        # compute normal-order 1-rdm from propagation
        sys = SCFSystem(m, T, mu, orbtype='g')
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=20)
        tdccsdT.run()
        tdccsdT._ccsd_lambda()
        dia = tdccsdT.dia
        dji = tdccsdT.dji
        dba = tdccsdT.dba
        dai = tdccsdT.dai
        # compute normal-order 1-rdm from propagation
        sys = SCFSystem(m, T, mu, orbtype='u')
        ea, eb = sys.u_energies_tot()
        na = ea.shape[0]
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=20)
        Etmp, Ecctmp = tdccsdT.run()
        Eout, Eccout = tdccsdT._uccsd_lambda()
        eia = numpy.linalg.norm(tdccsdT.dia[0] - dia[:na,:na])
        eIA = numpy.linalg.norm(tdccsdT.dia[1] - dia[na:,na:])
        eji = numpy.linalg.norm(tdccsdT.dji[0] - dji[:na,:na])
        eJI = numpy.linalg.norm(tdccsdT.dji[1] - dji[na:,na:])
        eba = numpy.linalg.norm(tdccsdT.dba[0] - dba[:na,:na])
        eBA = numpy.linalg.norm(tdccsdT.dba[1] - dba[na:,na:])
        eai = numpy.linalg.norm(tdccsdT.dai[0] - dai[:na,:na])
        eAI = numpy.linalg.norm(tdccsdT.dai[1] - dai[na:,na:])
        erroria = "Difference in pia: {}".format(eia)
        errorIA = "Difference in pIA: {}".format(eIA)
        errorji = "Difference in pji: {}".format(eji)
        errorJI = "Difference in pJI: {}".format(eJI)
        errorba = "Difference in pba: {}".format(eba)
        errorBA = "Difference in pBA: {}".format(eBA)
        errorai = "Difference in pai: {}".format(eai)
        errorAI = "Difference in pAI: {}".format(eAI)
        self.assertTrue(eia < 1e-12, erroria)
        self.assertTrue(eia < 1e-12, errorIA)
        self.assertTrue(eji < 1e-12, errorji)
        self.assertTrue(eji < 1e-12, errorJI)
        self.assertTrue(eba < 1e-12, errorba)
        self.assertTrue(eba < 1e-12, errorBA)
        self.assertTrue(eai < 1e-12, errorai)
        self.assertTrue(eai < 1e-12, errorAI)

    def test_Be_u_vs_g_active(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        T = 0.05
        mu = 0.0

        # compute normal-order 1-rdm from propagation
        sys = SCFSystem(m, T, mu, orbtype='g')
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=200, athresh=1e-20, saveT=True)
        tdccsdT.run()
        tdccsdT._ccsd_lambda()
        dia = tdccsdT.dia
        dji = tdccsdT.dji
        dba = tdccsdT.dba
        dai = tdccsdT.dai
        # compute normal-order 1-rdm from propagation
        sys = SCFSystem(m, T, mu, orbtype='u')
        ea, eb = sys.u_energies_tot()
        na = ea.shape[0]
        noa = na
        nva = na - 1
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=200, athresh=1e-20, saveT=True)
        Etmp, Ecctmp = tdccsdT.run()
        Eout, Eccout = tdccsdT._uccsd_lambda()
        eia = numpy.linalg.norm(tdccsdT.dia[0] - dia[:noa,:nva])
        eIA = numpy.linalg.norm(tdccsdT.dia[1] - dia[noa:,nva:])
        eji = numpy.linalg.norm(tdccsdT.dji[0] - dji[:noa,:noa])
        eJI = numpy.linalg.norm(tdccsdT.dji[1] - dji[noa:,noa:])
        eba = numpy.linalg.norm(tdccsdT.dba[0] - dba[:nva,:nva])
        eBA = numpy.linalg.norm(tdccsdT.dba[1] - dba[nva:,nva:])
        eai = numpy.linalg.norm(tdccsdT.dai[0] - dai[:nva,:noa])
        eAI = numpy.linalg.norm(tdccsdT.dai[1] - dai[nva:,noa:])
        erroria = "Difference in pia: {}".format(eia)
        errorIA = "Difference in pIA: {}".format(eIA)
        errorji = "Difference in pji: {}".format(eji)
        errorJI = "Difference in pJI: {}".format(eJI)
        errorba = "Difference in pba: {}".format(eba)
        errorBA = "Difference in pBA: {}".format(eBA)
        errorai = "Difference in pai: {}".format(eai)
        errorAI = "Difference in pAI: {}".format(eAI)
        self.assertTrue(eia < 1e-12, erroria)
        self.assertTrue(eia < 1e-12, errorIA)
        self.assertTrue(eji < 1e-12, errorji)
        self.assertTrue(eji < 1e-12, errorJI)
        self.assertTrue(eba < 1e-12, errorba)
        self.assertTrue(eba < 1e-12, errorBA)
        self.assertTrue(eai < 1e-12, errorai)
        self.assertTrue(eai < 1e-12, errorAI)

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

        # compute normal-order 1-rdm from propagation
        sys = SCFSystem(m, T, mu, orbtype='u')
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=20)
        tdccsdT.run()
        tdccsdT._uccsd_lambda()
        dia = tdccsdT.dia
        dji = tdccsdT.dji
        dba = tdccsdT.dba
        dai = tdccsdT.dai
        # compute normal-order 1-rdm from propagation
        sys = SCFSystem(m, T, mu, orbtype='r')
        ea, eb = sys.u_energies_tot()
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=20)
        Etmp, Ecctmp = tdccsdT.run()
        Eout, Eccout = tdccsdT._rccsd_lambda()
        eia = numpy.linalg.norm(dia[0] - tdccsdT.dia)
        eji = numpy.linalg.norm(dji[0] - tdccsdT.dji)
        eba = numpy.linalg.norm(dba[0] - tdccsdT.dba)
        eai = numpy.linalg.norm(dai[0] - tdccsdT.dai)
        erroria = "Difference in pia: {}".format(eia)
        errorji = "Difference in pji: {}".format(eji)
        errorba = "Difference in pba: {}".format(eba)
        errorai = "Difference in pai: {}".format(eai)
        self.assertTrue(eia < 1e-12, erroria)
        self.assertTrue(eji < 1e-12, errorji)
        self.assertTrue(eba < 1e-12, errorba)
        self.assertTrue(eai < 1e-12, errorai)


if __name__ == '__main__':
    unittest.main()
