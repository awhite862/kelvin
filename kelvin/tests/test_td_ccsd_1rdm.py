import unittest
import numpy
from pyscf import gto, scf
from kelvin.td_ccsd import TDCCSD
from kelvin.ccsd import ccsd
from kelvin.scf_system import scf_system

class TDCCSD1RDMTest(unittest.TestCase):
    def test_Be_rk1(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        T = 2.0
        mu = 0.0
        sys = scf_system(m,T,mu,orbtype='g')

        # compute normal-ordered 1-rdm 
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=40,iprint=0)
        Eref,Eccref = ccsdT.run()
        ccsdT._g_ft_1rdm()

        # compute normal-order 1-rdm from propagation
        prop = {"tprop" : "rk1", "lprop" : "rk1"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=160)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._ccsd_lambda()
        eia = numpy.linalg.norm(ccsdT.dia - tdccsdT.dia)
        eji = numpy.linalg.norm(ccsdT.dji - tdccsdT.dji)
        eba = numpy.linalg.norm(ccsdT.dba - tdccsdT.dba)
        eai = numpy.linalg.norm(ccsdT.dai - tdccsdT.dai)
        erroria = "Difference in pia: {}".format(eia)
        errorji = "Difference in pji: {}".format(eji)
        errorba = "Difference in pba: {}".format(eba)
        errorai = "Difference in pai: {}".format(eai)
        self.assertTrue(eia < 1e-3,erroria)
        self.assertTrue(eji < 1e-3,errorji)
        self.assertTrue(eba < 1e-3,errorba)
        self.assertTrue(eai < 1e-3,errorai)

    def test_Be_rk2(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        T = 0.5
        mu = 0.0
        sys = scf_system(m,T,mu,orbtype='g')

        # compute normal-ordered 1-rdm 
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=40,iprint=0)
        Eref,Eccref = ccsdT.run()
        ccsdT._g_ft_1rdm()

        # compute normal-order 1-rdm from propagation
        prop = {"tprop" : "rk2", "lprop" : "rk2"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=160)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._ccsd_lambda()
        eia = numpy.linalg.norm(ccsdT.dia - tdccsdT.dia)
        eji = numpy.linalg.norm(ccsdT.dji - tdccsdT.dji)
        eba = numpy.linalg.norm(ccsdT.dba - tdccsdT.dba)
        eai = numpy.linalg.norm(ccsdT.dai - tdccsdT.dai)
        erroria = "Difference in pia: {}".format(eia)
        errorji = "Difference in pji: {}".format(eji)
        errorba = "Difference in pba: {}".format(eba)
        errorai = "Difference in pai: {}".format(eai)
        self.assertTrue(eia < 1e-3,erroria)
        self.assertTrue(eji < 1e-3,errorji)
        self.assertTrue(eba < 1e-3,errorba)
        self.assertTrue(eai < 1e-3,errorai)

    def test_Be_rk4(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        T = 0.5
        mu = 0.0
        sys = scf_system(m,T,mu,orbtype='g')

        # compute normal-ordered 1-rdm 
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=80,iprint=0)
        Eref,Eccref = ccsdT.run()
        ccsdT._g_ft_1rdm()

        # compute normal-order 1-rdm from propagation
        prop = {"tprop" : "rk4", "lprop" : "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=160)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._ccsd_lambda()
        eia = numpy.linalg.norm(ccsdT.dia - tdccsdT.dia)
        eji = numpy.linalg.norm(ccsdT.dji - tdccsdT.dji)
        eba = numpy.linalg.norm(ccsdT.dba - tdccsdT.dba)
        eai = numpy.linalg.norm(ccsdT.dai - tdccsdT.dai)
        erroria = "Difference in pia: {}".format(eia)
        errorji = "Difference in pji: {}".format(eji)
        errorba = "Difference in pba: {}".format(eba)
        errorai = "Difference in pai: {}".format(eai)
        self.assertTrue(eia < 1e-5,erroria)
        self.assertTrue(eji < 1e-5,errorji)
        self.assertTrue(eba < 1e-5,errorba)
        self.assertTrue(eai < 1e-5,errorai)

    def test_Be_active(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        T = 0.05
        mu = 0.0
        sys = scf_system(m,T,mu,orbtype='g')

        # compute normal-ordered 1-rdm 
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=100,iprint=0,damp=0.4,athresh = 1e-20)
        Eref,Eccref = ccsdT.run()
        ccsdT._g_ft_1rdm()

        # compute normal-order 1-rdm from propagation
        prop = {"tprop" : "rk4", "lprop" : "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=640, athresh = 1e-20, saveT=True)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._ccsd_lambda()
        eia = numpy.linalg.norm(ccsdT.dia - tdccsdT.dia)
        eji = numpy.linalg.norm(ccsdT.dji - tdccsdT.dji)
        eba = numpy.linalg.norm(ccsdT.dba - tdccsdT.dba)
        eai = numpy.linalg.norm(ccsdT.dai - tdccsdT.dai)
        erroria = "Difference in pia: {}".format(eia)
        errorji = "Difference in pji: {}".format(eji)
        errorba = "Difference in pba: {}".format(eba)
        errorai = "Difference in pai: {}".format(eai)
        self.assertTrue(eia < 5e-4,erroria)
        self.assertTrue(eji < 5e-4,errorji)
        self.assertTrue(eba < 5e-4,errorba)
        self.assertTrue(eai < 5e-4,errorai)

if __name__ == '__main__':
    unittest.main()
