import unittest
import numpy
from pyscf import gto, scf
from kelvin.td_ccsd import TDCCSD
from kelvin.ccsd import ccsd
from kelvin.scf_system import scf_system

class TDCCSDReldenTest(unittest.TestCase):
    def test_Be_rk4(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        T = 1.0
        mu = 0.0
        sys = scf_system(m,T,mu,orbtype='g')

        # compute normal-ordered 1-rdm 
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=80,iprint=0)
        Eref,Eccref = ccsdT.run()
        ccsdT._g_ft_1rdm()
        ccsdT._g_ft_2rdm()
        ccsdT._g_ft_ron()
        ccsdT._g_ft_rorb()

        # compute normal-order 1-rdm from propagation
        prop = {"tprop" : "rk4", "lprop" : "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=160)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._ccsd_lambda(rdm2=True, erel=True)
        tdccsdT._g_ft_ron()

        diff = numpy.linalg.norm(ccsdT.ron1 - tdccsdT.ron1)/numpy.linalg.norm(ccsdT.ron1)
        self.assertTrue(diff < 1e-12,"Error in ron1: {}".format(diff))
        diff = numpy.linalg.norm(ccsdT.rono - tdccsdT.rono)/numpy.linalg.norm(ccsdT.rono)
        self.assertTrue(diff < 1e-5,"Error in rono: {}".format(diff))
        diff = numpy.linalg.norm(ccsdT.ronv - tdccsdT.ronv)/numpy.linalg.norm(ccsdT.ronv)
        self.assertTrue(diff < 1e-5,"Error in ronv: {}".format(diff))
        diff = numpy.linalg.norm(ccsdT.rorbo - tdccsdT.rorbo)/numpy.sqrt(float(ccsdT.rorbo.size))
        self.assertTrue(diff < 1e-5,"Error in rorbo: {}".format(diff))
        diff = numpy.linalg.norm(ccsdT.rorbv - tdccsdT.rorbv)/numpy.sqrt(float(ccsdT.rorbv.size))
        self.assertTrue(diff < 1e-5,"Error in rorbv: {}".format(diff))

    def test_Be_rk4_active(self):
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
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=120,iprint=0,athresh=1e-20)
        Eref,Eccref = ccsdT.run()
        ccsdT._g_ft_1rdm()
        ccsdT._g_ft_2rdm()
        ccsdT._g_ft_ron()
        ccsdT._g_ft_rorb()

        # compute normal-order 1-rdm from propagation
        prop = {"tprop" : "rk4", "lprop" : "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=440, athresh=1e-20, saveT=True)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._ccsd_lambda(rdm2=True,erel=True)
        tdccsdT._g_ft_ron()

        diff = numpy.linalg.norm(ccsdT.ron1 - tdccsdT.ron1)/numpy.linalg.norm(ccsdT.ron1)
        self.assertTrue(diff < 1e-12,"Error in ron1: {}".format(diff))
        diff = numpy.linalg.norm(ccsdT.rono - tdccsdT.rono)/numpy.linalg.norm(ccsdT.rono)
        self.assertTrue(diff < 1e-2,"Error in rono: {}".format(diff))
        diff = numpy.linalg.norm(ccsdT.ronv - tdccsdT.ronv)/numpy.linalg.norm(ccsdT.ronv)
        self.assertTrue(diff < 1e-2,"Error in ronv: {}".format(diff))
        diff = numpy.linalg.norm(ccsdT.rorbo - tdccsdT.rorbo)/numpy.sqrt(float(ccsdT.rorbo.size))
        self.assertTrue(diff < 1e-5,"Error in rorbo: {}".format(diff))
        diff = numpy.linalg.norm(ccsdT.rorbv - tdccsdT.rorbv)/numpy.sqrt(float(ccsdT.rorbv.size))
        self.assertTrue(diff < 1e-5,"Error in rorbv: {}".format(diff))

if __name__ == '__main__':
    unittest.main()
