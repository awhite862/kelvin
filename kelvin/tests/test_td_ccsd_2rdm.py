import unittest
import numpy
from pyscf import gto, scf
from kelvin.td_ccsd import TDCCSD
from kelvin.ccsd import ccsd
from kelvin.scf_system import scf_system

class TDCCSD2RDMTest(unittest.TestCase):
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
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=220,iprint=0)
        Eref,Eccref = ccsdT.run()
        ccsdT._g_ft_2rdm()

        # compute normal-order 1-rdm from propagation
        prop = {"tprop" : "rk4", "lprop" : "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=320)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._ccsd_lambda(rdm2=True)
        names = ["cdab", "ciab", "bcai", "ijab", "bjai", "abij", "jkai", "kaij", "klij"]
        diffs = [numpy.linalg.norm(r - o)/numpy.linalg.norm(r) for r,o in zip(ccsdT.P2, tdccsdT.P2)]
        errs = ["Difference in {}: {}".format(n,d) for n,d in zip(names, diffs)]
        for d,e in zip(diffs,errs):
            self.assertTrue(d < 5e-5, e) 

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
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=220,iprint=0,athresh=1e-20)
        Eref,Eccref = ccsdT.run()
        ccsdT._g_ft_2rdm()

        # compute normal-order 1-rdm from propagation
        prop = {"tprop" : "rk4", "lprop" : "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=320, athresh=1e-20, saveT=True)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._ccsd_lambda(rdm2=True)
        names = ["cdab", "ciab", "bcai", "ijab", "bjai", "abij", "jkai", "kaij", "klij"]
        diffs = [numpy.linalg.norm(r - o)/numpy.linalg.norm(r) for r,o in zip(ccsdT.P2, tdccsdT.P2)]
        errs = ["Difference in {}: {}".format(n,d) for n,d in zip(names, diffs)]
        for d,e in zip(diffs,errs):
            self.assertTrue(d < 1e-4, e) 

if __name__ == '__main__':
    unittest.main()
