import unittest
import numpy
from pyscf import gto, scf
from kelvin.rt_ccsd import RTCCSD
from kelvin.ccsd import ccsd
from kelvin.scf_system import scf_system

class RTCCSDTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-8

    def test_Be_rk1(self):
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
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=40,iprint=0)
        Eref,Eccref = ccsdT.run()
        rtccsdT = RTCCSD(sys, T=T, mu=mu, ngrid=640, prop="rk1")
        Eout,Eccout = rtccsdT.run()
        diff = abs(Eccref - Eccout)
        error = "Expected: {}  Actual: {}".format(Eccref,Eccout)
        self.assertTrue(diff < 1e-3,error)

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
        ng = 40
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=ng,iprint=0)
        Eref,Eccref = ccsdT.run()
        rtccsdT = RTCCSD(sys, T=T, mu=mu, ngrid=160, prop="rk2")
        Eout,Eccout = rtccsdT.run()
        diff = abs(Eccref - Eccout)
        error = "Expected: {}  Actual: {}".format(Eccref,Eccout)
        self.assertTrue(diff < 1e-4,error)

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
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=80,iprint=0)
        Eref,Eccref = ccsdT.run()
        rtccsdT = RTCCSD(sys, T=T, mu=mu, ngrid=80, prop="rk4")
        Eout,Eccout = rtccsdT.run()
        diff = abs(Eccref - Eccout)
        error = "Expected: {}  Actual: {}".format(Eccref,Eccout)
        self.assertTrue(diff < 1e-6,error)

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
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=100,iprint=0,athresh=1e-20)
        Eref,Eccref = ccsdT.run()
        rtccsdT = RTCCSD(sys, T=T, mu=mu, ngrid=160, prop="rk4", athresh=1e-20)
        Eout,Eccout = rtccsdT.run()
        diff = abs(Eccref - Eccout)
        error = "Expected: {}  Actual: {}".format(Eccref,Eccout)
        self.assertTrue(diff < 1e-5,error)

if __name__ == '__main__':
    unittest.main()
