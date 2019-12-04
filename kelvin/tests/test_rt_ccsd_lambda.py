import unittest
import numpy
from pyscf import gto, scf
from kelvin.rt_ccsd import RTCCSD
from kelvin.ccsd import ccsd
from kelvin.scf_system import scf_system
from kelvin import quadrature

class RTCCSDLambdaTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_Be_rk4_omega(self):
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
        rtccsdT = RTCCSD(sys, T=T, mu=mu, ngrid=120, prop="rk4")
        Eref,Eccref = rtccsdT.run()
        Eout,Eccout = rtccsdT._ccsd_lambda()
        diff = abs(Eccref - Eccout)
        error = "Expected: {}  Actual: {}".format(Eccref,Eccout)
        self.assertTrue(diff < 1e-6,error)

    def test_Be_omega_active(self):
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
        rtccsdT = RTCCSD(sys, T=T, mu=mu, ngrid=160, prop="rk4", athresh=1e-20, iprint=0, saveT=True)
        Eref,Eccref = rtccsdT.run()
        Eout,Eccout = rtccsdT._ccsd_lambda()
        diff = abs(Eccref - Eccout)
        error = "Expected: {}  Actual: {}".format(Eccref,Eccout)
        self.assertTrue(diff < 1e-6,error)

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

        # compute \bar{Lambda} at \tau = 0
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=80,iprint=0,quad="mid")
        Eref,Eccref = ccsdT.run()
        ccsdT._ft_ccsd_lambda()
        ng = ccsdT.ngrid
        L1 = ccsdT.L1
        L2 = ccsdT.L2
        ti = ccsdT.ti
        g = ccsdT.g
        G = ccsdT.G
        en = sys.g_energies_tot()
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
                - en[None,None,:,None] - en[None,None,None,:]
        L1new = quadrature.int_L1(ng,L1,ti,D1,g,G)
        L2new = quadrature.int_L2(ng,L2,ti,D2,g,G)

        # compute \bar{Lambda} from propagation using rk1
        rtccsdT = RTCCSD(sys, T=T, mu=mu, ngrid=160, prop="rk1")
        Eout,Eccout = rtccsdT.run()
        Etmp,Ecctmp = rtccsdT._ccsd_lambda()
        d1 = numpy.linalg.norm(rtccsdT.L1 - L1new[0])/numpy.sqrt(L1new[0].size)
        d2 = numpy.linalg.norm(rtccsdT.L2 - L2new[0])/numpy.sqrt(L2new[0].size)
        error1 = "Difference in L1: {}".format(d1)
        error2 = "Difference in L2: {}".format(d2)
        self.assertTrue(d1 < 1e-3,error1)
        self.assertTrue(d2 < 1e-3,error2)

    def test_Be_rk4(self):
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

        # compute \bar{Lambda} at \tau = 0
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=320,iprint=0,quad="mid")
        Eref,Eccref = ccsdT.run()
        ccsdT._ft_ccsd_lambda()
        ng = ccsdT.ngrid
        L1 = ccsdT.L1
        L2 = ccsdT.L2
        ti = ccsdT.ti
        g = ccsdT.g
        G = ccsdT.G
        en = sys.g_energies_tot()
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
                - en[None,None,:,None] - en[None,None,None,:]
        L1new = quadrature.int_L1(ng,L1,ti,D1,g,G)
        L2new = quadrature.int_L2(ng,L2,ti,D2,g,G)

        # compute \bar{Lambda} from propagation using rk1
        rtccsdT = RTCCSD(sys, T=T, mu=mu, ngrid=80, prop="rk4")
        Eout,Eccout = rtccsdT.run()
        Etmp,Ecctmp = rtccsdT._ccsd_lambda()
        d1 = numpy.linalg.norm(rtccsdT.L1 - L1new[0])/numpy.sqrt(L1new[0].size)
        d2 = numpy.linalg.norm(rtccsdT.L2 - L2new[0])/numpy.sqrt(L2new[0].size)
        error1 = "Difference in L1: {}".format(d1)
        error2 = "Difference in L2: {}".format(d2)
        self.assertTrue(d1 < 5e-5,error1)
        self.assertTrue(d2 < 5e-5,error2)

    def test_Be_cn(self):
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

        # compute \bar{Lambda} at \tau = 0
        ccsdT = ccsd(sys,T=T,mu=mu,ngrid=320,iprint=0,quad="mid")
        Eref,Eccref = ccsdT.run()
        ccsdT._ft_ccsd_lambda()
        ng = ccsdT.ngrid
        L1 = ccsdT.L1
        L2 = ccsdT.L2
        ti = ccsdT.ti
        g = ccsdT.g
        G = ccsdT.G
        en = sys.g_energies_tot()
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
                - en[None,None,:,None] - en[None,None,None,:]
        L1new = quadrature.int_L1(ng,L1,ti,D1,g,G)
        L2new = quadrature.int_L2(ng,L2,ti,D2,g,G)

        # compute \bar{Lambda} from propagation using rk1
        rtccsdT = RTCCSD(sys, T=T, mu=mu, ngrid=80, prop="cn")
        Eout,Eccout = rtccsdT.run()
        Etmp,Ecctmp = rtccsdT._ccsd_lambda()
        d1 = numpy.linalg.norm(rtccsdT.L1 - L1new[0])/numpy.sqrt(L1new[0].size)
        d2 = numpy.linalg.norm(rtccsdT.L2 - L2new[0])/numpy.sqrt(L2new[0].size)
        error1 = "Difference in L1: {}".format(d1)
        error2 = "Difference in L2: {}".format(d2)
        self.assertTrue(d1 < 5e-5,error1)

    def test_Be_rk124(self):
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

        rtccsdT4 = RTCCSD(sys, T=T, mu=mu, ngrid=320, prop="rk4")
        Eout,Eccout = rtccsdT4.run()
        Etmp,Ecctmp = rtccsdT4._ccsd_lambda()
        rtccsdT2 = RTCCSD(sys, T=T, mu=mu, ngrid=2400, prop="rk2")
        Eout,Eccout = rtccsdT2.run()
        Etmp,Ecctmp = rtccsdT2._ccsd_lambda()
        rtccsdT1 = RTCCSD(sys, T=T, mu=mu, ngrid=2400, prop="rk1")
        Eout,Eccout = rtccsdT1.run()
        Etmp,Ecctmp = rtccsdT1._ccsd_lambda()
        d1_14 = numpy.linalg.norm(rtccsdT4.L1 - rtccsdT1.L1)/numpy.sqrt(rtccsdT1.L1.size)
        d2_14 = numpy.linalg.norm(rtccsdT4.L2 - rtccsdT1.L2)/numpy.sqrt(rtccsdT1.L2.size)
        error1 = "Difference in 1-4 L1: {}".format(d1_14)
        error2 = "Difference in 1-4 L2: {}".format(d2_14)
        self.assertTrue(d1_14 < 1e-1,error1)
        self.assertTrue(d2_14 < 1e-1,error2)

        d1_24 = numpy.linalg.norm(rtccsdT4.L1 - rtccsdT2.L1)/numpy.sqrt(rtccsdT2.L1.size)
        d2_24 = numpy.linalg.norm(rtccsdT4.L2 - rtccsdT2.L2)/numpy.sqrt(rtccsdT2.L2.size)
        error1 = "Difference in 2-4 L1: {}".format(d1_24)
        error2 = "Difference in 2-4 L2: {}".format(d2_24)
        self.assertTrue(d1_24 < 1e-4,error1)
        self.assertTrue(d2_24 < 2e-4,error2)

    def test_Be_tsave(self):
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

        ccP = RTCCSD(sys, T=T, mu=mu, ngrid=320, prop="rk4")
        Eout,Eccout = ccP.run()
        Etmp,Ecctmp = ccP._ccsd_lambda()
        ccS = RTCCSD(sys, T=T, mu=mu, ngrid=320, prop="rk4", saveT=True)
        Eout,Eccout = ccS.run()
        Etmp,Ecctmp = ccS._ccsd_lambda()
        d1 = numpy.linalg.norm(ccS.L1 - ccP.L1)/numpy.sqrt(ccS.L1.size)
        d2 = numpy.linalg.norm(ccS.L2 - ccP.L2)/numpy.sqrt(ccS.L2.size)
        error1 = "Difference in 1-4 L1: {}".format(d1)
        error2 = "Difference in 1-4 L2: {}".format(d2)
        self.assertTrue(d1 < 1e-8,error1)
        self.assertTrue(d2 < 1e-8,error2)

if __name__ == '__main__':
    unittest.main()
