import unittest
import numpy
from pyscf import gto, scf
from kelvin.td_ccsd import TDCCSD
from kelvin.ccsd import ccsd
from kelvin.scf_system import scf_system
from kelvin import quadrature


class TDCCSDLambdaTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_Be_rk4_omega(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        T = 0.5
        mu = 0.0
        sys = scf_system(m, T, mu, orbtype='g')
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=120)
        Eref,Eccref = tdccsdT.run()
        Eout,Eccout = tdccsdT._ccsd_lambda()
        diff = abs(Eccref - Eccout)
        error = "Expected: {}  Actual: {}".format(Eccref, Eccout)
        self.assertTrue(diff < 1e-6, error)

    def test_Be_omega_active(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        T = 0.05
        mu = 0.0
        sys = scf_system(m, T, mu, orbtype='g')
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=160, athresh=1e-20, iprint=0, saveT=True)
        Eref,Eccref = tdccsdT.run()
        Eout,Eccout = tdccsdT._ccsd_lambda()
        diff = abs(Eccref - Eccout)
        error = "Expected: {}  Actual: {}".format(Eccref, Eccout)
        self.assertTrue(diff < 1e-6, error)

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
        sys = scf_system(m, T, mu, orbtype='g')

        # compute \bar{Lambda} at \tau = 0
        ccsdT = ccsd(sys, T=T, mu=mu, ngrid=80, iprint=0, quad="mid")
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
        prop = {"tprop": "rk1", "lprop": "rk1"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=160)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._ccsd_lambda()
        d1 = numpy.linalg.norm(tdccsdT.L1 - L1new[0])/numpy.sqrt(L1new[0].size)
        d2 = numpy.linalg.norm(tdccsdT.L2 - L2new[0])/numpy.sqrt(L2new[0].size)
        error1 = "Difference in L1: {}".format(d1)
        error2 = "Difference in L2: {}".format(d2)
        self.assertTrue(d1 < 1e-3, error1)
        self.assertTrue(d2 < 1e-3, error2)

    def test_Be_rk4(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        T = 2.0
        mu = 0.0
        sys = scf_system(m, T, mu, orbtype='g')

        # compute \bar{Lambda} at \tau = 0
        ccsdT = ccsd(sys, T=T, mu=mu, ngrid=320, iprint=0, quad="mid")
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
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=80)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._ccsd_lambda()
        d1 = numpy.linalg.norm(tdccsdT.L1 - L1new[0])/numpy.sqrt(L1new[0].size)
        d2 = numpy.linalg.norm(tdccsdT.L2 - L2new[0])/numpy.sqrt(L2new[0].size)
        error1 = "Difference in L1: {}".format(d1)
        error2 = "Difference in L2: {}".format(d2)
        self.assertTrue(d1 < 5e-5, error1)
        self.assertTrue(d2 < 5e-5, error2)

    def test_Be_cn(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        T = 2.0
        mu = 0.0
        sys = scf_system(m, T, mu, orbtype='g')

        # compute \bar{Lambda} at \tau = 0
        ccsdT = ccsd(sys, T=T, mu=mu, ngrid=320, iprint=0, quad="mid")
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
        prop = {"tprop": "cn", "lprop": "cn", "max_iter": 200, "damp": 0.3, "thresh": 1e-5}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=80)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._ccsd_lambda()
        d1 = numpy.linalg.norm(tdccsdT.L1 - L1new[0])/numpy.sqrt(L1new[0].size)
        d2 = numpy.linalg.norm(tdccsdT.L2 - L2new[0])/numpy.sqrt(L2new[0].size)
        error1 = "Difference in L1: {}".format(d1)
        error2 = "Difference in L2: {}".format(d2)
        self.assertTrue(d1 < 5e-5, error1)
        self.assertTrue(d2 < 2e-4, error2)

    def test_Be_rk124(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        T = 0.5
        mu = 0.0
        sys = scf_system(m, T, mu, orbtype='g')

        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT4 = TDCCSD(sys, prop, T=T, mu=mu, ngrid=320)
        Eout,Eccout = tdccsdT4.run()
        Etmp,Ecctmp = tdccsdT4._ccsd_lambda()
        prop = {"tprop": "rk2", "lprop": "rk2"}
        tdccsdT2 = TDCCSD(sys, prop, T=T, mu=mu, ngrid=2400)
        Eout,Eccout = tdccsdT2.run()
        Etmp,Ecctmp = tdccsdT2._ccsd_lambda()
        prop = {"tprop": "rk1", "lprop": "rk1"}
        tdccsdT1 = TDCCSD(sys, prop, T=T, mu=mu, ngrid=2400)
        Eout,Eccout = tdccsdT1.run()
        Etmp,Ecctmp = tdccsdT1._ccsd_lambda()
        d1_14 = numpy.linalg.norm(tdccsdT4.L1 - tdccsdT1.L1)/numpy.sqrt(tdccsdT1.L1.size)
        d2_14 = numpy.linalg.norm(tdccsdT4.L2 - tdccsdT1.L2)/numpy.sqrt(tdccsdT1.L2.size)
        error1 = "Difference in 1-4 L1: {}".format(d1_14)
        error2 = "Difference in 1-4 L2: {}".format(d2_14)
        self.assertTrue(d1_14 < 1e-1, error1)
        self.assertTrue(d2_14 < 1e-1, error2)

        d1_24 = numpy.linalg.norm(tdccsdT4.L1 - tdccsdT2.L1)/numpy.sqrt(tdccsdT2.L1.size)
        d2_24 = numpy.linalg.norm(tdccsdT4.L2 - tdccsdT2.L2)/numpy.sqrt(tdccsdT2.L2.size)
        error1 = "Difference in 2-4 L1: {}".format(d1_24)
        error2 = "Difference in 2-4 L2: {}".format(d2_24)
        self.assertTrue(d1_24 < 1e-4, error1)
        self.assertTrue(d2_24 < 2e-4, error2)

    def test_Be_tsave(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        T = 0.5
        mu = 0.0
        sys = scf_system(m, T, mu, orbtype='g')

        prop = {"tprop": "rk4", "lprop": "rk4"}
        ccP = TDCCSD(sys, prop, T=T, mu=mu, ngrid=320)
        Eout,Eccout = ccP.run()
        Etmp,Ecctmp = ccP._ccsd_lambda()
        ccS = TDCCSD(sys, prop, T=T, mu=mu, ngrid=320, saveT=True)
        Eout,Eccout = ccS.run()
        Etmp,Ecctmp = ccS._ccsd_lambda()
        d1 = numpy.linalg.norm(ccS.L1 - ccP.L1)/numpy.sqrt(ccS.L1.size)
        d2 = numpy.linalg.norm(ccS.L2 - ccP.L2)/numpy.sqrt(ccS.L2.size)
        error1 = "Difference in 1-4 L1: {}".format(d1)
        error2 = "Difference in 1-4 L2: {}".format(d2)
        self.assertTrue(d1 < 1e-8, error1)
        self.assertTrue(d2 < 1e-8, error2)

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
        sys = scf_system(m, T, mu, orbtype='g')
        ea,eb = sys.u_energies_tot()
        na = ea.shape[0]
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=160)
        Etmp,Ecctmp = tdccsdT.run()
        Eref,Eccref = tdccsdT._ccsd_lambda()
        l1aref = tdccsdT.L1[:na,:na]
        l1bref = tdccsdT.L1[na:,na:]
        l2aaref = tdccsdT.L2[:na,:na,:na,:na]
        l2abref = tdccsdT.L2[:na,na:,:na,na:]
        l2bbref = tdccsdT.L2[na:,na:,na:,na:]
        # compute normal-order 1-rdm from propagation
        sys = scf_system(m, T, mu, orbtype='u')
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=160)
        Etmp,Ecctmp = tdccsdT.run()
        Eout,Eccout = tdccsdT._uccsd_lambda()
        d1a = numpy.linalg.norm(l1aref - tdccsdT.L1[0])/numpy.linalg.norm(l1aref)
        d1b = numpy.linalg.norm(l1bref - tdccsdT.L1[1])/numpy.linalg.norm(l1bref)
        d2aa = numpy.linalg.norm(l2aaref - tdccsdT.L2[0])/numpy.linalg.norm(l2aaref)
        d2ab = numpy.linalg.norm(l2abref - tdccsdT.L2[1])/numpy.linalg.norm(l2abref)
        d2bb = numpy.linalg.norm(l2bbref - tdccsdT.L2[2])/numpy.linalg.norm(l2bbref)
        e1a = "Error in L1a: {}".format(d1a)
        e1b = "Error in L1b: {}".format(d1b)
        e2aa = "Error in L2aa: {}".format(d2aa)
        e2ab = "Error in L2ab: {}".format(d2ab)
        e2bb = "Error in L2bb: {}".format(d2bb)
        self.assertTrue(d1a < 1e-13, e1a)
        self.assertTrue(d1b < 1e-13, e1b)
        self.assertTrue(d2aa < 1e-13, e2aa)
        self.assertTrue(d2ab < 1e-13, e2ab)
        self.assertTrue(d2bb < 1e-13, e2bb)

    def test_Be_Lsave(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        T = 0.5
        mu = 0.0
        sys = scf_system(m, T, mu, orbtype='g')

        prop = {"tprop": "rk4", "lprop": "rk4"}
        ccP = TDCCSD(sys, prop, T=T, mu=mu, ngrid=80)
        Eout,Eccout = ccP.run()
        Etmp,Ecctmp = ccP._ccsd_lambda()
        piaref = ccP.dia
        ccS = TDCCSD(sys, prop, T=T, mu=mu, ngrid=80, saveL=True)
        Eout,Eccout = ccS.run()
        Etmp,Ecctmp = ccS._ccsd_lambda()
        g = ccS.g
        piaout = numpy.einsum('xia,x->ia', numpy.asarray(ccS.L1), g)
        d1 = numpy.linalg.norm(piaout - piaref)/numpy.sqrt(piaout.size)
        error1 = "Difference in 1-4 L1: {}".format(d1)
        self.assertTrue(d1 < 1e-8, error1)

    def test_Be_ccd(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        T = 2.0
        mu = 0.0
        sys = scf_system(m, T, mu, orbtype='g')

        # compute \bar{Lambda} at \tau = 0
        ccsdT = ccsd(sys, T=T, mu=mu, ngrid=320, iprint=0, quad="mid", singles=False)
        Eref,Eccref = ccsdT.run()
        ccsdT._ft_ccsd_lambda()
        ng = ccsdT.ngrid
        L2 = ccsdT.L2
        ti = ccsdT.ti
        g = ccsdT.g
        G = ccsdT.G
        en = sys.g_energies_tot()
        D2 = en[:,None,None,None] + en[None,:,None,None] \
            - en[None,None,:,None] - en[None,None,None,:]
        L2new = quadrature.int_L2(ng,L2,ti,D2,g,G)

        # compute \bar{Lambda} from propagation using rk1
        prop = {"tprop": "rk4", "lprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=80, singles=False)
        Eout,Eccout = tdccsdT.run()
        Etmp,Ecctmp = tdccsdT._ccsd_lambda()
        d2 = numpy.linalg.norm(tdccsdT.L2 - L2new[0])/numpy.sqrt(L2new[0].size)
        error2 = "Difference in L2: {}".format(d2)
        self.assertTrue(d2 < 5e-5, error2)


if __name__ == '__main__':
    unittest.main()
