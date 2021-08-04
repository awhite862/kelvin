import unittest
from pyscf import gto, scf
from kelvin.td_ccsd import TDCCSD
from kelvin.ccsd import ccsd
from kelvin.scf_system import SCFSystem


class TDCCSDTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-8

    def test_Be_rk1(self):
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
        ccsdT = ccsd(sys, T=T, mu=mu, ngrid=10, iprint=0)
        Eref, Eccref = ccsdT.run()
        prop = {"tprop": "rk1"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=380)
        Eout, Eccout = tdccsdT.run()
        diff = abs(Eccref - Eccout)
        error = "Expected: {}  Actual: {}".format(Eccref, Eccout)
        self.assertTrue(diff < 1e-3, error)

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
        ng = 20
        sys = SCFSystem(m, T, mu, orbtype='g')
        ccsdT = ccsd(sys, T=T, mu=mu, ngrid=ng, iprint=0)
        Eref, Eccref = ccsdT.run()
        prop = {"tprop": "rk2"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=60)
        Eout, Eccout = tdccsdT.run()
        diff = abs(Eccref - Eccout)
        error = "Expected: {}  Actual: {}".format(Eccref, Eccout)
        self.assertTrue(diff < 1e-4, error)

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
        ccsdT = ccsd(sys, T=T, mu=mu, ngrid=70, iprint=0)
        Eref, Eccref = ccsdT.run()
        prop = {"tprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=60)
        Eout, Eccout = tdccsdT.run()
        diff = abs(Eccref - Eccout)
        error = "Expected: {}  Actual: {}".format(Eccref, Eccout)
        self.assertTrue(diff < 1e-6, error)

    def test_Be_cn(self):
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
        prop = {"tprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=160)
        Eref, Eccref = tdccsdT.run()
        prop = {"tprop": "cn", "max_iter": 200, "damp": 0.4, "thresh": 1e-5}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=60, iprint=0)
        Eout, Eccout = tdccsdT.run()
        diff = abs(Eccref - Eccout)
        error = "Expected: {}  Actual: {}".format(Eccref, Eccout)
        self.assertTrue(diff < 1e-4, error)

    def test_Be_am2(self):
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
        prop = {"tprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=20)
        Eref, Eccref = tdccsdT.run()
        prop = {"tprop": "am2", "max_iter": 200, "damp": 0.3, "thresh": 1e-5}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=50, iprint=0)
        Eout, Eccout = tdccsdT.run()
        diff = abs(Eccref - Eccout)
        error = "Expected: {}  Actual: {}".format(Eccref, Eccout)
        self.assertTrue(diff < 1e-4, error)

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
        ccsdT = ccsd(sys, T=T, mu=mu, ngrid=90, iprint=0, athresh=1e-20)
        Eref, Eccref = ccsdT.run()
        prop = {"tprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=100,
                         athresh=1e-20, iprint=0)
        Eout, Eccout = tdccsdT.run()
        diff = abs(Eccref - Eccout)
        error = "Expected: {}  Actual: {}".format(Eccref, Eccout)
        self.assertTrue(diff < 1e-5, error)

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
        sys = SCFSystem(m, T, mu, orbtype='g')
        prop = {"tprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=15)
        Eref, Eccref = tdccsdT.run()
        sys = SCFSystem(m, T, mu, orbtype='u')
        prop = {"tprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=15)
        Eout, Eccout = tdccsdT.run()
        diff = abs(Eccref - Eccout)
        error = "Expected: {}  Actual: {}".format(Eccref, Eccout)
        self.assertTrue(diff < 1e-14, error)

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
        sys = SCFSystem(m, T, mu, orbtype='g')
        prop = {"tprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=100, athresh=1e-20)
        Eref, Eccref = tdccsdT.run()
        sys = SCFSystem(m, T, mu, orbtype='u')
        prop = {"tprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=100, athresh=1e-20)
        Eout, Eccout = tdccsdT.run()
        diff = abs(Eccref - Eccout)
        error = "Expected: {}  Actual: {}".format(Eccref, Eccout)
        self.assertTrue(diff < 1e-5, error)

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
        sys = SCFSystem(m, T, mu, orbtype='u')
        prop = {"tprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=15)
        Eref, Eccref = tdccsdT.run()
        sys = SCFSystem(m, T, mu, orbtype='r')
        prop = {"tprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=15)
        Eout, Eccout = tdccsdT.run()
        diff = abs(Eccref - Eccout)
        error = "Expected: {}  Actual: {}".format(Eccref, Eccout)
        self.assertTrue(diff < 1e-14, error)

    def test_Be_r_vs_u_active(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        T = 0.05
        mu = 0.0
        sys = SCFSystem(m, T, mu, orbtype='u')
        prop = {"tprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=100, athresh=1e-20)
        Eref, Eccref = tdccsdT.run()
        sys = SCFSystem(m, T, mu, orbtype='r')
        prop = {"tprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=100, athresh=1e-20)
        Eout, Eccout = tdccsdT.run()
        diff = abs(Eccref - Eccout)
        error = "Expected: {}  Actual: {}".format(Eccref, Eccout)
        self.assertTrue(diff < 1e-5, error)

    def test_Be_ccd(self):
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
        ccsdT = ccsd(sys, T=T, mu=mu, ngrid=50, iprint=0, singles=False)
        Eref, Eccref = ccsdT.run()
        prop = {"tprop": "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=40, singles=False)
        Eout, Eccout = tdccsdT.run()
        diff = abs(Eccref - Eccout)
        error = "Expected: {}  Actual: {}".format(Eccref, Eccout)
        self.assertTrue(diff < 1e-6, error)


if __name__ == '__main__':
    unittest.main()
