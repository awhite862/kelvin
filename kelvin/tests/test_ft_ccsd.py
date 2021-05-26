import unittest
import numpy
from pyscf import gto, scf
from kelvin.ccsd import ccsd
from kelvin.fci import FCI
from kelvin.scf_system import SCFSystem
from kelvin.ueg_system import UEGSystem
from kelvin.pueg_system import PUEGSystem


def compute_ft_ccsd(m, T, mu):
    sys = SCFSystem(m, T, mu)
    ccsdT = ccsd(sys, T=T, mu=mu, iprint=0)
    return ccsdT.run()


class FTCCSDTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-9
        self.Be_ref = -0.101954860211
        self.Be_ref_rt = -0.050492135921
        self.Be_ref_ac = -0.061275804345406214
        self.ueg_ref = -0.010750238811
        self.pueg_ref = -0.001403909274

    def test_2orb(self):
        mol = gto.M(
            verbose=0,
            atom='He 0 0 0',
            basis='sto-3g')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()

        T = 1.0
        mu = 0.0
        sys = SCFSystem(m, T, mu)
        ccsdT = ccsd(sys, iprint=0, T=T, mu=mu, max_iter=24, econv=1e-11, ngrid=220)
        Ecc = ccsdT.run()

        fciT = FCI(sys, T=T, mu=mu)
        Efci = fciT.run()
        error = "Expected: {}  Actual: {}".format(Efci[1], Ecc[1])
        diff = abs(Efci[1] - Ecc[1])
        self.assertTrue(diff < self.thresh, error)

    def test_Be(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        cc = compute_ft_ccsd(m, 5.0, 0.0)
        diff = abs(self.Be_ref - cc[1])
        error = "Expected: {}  Actual: {}".format(self.Be_ref, cc[1])
        self.assertTrue(diff < self.thresh, error)

    def test_Be_rt(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        sys = SCFSystem(m, 0.0, 0.0)
        ccsdT = ccsd(sys, realtime=True, ngrid=160, iprint=0)
        cc = ccsdT.run()
        diff = abs(self.Be_ref_rt - cc[1])
        error = "Expected: {}  Actual: {}".format(self.Be_ref_rt, cc[1])
        self.assertTrue(diff < self.thresh, error)

    def test_Be_active(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        T = 0.02
        mu = 0.0
        sys = SCFSystem(m, T, mu, orbtype='g')
        ccsdT = ccsd(
                sys, iprint=0, singles=True, T=T, mu=mu, damp=0.1,
                max_iter=80, ngrid=160, athresh=1e-30)
        cc = ccsdT.run()
        diff = abs(self.Be_ref_ac - cc[1])
        error = "Expected: {}  Actual: {}".format(self.Be_ref_ac, cc[1])
        self.assertTrue(diff < self.thresh, error)

    def test_Be_uactive(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')

        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        T = 0.02
        mu = 0.0
        sys = SCFSystem(m, T, mu, orbtype='u')
        ccsdT = ccsd(
                sys, iprint=0, singles=True, T=T, mu=mu, damp=0.1,
                max_iter=80, ngrid=160, athresh=1e-30)
        cc = ccsdT.run()
        diff = abs(self.Be_ref_ac - cc[1])
        error = "Expected: {}  Actual: {}".format(self.Be_ref_ac, cc[1])
        self.assertTrue(diff < self.thresh, error)

    def test_ueg_gen(self):
        T = 0.1
        mu = 0.1
        L = 2*numpy.pi/numpy.sqrt(1.0)
        norb = 7
        cut = 1.2
        damp = 0.2
        mi = 50
        ueg = UEGSystem(T, L, cut, mu=mu, norb=norb, orbtype='g')
        ccsdT = ccsd(ueg, T=T, mu=mu, iprint=0, max_iter=mi, damp=damp, ngrid=10)
        Ecctot, Ecc = ccsdT.run()
        diff = abs(self.ueg_ref - Ecc)
        error = "Expected: {}  Actual: {}".format(self.ueg_ref, Ecc)
        self.assertTrue(diff < self.thresh, error)

    def test_ueg(self):
        T = 0.1
        mu = 0.1
        L = 2*numpy.pi/numpy.sqrt(1.0)
        norb = 7
        cut = 1.2
        damp = 0.2
        mi = 50
        ueg = UEGSystem(T, L, cut, mu=mu, norb=norb)
        ccsdT = ccsd(ueg, T=T, mu=mu, iprint=0, max_iter=mi, damp=damp, ngrid=10)
        Ecctot, Ecc = ccsdT.run()
        diff = abs(self.ueg_ref - Ecc)
        error = "Expected: {}  Actual: {}".format(self.ueg_ref, Ecc)
        self.assertTrue(diff < self.thresh, error)

    def test_pueg(self):
        T = 0.1
        mu = 0.1
        L = 2*numpy.pi/numpy.sqrt(1.0)
        norb = 7
        cut = 1.2
        damp = 0.2
        mi = 50
        ueg = PUEGSystem(T, L, cut, mu=mu, norb=norb)
        ccsdT = ccsd(ueg, T=T, mu=mu, iprint=0, max_iter=mi, damp=damp, ngrid=10)
        Ecctot, Ecc = ccsdT.run()
        diff = abs(self.pueg_ref - Ecc)
        error = "Expected: {}  Actual: {}".format(self.pueg_ref, Ecc)
        self.assertTrue(diff < 1e-8, error)

    def test_ueg_gen_conv(self):
        T = 0.1
        mu = 0.1
        L = 2*numpy.pi/numpy.sqrt(1.0)
        norb = 7
        cut = 1.2
        damp = 0.2
        mi = 50
        ueg = UEGSystem(T, L, cut, mu=mu, norb=norb, orbtype='g')
        ccsdT = ccsd(ueg, T=T, mu=mu, iprint=0, max_iter=mi, damp=damp, tconv=1e-8, ngrid=10)
        Ecctot1, Ecc1 = ccsdT.run()
        ccsdT = ccsd(ueg, T=T, mu=mu, iprint=0, max_iter=mi, damp=damp, tconv=1e-8, rt_iter="point")
        Ecctot2, Ecc2 = ccsdT.run()
        diff = abs(Ecc1 - Ecc2)
        error = "Expected: {}  Actual: {}".format(Ecc1, Ecc2)
        self.assertTrue(diff < 1e-8, error)

    def test_ueg_conv(self):
        T = 0.1
        mu = 0.1
        L = 2*numpy.pi/numpy.sqrt(1.0)
        norb = 7
        cut = 1.2
        damp = 0.2
        mi = 50
        ueg = UEGSystem(T, L, cut, mu=mu, norb=norb, orbtype='u')
        ccsdT = ccsd(ueg, T=T, mu=mu, iprint=0, max_iter=mi, damp=damp, tconv=1e-8, ngrid=10)
        Ecctot1,Ecc1 = ccsdT.run()
        ccsdT = ccsd(ueg, T=T, mu=mu, iprint=0, max_iter=mi, damp=damp, tconv=1e-8, rt_iter="point")
        Ecctot2,Ecc2 = ccsdT.run()
        diff = abs(Ecc1 - Ecc2)
        error = "Expected: {}  Actual: {}".format(Ecc1, Ecc2)
        self.assertTrue(diff < 1e-8, error)


if __name__ == '__main__':
    unittest.main()
