import unittest
from kelvin.mp2 import mp2
from kelvin.fci import FCI
from kelvin.scf_system import scf_system


def compute_ft_mp2(m, T, mu):
    sys = scf_system(m, T, mu)
    mp2T = mp2(sys, T=T, mu=mu, iprint=0)
    E0T, E1T, E2T = mp2T.run()
    return (E0T, E1T, E2T)


def compute_zt_mp2(m):
    sys = scf_system(m, 0.0, 0.0)
    mp20 = mp2(sys, iprint=0)
    E00, E10, E20 = mp20.run()
    return (E00, E10, E20)


def compute_G012_fci(m, T, mu):
    sys = scf_system(m, T, mu)
    fciT = FCI(sys, T=T, mu=mu)
    delta = 1e-4
    fciT.lam = delta
    Ef = fciT.run()[0]
    fciT.lam = 0.0
    Ec = fciT.run()[0]
    fciT.lam = -delta
    Eb = fciT.run()[0]
    E1r = (Ef - Eb)/(2.0*delta)
    E2r = (Ef - 2.0*Ec + Eb) / (2.0*delta*delta)
    return (Ec, E1r, E2r)


class FTMP2Test(unittest.TestCase):
    def setUp(self):
        self.t0 = 1e-9
        self.t1 = 1e-6
        self.t2 = 1e-5

    def _test_Be_vs_fd(self, T, mu):
        from pyscf import gto, scf
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        out = compute_ft_mp2(m, T, mu)
        ref = compute_G012_fci(m, T, mu)
        e0str = "Expected: {}  Actual: {}".format(ref[0], out[0])
        e1str = "Expected: {}  Actual: {}".format(ref[1], out[1])
        e2str = "Expected: {}  Actual: {}".format(ref[2], out[2])
        d0 = abs(ref[0] - out[0])
        d1 = abs(ref[1] - out[1])
        d2 = abs(ref[2] - out[2])
        self.assertTrue(d0 < self.t0, e0str)
        self.assertTrue(d1 < self.t1, e1str)
        self.assertTrue(d2 < self.t2, e2str)

    def test_Be_vs_fd_all(self):
        self._test_Be_vs_fd(0.1, 0.0)
        self._test_Be_vs_fd(1.0, 0.0)
        self._test_Be_vs_fd(10.0, 0.0)

        self._test_Be_vs_fd(0.1, 0.3)
        self._test_Be_vs_fd(1.0, 0.2)
        self._test_Be_vs_fd(10.0, 0.1)

    def test_0T_Be_sto3g(self):
        from pyscf import gto, scf
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        out = compute_ft_mp2(m, 0.01, 0.0)
        ref = compute_zt_mp2(m)

        e0str = "Expected: {}  Actual: {}".format(ref[0], out[0])
        e1str = "Expected: {}  Actual: {}".format(ref[1], out[1])
        e2str = "Expected: {}  Actual: {}".format(ref[2], out[2])
        d0 = abs(ref[0] - out[0])
        d1 = abs(ref[1] - out[1])
        d2 = abs(ref[2] - out[2])
        self.assertTrue(d0 < 1e-9, e0str)
        self.assertTrue(d1 < 1e-9, e1str)
        self.assertTrue(d2 < 1e-9, e2str)

    def test_Hdiamond_vs_fd(self):
        import pyscf.pbc.gto as pbc_gto
        import pyscf.pbc.scf as pbc_scf
        cell = pbc_gto.Cell()
        cell.atom = '''
        He 0.000000000000   0.000000000000   0.000000000000
        He 1.685068664391   1.685068664391   1.685068664391
        '''
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.unit = 'B'
        cell.verbose = 0
        cell.build()
        mf = pbc_scf.RHF(cell, exxdiv=None)
        mf.kernel()
        T = 0.5
        mu = 0.0
        out = compute_ft_mp2(mf, T, mu)
        ref = compute_G012_fci(mf, T, mu)

        e0str = "Expected: {}  Actual: {}".format(ref[0], out[0])
        e1str = "Expected: {}  Actual: {}".format(ref[1], out[1])
        e2str = "Expected: {}  Actual: {}".format(ref[2], out[2])
        d0 = abs(ref[0] - out[0])
        d1 = abs(ref[1] - out[1])
        d2 = abs(ref[2] - out[2])
        self.assertTrue(d0 < 1e-9, e0str)
        self.assertTrue(d1 < 1e-9, e1str)
        self.assertTrue(d2 < 1e-7, e2str)


if __name__ == '__main__':
    unittest.main()
