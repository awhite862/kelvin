import unittest
from pyscf import gto, scf, mp
from kelvin.mp2 import MP2
from kelvin.scf_system import SCFSystem


def get_mp2(m):
    sys = SCFSystem(m, 0.0, 0.0)
    mp20 = MP2(sys, iprint=0)
    E00, E10, E20 = mp20.run()

    pt = mp.MP2(m)
    Emp, temp = pt.kernel()

    return (Emp,E20)


class MP2Test(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-12

    def test_Be_sto3g(self):
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        res = get_mp2(m)
        diff = abs(res[1] - res[0])
        self.assertTrue(diff < self.thresh)

    def test_N2p_631G(self):
        mol = gto.M(
            verbose=0,
            atom='N 0 0 0; N 0 0 1.1',
            basis='6-31G',
            charge=1,
            spin=1)
        m = scf.UHF(mol)
        m.conv_tol = 1e-12
        m.scf()
        res = get_mp2(m)
        diff = abs(res[1] - res[0])
        self.assertTrue(diff < self.thresh)

    @unittest.skip("Skipped for time")
    def test_diamond(self):
        from pyscf.pbc import gto, scf
        cell = gto.Cell()
        cell.a = '''
        3.5668  0       0
        0       3.5668  0
        0       0       3.5668'''
        cell.atom = '''C     0.      0.      0.
                      C     0.8917  0.8917  0.8917
                      C     1.7834  1.7834  0.
                      C     2.6751  2.6751  0.8917
                      C     1.7834  0.      1.7834
                      C     2.6751  0.8917  2.6751
                      C     0.      1.7834  1.7834
                      C     0.8917  2.6751  2.6751'''
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.verbose = 1
        cell.build()

        mf = scf.RHF(cell)
        mf.conv_tol_grad = 1e-8
        mf.conv_tol = 1e-12
        mf.kernel()
        pt = mp.MP2(mf)
        Emp, temp = pt.kernel()
        sys = SCFSystem(mf, 0.0, 0.0)
        mp20 = MP2(sys, iprint=0)
        E00, E10, E20 = mp20.run()
        diff = abs(E20 - Emp)
        self.assertTrue(diff < self.thresh)


if __name__ == '__main__':
    unittest.main()
