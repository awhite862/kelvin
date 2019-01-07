import unittest
from pyscf import gto, scf, mp
from kelvin.mp2 import mp2
from kelvin.scf_system import scf_system

def test_mp2(m):
    sys = scf_system(m,0.0,0.0)
    mp20 = mp2(sys,iprint=0)
    E00,E10,E20 = mp20.run()

    pt = mp.MP2(m)
    Emp, temp = pt.kernel()

    return (Emp,E20)

class MP2Test(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-12

    def test_Be_sto3g(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        res = test_mp2(m)
        diff = abs(res[1] - res[0])
        self.assertTrue(diff < self.thresh)

    def test_N2p_631G(self):
        mol = gto.M(
            verbose = 0,
            atom = 'N 0 0 0; N 0 0 1.1',
            basis = '6-31G',
            charge = 1,
            spin = 1)
        m = scf.UHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        res = test_mp2(m)
        diff = abs(res[1] - res[0])
        self.assertTrue(diff < self.thresh)

if __name__ == '__main__':
    unittest.main()
