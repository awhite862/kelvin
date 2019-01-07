import unittest
from pyscf import gto, scf, cc
from kelvin.ccsd import ccsd
from kelvin.scf_system import scf_system

def test_ccsd_gen(m):
    mycc = cc.CCSD(m)
    mycc.conv_tol = 1e-12
    mycc.run()
    sys = scf_system(m,0.0,0.0,orbtype='g')
    ccsd0 = ccsd(sys,iprint=0,max_iter=44,econv=1e-12)
    Etot,Ecc = ccsd0.run()

    return (mycc.e_corr, Ecc)

def test_ccsd(m):
    mycc = cc.CCSD(m)
    mycc.conv_tol = 1e-12
    mycc.run()
    sys = scf_system(m,0.0,0.0)
    ccsd0 = ccsd(sys,iprint=0,max_iter=44,econv=1e-12)
    Etot,Ecc = ccsd0.run()

    return (mycc.e_corr, Ecc)

class CCSDTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-10

    def test_Be_sto3g_gen(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-13
        Escf = m.scf()
        res = test_ccsd_gen(m)
        diff = abs(res[1] - res[0])
        error = "Expected: {}  Actual: {}".format(res[0],res[1])
        self.assertTrue(diff < self.thresh,error)
        m = scf.UHF(mol)

    def test_N2p_631G_gen(self):
        mol = gto.M(
            verbose = 0,
            atom = 'N 0 0 0; N 0 0 1.1',
            basis = '6-31G',
            charge = 1,
            spin = 1)
        m = scf.UHF(mol)
        m.conv_tol = 1e-13
        Escf = m.scf()
        res = test_ccsd_gen(m)
        diff = abs(res[1] - res[0])
        error = "Expected: {}  Actual: {}".format(res[0],res[1])
        self.assertTrue(diff < self.thresh,error)

    def test_Be_sto3g(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-13
        Escf = m.scf()
        res = test_ccsd(m)
        diff = abs(res[1] - res[0])
        error = "Expected: {}  Actual: {}".format(res[0],res[1])
        self.assertTrue(diff < self.thresh,error)
        m = scf.UHF(mol)

    def test_N2p_631G(self):
        mol = gto.M(
            verbose = 0,
            atom = 'N 0 0 0; N 0 0 1.1',
            basis = '6-31G',
            charge = 1,
            spin = 1)
        m = scf.UHF(mol)
        m.conv_tol = 1e-13
        Escf = m.scf()
        res = test_ccsd(m)
        diff = abs(res[1] - res[0])
        error = "Expected: {}  Actual: {}".format(res[0],res[1])
        self.assertTrue(diff < self.thresh,error)

if __name__ == '__main__':
    unittest.main()
