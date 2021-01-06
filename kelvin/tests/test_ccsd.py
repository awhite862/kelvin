import unittest
import numpy
from pyscf import gto, scf, cc
from pyscf.pbc import cc as pbc_cc
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

    def test_diamond_g(self):
        from pyscf.pbc import gto, scf, dft
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
        cell.verbose = 0
        cell.build()

        mf = scf.RHF(cell,exxdiv=None)
        mf.conv_tol_grad = 1e-8
        mf.conv_tol = 1e-12
        Escf = mf.kernel()
        mycc = cc.CCSD(mf)
        mycc.conv_tol = 1e-11
        mycc.conv_tol_normt = 1e-9
        Ecc = mycc.kernel()
        sys = scf_system(mf,0.0,0.0,orbtype='g')
        ccsd0 = ccsd(sys,iprint=0,max_iter=100,econv=1e-11,damp=0.0)
        Etot,Ecc2 = ccsd0.run()
        diff = abs(Ecc[0] - Ecc2)
        #print(diff)
        self.assertTrue(diff < self.thresh)
        #print(Ecc[0],Ecc2)

    def test_diamond_u(self):
        from pyscf.pbc import gto, scf, dft
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
        cell.verbose = 0
        cell.build()

        mf = scf.RHF(cell,exxdiv=None)
        mf.conv_tol_grad = 1e-8
        mf.conv_tol = 1e-12
        Escf = mf.kernel()
        mycc = cc.CCSD(mf)
        mycc.conv_tol = 1e-11
        mycc.conv_tol_normt = 1e-9
        Ecc = mycc.kernel()
        sys = scf_system(mf,0.0,0.0)
        ccsd0 = ccsd(sys,iprint=0,max_iter=100,econv=1e-11,damp=0.0)
        Etot,Ecc2 = ccsd0.run()
        diff = abs(Ecc[0] - Ecc2)
        self.assertTrue(diff < self.thresh)

    def test_diamond_uk(self):
        from pyscf.pbc import gto, scf, dft
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
        cell.verbose = 4
        cell.build()
        kpt = cell.make_kpts((1,1,1), scaled_center=(0,0,1./3.))

        mf = scf.RHF(cell,kpt=kpt,exxdiv=None)
        mf.conv_tol_grad = 1e-8
        mf.conv_tol = 1e-12
        Escf = mf.kernel()
        mycc = pbc_cc.CCSD(mf)
        mycc.conv_tol = 1e-11
        mycc.conv_tol_normt = 1e-9
        Ecc = mycc.kernel()
        sys = scf_system(mf,0.0,0.0)
        ccsd0 = ccsd(sys,iprint=1,max_iter=100,econv=1e-11,damp=0.0)
        Etot,Ecc2 = ccsd0.run()
        diff = abs(Ecc[0] - Ecc2)
        self.assertTrue(diff < self.thresh)

if __name__ == '__main__':
    unittest.main()
