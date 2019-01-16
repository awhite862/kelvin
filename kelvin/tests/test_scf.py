import unittest
from kelvin import zt_mp
from kelvin.scf_system import scf_system

class SCFTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-10

    def test_Be_sto3g(self):
        from pyscf import gto, scf, cc
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-13
        Escf = m.scf()
        sys = scf_system(m,0.0,0.0)
        eo,ev = sys.g_energies()
        En = sys.const_energy()
        E0 = zt_mp.mp0(eo) + En
        E1 = sys.get_mp1()
        Ehf = E0 + E1
        diff = abs(Ehf - Escf)
        self.assertTrue(diff < self.thresh)

    def test_diamond(self):
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
        cell.verbose = 1
        cell.build()

        mf = scf.RHF(cell)
        mf.conv_tol_grad = 1e-8
        mf.conv_tol = 1e-12
        Escf = mf.kernel()
        #print("HF energy (per unit cell) = %.17g" % Escf)
        sys = scf_system(mf,0.0,0.0)
        eo,ev = sys.g_energies()
        En = sys.const_energy()
        E0 = zt_mp.mp0(eo) + En
        E1 = sys.get_mp1()
        Ehf = E0 + E1
        #print(Ehf)
        diff = abs(Ehf - Escf)
        self.assertTrue(diff < self.thresh)

if __name__ == '__main__':
    unittest.main()
