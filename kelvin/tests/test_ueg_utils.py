import unittest
import numpy

from cqcpy import spin_utils

from kelvin.ueg_system import ueg_system
from kelvin import ueg_utils

class UEGUtilsTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_integral_sym(self):
        na = 7
        nb = 7
        N = na + nb
        L = 3.8855
        mu = 2.0
        ueg = ueg_system(0.0,L,7.4,mu=mu,norb=19,orbtype='g')
        I = ueg.g_int_tot()
        P12 = numpy.linalg.norm(I - I.transpose((1,0,3,2)))/numpy.sqrt(I.size)
        Preal = numpy.linalg.norm(I - I.transpose((2,3,0,1)))/numpy.sqrt(I.size)
        self.assertTrue(P12 < 1e-12)
        self.assertTrue(Preal < 1e-12)

    def test_ft_uint(self):
        na = 7
        nb = 7
        N = na + nb
        L = 3.8855
        mu = 2.0
        ueg = ueg_system(0.0,L,7.4,mu=mu,norb=19,orbtype='u')
        I = ueg.g_aint_tot()
        n = I.shape[0]
        na = n//2
        Irefaa = I[:na,:na,:na,:na]
        Irefbb = Irefaa
        Irefab = I[:na,na:,:na,na:]
        Ia,Ib,Iabab = ueg.u_aint_tot()

        da = numpy.linalg.norm(Irefaa - Ia)/numpy.sqrt(Ia.size)
        db = numpy.linalg.norm(Irefbb - Ib)/numpy.sqrt(Ib.size)
        dab = numpy.linalg.norm(Irefab - Iabab)/numpy.sqrt(Iabab.size)
        self.assertTrue(da < 1e-12)
        self.assertTrue(db < 1e-12)
        self.assertTrue(dab < 1e-12)

    def test_ufock(self):
        na = 1
        nb = 1
        N = na + nb
        L = 3.8855
        mu = 0.01
        ueg = ueg_system(0.0,L,7.4,mu=mu,norb=7,orbtype='u')
        Fa,Fb = ueg.u_fock()
        F = ueg.g_fock()
        diff = numpy.linalg.norm(Fa.vv - F.vv[:6,:6])
        self.assertTrue(diff < 1e-12)
        diff = numpy.linalg.norm(Fa.oo - F.oo[:1,:1])
        self.assertTrue(diff < 1e-12)

if __name__ == '__main__':
    unittest.main()
