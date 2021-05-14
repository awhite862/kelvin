import unittest
from kelvin.fci import FCI
from kelvin.ueg_system import UEGSystem
from kelvin.ueg_scf_system import ueg_scf_system
import numpy


def rs_to_L(rs, N):
    x = 4.0*numpy.pi*N/(3.0)
    return rs*pow(x, 1.0/3.0)


def ueg_fci(L, Emax, norb, na, nb):
    ueg = UEGSystem(0.0, L, Emax, na=na, nb=nb, norb=7)
    assert(ueg.basis.get_nbsf() == norb)
    fci0 = FCI(ueg, T=0.0, nalpha=na, nbeta=nb, iprint=0)
    return fci0.run()[1]


def scf_ueg_fci(L, Emax, norb, na, nb):
    ueg = ueg_scf_system(0.0, L, Emax, na=na, nb=nb, norb=7)
    assert(ueg.basis.get_nbsf() == norb)
    fci0 = FCI(ueg, T=0.0, nalpha=na, nbeta=nb, iprint=0)
    return fci0.run()[1]


class UEGTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-10
        # in Hartrees
        self.ref_2_19_4 = -0.00733580497*2
        self.ref_2_19_8 = -0.00589674127*2
        self.ref_2_07_4 = -0.00658416580*2
        self.ref_2_07_8 = -0.00562036146*2

    def test_2_07_4(self):
        L = rs_to_L(4.0, 2.0)
        E = ueg_fci(L, 1.0, 7, 1, 1)
        ref = self.ref_2_07_4
        diff = abs(E - ref)
        error = "Expected: {}  Actual: {}".format(ref, E)
        self.assertTrue(diff < self.thresh, error)

        E = scf_ueg_fci(L, 1.0, 7, 1, 1)
        diff = abs(E - ref)
        error = "Expected: {}  Actual: {}".format(ref, E)
        self.assertTrue(diff < self.thresh, error)

    def test_2_07_8(self):
        L = rs_to_L(8.0, 2.0)
        E = ueg_fci(L, 1.0, 7, 1, 1)
        ref = self.ref_2_07_8
        diff = abs(E - ref)
        error = "Expected: {}  Actual: {}".format(ref, E)
        self.assertTrue(diff < self.thresh, error)

        E = scf_ueg_fci(L, 1.0, 7, 1, 1)
        diff = abs(E - ref)
        error = "Expected: {}  Actual: {}".format(ref, E)
        self.assertTrue(diff < self.thresh, error)


if __name__ == '__main__':
    unittest.main()
