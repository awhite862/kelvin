import unittest
import numpy
from kelvin import zt_mp
from kelvin.scf_system import scf_system


def get_Be_sto3g():
    from pyscf import gto, scf
    mol = gto.M(
        verbose=0,
        atom='Be 0 0 0',
        basis='sto-3G')
    m = scf.RHF(mol)
    m.conv_tol = 1e-13
    return m


def get_diamond():
    import pyscf.pbc.gto as pbc_gto
    import pyscf.pbc.scf as pbc_scf
    cell = pbc_gto.Cell()
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
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
    mf.conv_tol_grad = 1e-8
    mf.conv_tol = 1e-12
    return mf


def get_diamond_k():
    import pyscf.pbc.gto as pbc_gto
    import pyscf.pbc.scf as pbc_scf
    cell = pbc_gto.Cell()
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
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
    mf = pbc_scf.RHF(cell, kpt=(0.1,0.1,0.1), exxdiv=None)
    mf.conv_tol_grad = 1e-8
    mf.conv_tol = 1e-12
    return mf


class SCFTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-10

    def test_Be_sto3g_energy(self):
        m = get_Be_sto3g()
        Escf = m.scf()
        sys = scf_system(m, 0.0, 0.0)
        eo,ev = sys.g_energies()
        En = sys.const_energy()
        E0 = zt_mp.mp0(eo) + En
        E1 = sys.get_mp1()
        Ehf = E0 + E1
        diff = abs(Ehf - Escf)
        self.assertTrue(diff < self.thresh)

    def test_diamond_energy(self):
        mf = get_diamond()
        Escf = mf.kernel()
        sys = scf_system(mf, 0.0, 0.0)
        eo,ev = sys.g_energies()
        En = sys.const_energy()
        E0 = zt_mp.mp0(eo) + En
        E1 = sys.get_mp1()
        Ehf = E0 + E1
        diff = abs(Ehf - Escf)
        self.assertTrue(diff < self.thresh)

    def test_Be_sto3g_ft_deriv(self):
        T = 0.5
        mu = 0.0
        beta = 1.0/T
        m = get_Be_sto3g()
        m.scf()
        sys = scf_system(m,T,mu)

        # derivative with respect to mu
        en = sys.g_energies_tot()
        dvec = -beta*numpy.ones(en.shape)
        Fda = sys.g_fock_d_tot(dvec)

        delta = 5e-4
        sysf = scf_system(m, T, mu + delta)
        Ff = sysf.g_fock_tot()
        sysb = scf_system(m, T, mu - delta)
        Fb = sysb.g_fock_tot()
        Fdd = 0.5*(Ff - Fb)/delta
        diff = numpy.linalg.norm(Fdd - Fda)
        self.assertTrue(diff < 1e-5)

    def test_diamond_ft_deriv(self):
        T = 0.5
        mu = 0.0
        beta = 1.0/T
        m = get_diamond()
        m.scf()
        sys = scf_system(m, T, mu)

        # derivative with respect to mu
        en = sys.g_energies_tot()
        dvec = -beta*numpy.ones(en.shape)
        Fda = sys.g_fock_d_tot(dvec)

        delta = 5e-4
        sysf = scf_system(m, T, mu + delta)
        Ff = sysf.g_fock_tot()
        sysb = scf_system(m, T, mu - delta)
        Fb = sysb.g_fock_tot()
        Fdd = 0.5*(Ff - Fb)/delta
        diff = numpy.linalg.norm(Fdd - Fda)
        self.assertTrue(diff < 1e-5)

    def test_Be_sto3g_ft_mp_deriv(self):
        T = 0.5
        mu = 0.0
        beta = 1.0/T
        m = get_Be_sto3g()
        m.scf()
        sys = scf_system(m, T, mu)

        # derivative with respect to mu
        en = sys.g_energies_tot()
        dvec = -beta*numpy.ones(en.shape)
        dMP1a = sys.g_d_mp1(dvec)

        delta = 5e-4
        sysf = scf_system(m, T, mu + delta)
        MP1f = sysf.get_mp1()
        sysb = scf_system(m, T, mu - delta)
        MP1b = sysb.get_mp1()
        dMP1d = 0.5*(MP1f - MP1b)/delta
        diff = abs(dMP1a - dMP1d)
        self.assertTrue(diff < 1e-6)

    def test_diamond_ft_mp_deriv(self):
        T = 0.5
        mu = 0.0
        beta = 1.0/T
        m = get_diamond()
        m.scf()
        sys = scf_system(m, T, mu)

        # derivative with respect to mu
        en = sys.g_energies_tot()
        dvec = -beta*numpy.ones(en.shape)
        ea,eb = sys.u_energies_tot()
        dveca = -beta*numpy.ones(ea.shape)
        dvecb = -beta*numpy.ones(eb.shape)
        dMP1au = sys.u_d_mp1(dveca,dvecb)
        dMP1a = sys.g_d_mp1(dvec)

        delta = 5e-4
        sysf = scf_system(m, T, mu + delta)
        MP1f = sysf.get_mp1()
        sysb = scf_system(m, T, mu - delta)
        MP1b = sysb.get_mp1()
        dMP1d = 0.5*(MP1f - MP1b)/delta
        diff = abs(dMP1a - dMP1d)
        self.assertTrue(diff < 1e-6)
        diff2 = abs(dMP1a - dMP1au)
        self.assertTrue(diff2 < 1e-6)

    def test_diamond_ft_mp_deriv_k(self):
        T = 0.5
        mu = 0.0
        beta = 1.0/T
        m = get_diamond_k()
        m.scf()
        sys = scf_system(m,T,mu)

        # derivative with respect to mu
        en = sys.g_energies_tot()
        dvec = -beta*numpy.ones(en.shape)
        ea,eb = sys.u_energies_tot()
        dveca = -beta*numpy.ones(ea.shape)
        dvecb = -beta*numpy.ones(eb.shape)
        dMP1au = sys.u_d_mp1(dveca,dvecb)
        dMP1a = sys.g_d_mp1(dvec)

        delta = 5e-4
        sysf = scf_system(m, T, mu + delta)
        MP1f = sysf.get_mp1()
        sysb = scf_system(m, T, mu - delta)
        MP1b = sysb.get_mp1()
        dMP1d = 0.5*(MP1f - MP1b)/delta
        diff = abs(dMP1a - dMP1d)
        self.assertTrue(diff < 1e-6)
        diff2 = abs(dMP1a - dMP1au)
        self.assertTrue(diff2 < 1e-6)


if __name__ == '__main__':
    unittest.main()
