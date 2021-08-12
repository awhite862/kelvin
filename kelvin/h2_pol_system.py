import numpy
from pyscf import gto, scf
from cqcpy import ft_utils
from cqcpy import integrals
from .system import System


class h2_pol_system(System):
    def __init__(self, T, mu):
        mol = gto.M(
            verbose=0,
            atom='H 0 0 -0.6; H 0 0 0.0',
            basis='STO-3G',
            charge=1,
            spin=1)

        self.m = scf.UHF(mol)
        self.m.scf()
        self.T = T
        self.mu = mu
        mos = self.m.mo_coeff[0]
        self.eri = integrals.get_phys(mol, mos, mos, mos, mos)
        self.hcore = numpy.einsum('mp,mn,nq->pq', mos, self.m.get_hcore(mol), mos)
        self.beta = 1.0 / self.T if self.T > 0.0 else 1.0e20

    def verify(self, T, mu):
        if not (T == self.T and mu == self.mu):
            return False
        else:
            return True

    def has_g(self):
        return True

    def has_u(self):
        return False

    def const_energy(self):
        return self.m.mol.energy_nuc()

    def get_mp1(self):
        en = self.g_energies_tot()
        fo = ft_utils.ff(self.beta, en, self.mu)
        E1 = numpy.einsum('ii,i->', self.hcore, fo) - (self.g_energies_tot()*fo).sum()
        E1 += 0.5*numpy.einsum('ijij,i,j->', self.eri, fo, fo)
        E1 -= 0.5*numpy.einsum('ijji,i,j->', self.eri, fo, fo)
        return E1

    def g_energies_tot(self):
        F = self.hcore + self.eri[:, 0, :, 0] - self.eri[:, 0, 0, :]
        e, v = numpy.linalg.eigh(F)
        return e

    def g_fock_tot(self):
        en = self.g_energies_tot()
        fo = ft_utils.ff(self.beta, en, self.mu)
        F = self.hcore + \
            (self.eri[:, 0, :, 0] - self.eri[:, 0, 0, :])*fo[0] +\
            (self.eri[:, 1, :, 1] - self.eri[:, 1, 1, :])*fo[1]
        return F

    def g_aint_tot(self):
        return self.eri - self.eri.transpose((0, 1, 3, 2))
