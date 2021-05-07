import numpy
from pyscf import gto, scf
from cqcpy import ft_utils
from cqcpy import integrals
from .system import system
from .neq_system import NeqSystem


class h2_field_system(system):
    """H2 molecule in a TD field (deprecated)"""
    def __init__(self, T, mu, omega, ti, O=None, ot=None):
        mol = gto.M(
            verbose=0,
            atom='H 0 0 -0.6; H 0 0 0.0',
            basis='STO-3G',
            charge=1,
            spin=1)

        self.m = scf.UHF(mol)
        self.m.scf()
        self.T = T
        self.beta = 1/T if T > 0.0 else 1.0e20
        self.mu = mu
        self.omega = omega
        self.ti = ti
        self.O = O
        self.ot = ot
        self.ot = ot
        mos = self.m.mo_coeff[0]
        self.eri = integrals.get_phys(mol, mos, mos, mos, mos)
        self.hcore = numpy.einsum('mp,mn,nq->pq', mos, self.m.get_hcore(mol), mos)

    def reversible(self):
        if self.O is None:
            return True
        else:
            return False

    def verify(self, T, mu):
        if not (T == self.T and mu == self.mu):
            return False
        else:
            return True

    def const_energy(self):
        return self.m.mol.energy_nuc()

    def get_mp1(self):
        en = self.g_energies_tot()
        fo = ft_utils.ff(self.beta, en, self.mu)
        E1 = numpy.einsum('ii,i->', self.hcore, fo) - (self.g_energies_tot()*fo).sum()
        E1 += 0.5*numpy.einsum('ijij,i,j->', self.eri, fo, fo)
        E1 -= 0.5*numpy.einsum('ijji,i,j->', self.eri, fo, fo)
        if self.O is not None:
            Eo = numpy.einsum('ii,i->', self.O, fo)
            E1 += Eo
        return E1

    def g_energies_tot(self):
        F = self.hcore + self.eri[:,0,:,0] - self.eri[:,0,0,:]
        e,v = numpy.linalg.eigh(F)
        return e

    def g_fock_tot(self, direc='f'):
        en = self.g_energies_tot()
        fo = ft_utils.ff(self.beta, en, self.mu)
        E = numpy.zeros(3)
        mol = self.m.mol
        mos = self.m.mo_coeff[0]
        E[2] = 1.0
        field = numpy.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3))
        field = numpy.einsum('mp,mn,nq->pq', mos, field, mos)
        F = self.hcore + \
            (self.eri[:,0,:,0] - self.eri[:,0,0,:])*fo[0] +\
            (self.eri[:,1,:,1] - self.eri[:,1,1,:])*fo[1]
        ng = len(self.ti)
        I = numpy.ones(ng)
        Fock = (I[:,None,None]*F[None,:,:]).astype(complex)
        ot = self.ot
        ti = self.ti
        for i in range(ng):
            temp = field*numpy.sin(self.omega*self.ti[i])
            Fock[i] += temp
        if self.O is not None:
            delta = ti[ot] - ti[ot - 1] #if ot > 0 else ti[ot + 1] - ti[ot]
        if direc == 'f' and self.O is not None:
            Fock[self.ot] += -1.j*self.beta*self.O/delta
        elif direc == 'b' and self.O is not None:
            Fock[self.ot] -= -0.j*self.beta*self.O/delta

        if direc == 'b':
            temp = Fock.copy()
            for i in range(ng):
                Fock[i] = temp[ng - i - 1]
            return Fock
        else:
            return Fock

    def g_aint_tot(self):
        return (self.eri - self.eri.transpose((0,1,3,2)))


class H2FieldSystem(NeqSystem):
    """H2 molecule in a TD field (deprecated)"""
    def __init__(self, T, mu, omega):
        mol = gto.M(
            verbose=0,
            atom='H 0 0 -0.6; H 0 0 0.0',
            basis='STO-3G',
            charge=1,
            spin=1)

        self.m = scf.UHF(mol)
        self.m.scf()
        self.T = T
        self.beta = 1/T if T > 0.0 else 1.0e20
        self.mu = mu
        self.omega = omega
        mos = self.m.mo_coeff[0]
        self.eri = integrals.get_phys(mol, mos, mos, mos, mos)
        self.hcore = numpy.einsum('mp,mn,nq->pq', mos, self.m.get_hcore(mol), mos)

    def verify(self, T, mu):
        if not (T == self.T and mu == self.mu):
            return False
        else:
            return True

    def has_g(self): return True
    def has_u(self): return False
    def has_r(self): return False

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
        F = self.hcore + self.eri[:,0,:,0] - self.eri[:,0,0,:]
        e,v = numpy.linalg.eigh(F)
        return e

    def g_fock_tot(self, t=0):
        en = self.g_energies_tot()
        fo = ft_utils.ff(self.beta, en, self.mu)
        E = numpy.zeros(3)
        mol = self.m.mol
        mos = self.m.mo_coeff[0]
        E[2] = 1.0
        field = numpy.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3))
        field = numpy.einsum('mp,mn,nq->pq', mos, field, mos)
        F = self.hcore + \
            (self.eri[:,0,:,0] - self.eri[:,0,0,:])*fo[0] +\
            (self.eri[:,1,:,1] - self.eri[:,1,1,:])*fo[1]
        Fock = F.astype(complex)
        temp = field*numpy.sin(self.omega*t)
        Fock += temp
        return Fock

    def g_aint_tot(self):
        return (self.eri - self.eri.transpose((0,1,3,2)))
