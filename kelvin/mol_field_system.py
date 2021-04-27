import numpy
from pyscf import gto, scf
from cqcpy import ft_utils
from cqcpy import utils
from . import integrals
from . import scf_utils
from . import zt_mp
from . import ft_mp
from .integrals import eri_blocks
from .system import *
from .neq_system import *

class mol_field_system(system):
    """Object representing a molecular system in a TD field (deprecated).

    Attributes:
        mf (pyscf.scf): SCF object (not necessarily converged).
        T (float): Temperature.
        mu (float): Chemical potential.
    """
    def __init__(self,mf,T,mu,omega,ti,Emax=1.0,O=None,ot=None):
        self.mf = mf
        self.T = T
        self.mu = mu
        self.omega = omega
        self.ti = ti
        self.nt = ti.shape[0]
        self.O = O
        self.ot = ot
        self.beta = 1/T if T > 0.0 else 1.0e20

    def reversible(self):
        if self.O is None:
            return True
        else:
            return False

    def verify(self,T,mu):
        if not (T == self.T and mu == self.mu):
            return False
        else:
            return True

    def const_energy(self):
        return self.mf.mol.energy_nuc()

    def get_mp1(self):
        # contribution from imaginary contour
        hcore = self.mf.get_hcore(self.mf.mol)
        h = utils.block_diag(hcore,hcore)
        en = self.g_energies_tot()
        fo = ft_utils.ff(self.beta, en, self.mu)
        p = scf_utils.get_ao_ft_den(self.mf, fo)
        f0 = scf_utils.get_ao_fock(self.mf)
        fao = scf_utils.get_ao_ft_fock(self.mf, fo)
        E1 = ft_mp.mp1(p,2*f0 - fao,h)

        # contribution from real-time contour
        if self.O is not None:
            E1 += numpy.einsum('ii,i->',utils.block_diag(self.O,self.O),fo)
        return E1

    def g_energies_tot(self):
        return scf_utils.get_orbital_energies_gen(self.mf)

    def g_fock_tot(self,direc='f'):
        en = self.g_energies_tot()
        fo = ft_utils.ff(self.beta, en, self.mu)
        F = scf_utils.get_mo_ft_fock(self.mf, fo)
        E = numpy.zeros((3))
        E[2] = 1.0
        field = numpy.einsum('x,xij->ij', E, self.mf.mol.intor('cint1e_r_sph', comp=3))
        field = scf_utils.mo_tran_1e(self.mf,field)
        I = numpy.ones(self.nt)
        Fock = (I[:,None,None]*F[None,:,:]).astype(complex)
        delta = self.ti[self.ot] - self.ti[self.ot - 1]
        ng = len(self.ti)
        for i in range(ng):
            temp = field*numpy.sin(self.omega*self.ti[i])
            Fock[i] += temp
        if direc == 'f':
            Fock[self.ot] += -3.j*self.beta*utils.block_diag(self.O,self.O)/delta
        elif direc == 'b':
            Fock[self.ot] -= 0.j*self.beta*utils.block_diag(self.O,self.O)/delta
        else:
            raise Exception("Unrecognized direction: " + str(direc))

        if direc == 'b':
            temp = Fock.copy()
            for i in range(ng):
                Fock[i] = temp[ng - i - 1]
            return Fock
        else:
            return Fock

    def r_hcore(self):
        hcore = self.mf.get_hcore()
        thcore = I[:,None,None,]*hcore[None,:,:]
        E = numpy.zeros((3))
        E[2] = 1.0
        F = numpy.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3))
        for i in range(self.nt):
            thcore[i] += numpy.sin(self.omega*ti[i])*F
        for i in range(self.nt):
            thcore[i] = scf_utils.mo_tran_1e(self.mf,thcore[i])
        if direc == 'f':
            thcore[ot] += O
        elif direc == 'b':
            thcore[ot] -= O
            thcore = numpy.flip(thcore,0)
        return thcore

    def g_aint_tot(self):
        return integrals.get_phys_antiu_all_gen(self.mf)
