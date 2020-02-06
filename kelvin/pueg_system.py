import numpy
from cqcpy import ft_utils
from cqcpy.ov_blocks import one_e_blocks
from cqcpy.ov_blocks import two_e_blocks
from . import zt_mp
from . import ft_mp
from .ueg_utils import ueg_basis
from .system import system

class pueg_system(system):
    """The polarized uniform electron gas in a plane-wave basis set.
    
    Attributes: 
        T (float): Temperature.
        L (float): Box-length.
        basis: UEG plane-wave basis set.
        mu (float): Chemical potential.
        N (float): Number of electrons.
        den (float): Number density.
        rs (float): Wigner-Seitz radius.
        Ef (float): Fermi-energy (of non-interacting system).
        Tf (float): Redued temperature.
    """
    def __init__(self,T,L,Emax,mu=None,n=None,norb=None):
        self.T = T
        self.L = L
        self.basis = ueg_basis(L,Emax,norb=norb)
        if n is None:
            assert(mu is not None)
            self.mu = mu
            beta = 1.0 / self.T if self.T > 0.0 else 1.0e20
            en = self.g_energies_tot()
            fo = ft_utils.ff(beta, en, self.mu)
            N = fo.sum()
        else:
            self.N = n
            mu = self.basis.Es[self.N - 1] + 0.00001
            self.mu = mu
            assert(self.T == 0.0)

        self.N = N
        self.den = self.N/(L*L*L)
        self.rs = (3/(4.0*numpy.pi*self.den))**(1.0/3.0)
        pi2 = numpy.pi*numpy.pi
        self.Ef = 0.5*(3.0*pi2*self.den)**(2.0/3.0)
        self.Tf = self.T / self.Ef
        self.orbtype = 'g'

    def has_g(self):
        return True

    def has_u(self):
        return False

    def has_r(self):
        return False

    def verify(self,T,mu):
        if T > 0.0:
            s = T == self.T and mu == self.mu
        else:
            s = T == self.T
        if not s:
            return False
        else:
            return True

    def const_energy(self):
        return 0.0

    def get_mp1(self):
        if self.T > 0:
            V = self.g_aint_tot()
            beta = 1.0 / self.T
            en = self.g_energies_tot()
            fo = ft_utils.ff(beta, en, self.mu)
            return 0.5*numpy.einsum('ijij,i,j->',
                V,fo,fo)
        else:
            V = self.g_aint()
            return 0.5*numpy.einsum('ijij->',V.oooo)

    def g_d_mp1(self,dvec):
        if self.T > 0:
            V = self.g_aint_tot()
            beta = 1.0 / self.T
            en = self.g_energies_tot()
            fo = ft_utils.ff(beta, en, self.mu)
            fv = ft_utils.ffv(beta, en, self.mu)
            vec = dvec*fo*fv
            return -numpy.einsum('ijij,i,j->',V,vec,fo)
        else:
            print("WARNING: Derivative of MP1 energy is zero at OK")
            return 0.0

    def g_mp1_den(self):
        assert(self.T > 0.0)
        V = self.g_aint_tot()
        beta = 1.0 / self.T
        en = self.g_energies_tot()
        fo = ft_utils.ff(beta, en, self.mu)
        fv = ft_utils.ffv(beta, en, self.mu)
        vec = fo*fv
        return -beta*numpy.einsum('ijij,i,j->i',V,vec,fo)

    def g_energies(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        d = self.g_energies_tot()
        nbsf = self.basis.get_nbsf()
        n = int(self.N)
        eo = d[n:]
        ev = d[:n]
        return (eo,ev)

    def g_energies_tot(self):
        return self.basis.r_build_diag()

    def g_fock(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        mu = self.mu
        d = self.g_energies_tot()
        F = self.g_hcore()
        n = d.shape[0]
        occ = []
        vir = []
        for p in range(n):
            if d[p] < self.mu:
                occ.append(p)
            if d[p] > self.mu:
                vir.append(p)
        oidx = numpy.r_[occ]
        vidx = numpy.r_[vir]
        V = self.g_aint_tot()
        V = V[numpy.ix_(numpy.arange(n),oidx,numpy.arange(n),oidx)]
        F = F + numpy.einsum('piri->pr',V)
        Foo = F[numpy.ix_(oidx,oidx)]
        Fvv = F[numpy.ix_(vidx,vidx)]
        Fov = F[numpy.ix_(oidx,vidx)]
        Fvo = F[numpy.ix_(vidx,oidx)]
        return one_e_blocks(Foo,Fov,Fvo,Fvv)

    def g_fock_tot(self):
        T = self.basis.build_r_ke_matrix()
        d = self.g_energies_tot()
        n = d.shape[0]
        if self.T > 0.0:
            beta = 1.0 / self.T
            fo = ft_utils.ff(beta, d, self.mu)
            I = numpy.identity(n)
            den = numpy.einsum('pi,i,qi->pq',I,fo,I)
        else:
            to = numpy.zeros((n,self.N))
            i = 0
            for p in range(n):
                if d[p] < self.mu:
                    to[p,i] = 1.0
                    i = i+1
            den = numpy.einsum('pi,qi->pq',to,to)
        V = self.g_aint_tot()
        JK = numpy.einsum('prqs,rs->pq',V,den)
        return T + JK

    def g_fock_d_tot(self,dvec):
        d = self.g_energies_tot()
        n = d.shape[0]
        if self.T == 0.0:
            print("WARNING: Occupation derivatives are zero at 0K")
            return numpy.zeros((n,n))
        beta = 1.0 / self.T
        fo = ft_utils.ff(beta, d, self.mu)
        fv = ft_utils.ffv(beta, d, self.mu)
        vec = dvec*fo*fv
        I = numpy.identity(n)
        den = numpy.einsum('pi,i,qi->pq',I,vec,I)
        V = self.g_aint_tot()
        JK = -numpy.einsum('prqs,rs->pq',V,den)
        return JK

    def g_fock_d_den(self):
        d = self.g_energies_tot()
        n = d.shape[0]
        if self.T == 0.0:
            print("WARNING: Occupation derivatives are zero at 0K")
            return numpy.zeros((n,n))
        beta = 1.0 / self.T
        fo = ft_utils.ff(beta, d, self.mu)
        fv = ft_utils.ffv(beta, d, self.mu)
        vec = fo*fv
        V = self.g_aint_tot()
        #I = numpy.identity(n)
        #den = numpy.einsum('pi,i,qi->pq',I,vec,I)
        JK = numpy.einsum('piqi,i->pqi',V,vec)
        return JK

    def g_hcore(self):
        return self.basis.build_rke_matrix()

    def g_aint_tot(self):
        V = self.basis.build_r2e_matrix()
        V = V - V.transpose((0,1,3,2))
        return V

    def g_aint(self,code=0):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        d = self.g_energies_tot()
        n = d.shape[0]
        occ = []
        vir = []
        for p in range(n):
            if d[p] < self.mu:
                occ.append(p)
            if d[p] > self.mu:
                vir.append(p)
        V = self.g_aint_tot()
        Vvvvv = None
        Vvvvo = None
        Vvovv = None
        Vvvoo = None
        Vvovo = None
        Voovv = None
        Vvooo = None
        Vooov = None
        Voooo = None
        oidx = numpy.r_[occ]
        vidx = numpy.r_[vir]
        if code == 0 or code == 1:
            Vvvvv = V[numpy.ix_(vidx,vidx,vidx,vidx)]
        if code == 0 or code == 2:
            Vvvvo = V[numpy.ix_(vidx,vidx,vidx,oidx)] 
        if code == 0 or code == 3:
            Vvovv = V[numpy.ix_(vidx,oidx,vidx,vidx)]
        if code == 0 or code == 4:
            Vvvoo = V[numpy.ix_(vidx,vidx,oidx,oidx)]
        if code == 0 or code == 5:
            Vvovo = V[numpy.ix_(vidx,oidx,vidx,oidx)]
        if code == 0 or code == 6:
            Voovv = V[numpy.ix_(oidx,oidx,vidx,vidx)]
        if code == 0 or code == 7:
            Vvooo = V[numpy.ix_(vidx,oidx,oidx,oidx)]
        if code == 0 or code == 8:
            Vooov = V[numpy.ix_(oidx,oidx,oidx,vidx)]
        if code == 0 or code == 9:
            Voooo = V[numpy.ix_(oidx,oidx,oidx,oidx)]
        return two_e_blocks(
            vvvv=Vvvvv,vvvo=Vvvvo,
            vovv=Vvovv,vvoo=Vvvoo,
            vovo=Vvovo,oovv=Voovv,
            vooo=Vvooo,ooov=Vooov,
            oooo=Voooo)
