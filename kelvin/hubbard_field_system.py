import numpy
import lattice
from cqcpy import ft_utils
from cqcpy import utils

from .system import *
from .neq_system import *

class hubbard_field_system(system):
    """Hubbard model system with a time-dependent Pierls phase

    Attributes:
        T (float): Temperature.
        model: Object specifying details of the model.
        Pa: Mean-field alpha density
        Pa: Mean-field beta density
        mu (float): Chemical potential.
        Na (float): Number of alpha electrons.
        Nb (float): Number of beta electrons.
    """
    def __init__(self,T,model,ti,A0,t0,sigma,omega,phi=0.0,
            Pa=None,Pb=None,mu=None,na=None,nb=None,ua=None,ub=None):
        self.A0 = A0
        self.t0 = t0
        self.sigma = sigma
        self.omega = omega
        self.phi = phi
        self.ti = ti
        self.T = T
        self.model = model
        self.Pa = Pa
        self.Pb = Pb
        self.ot = None
        self.O = None
        if na is None:
            assert(nb is None)
            assert(mu is not None)
            self.mu = mu
            self.na = na
            self.nb = nb
            self.beta = 1.0 / self.T
        else:
            self.na = na
            self.nb = nb
            assert(na > 0)
            assert(nb > 0)
            assert(self.T == 0.0)
            self.mu = None
            self.beta = 1.0e20

        # Build T = 0 fock matrices
        self.orbtype = 'u'
        if Pa is None and Pb is None:
            if ua is None or ub is None:
                raise Exception("No reference provided")
            self.Pa = numpy.einsum('pi,qi->pq',ua[:na,:],ua[:na,:])
            self.Pb = numpy.einsum('pi,qi->pq',ua[:nb,:],ua[:nb,:])
            self.ua = ua
            self.ub = ub
        # build and diagonalize fock matrices
        V = self.model.get_umatS()
        Va = V - V.transpose((0,1,3,2))
        Fa = self.model.get_tmatS()
        Fb = self.model.get_tmatS()
        Fa += numpy.einsum('pqrs,qs->pr',Va,Pa)
        Fa += numpy.einsum('pqrs,qs->pr',V,Pb)
        Fb += numpy.einsum('pqrs,qs->pr',Va,Pb)
        Fb += numpy.einsum('pqrs,pr->qs',V,Pa)
        self.Fa = Fa
        self.Fb = Fb
        if ua is None:
            assert(ub is None)
            self.ea,self.ua = numpy.linalg.eigh(self.Fa)
            self.eb,self.ub = numpy.linalg.eigh(self.Fb)
        else:
            self.ea = numpy.einsum('ij,ip,jq->pq',self.Fa,self.ua,self.ua).diagonal()
            self.eb = numpy.einsum('ij,ip,jq->pq',self.Fb,self.ua,self.ub).diagonal()

    def verify(self,T,mu):
        if not (T == self.T and mu == self.mu):
            return False
        else:
            return True

    def const_energy(self):
        return 0.0

    def get_mp1(self):
        if self.T == 0:
            ea,eb = self.u_energies_tot()
            Va,Vb,Vabab = self.u_aint_tot()
            foa = numpy.zeros(ea.shape)
            fob = numpy.zeros(eb.shape)
            for i in range(self.na):
                foa[i] = 1.0
            for i in range(self.nb):
                fob[i] = 1.0
            E1 = -0.5*numpy.einsum('ijij,i,j->',Va,foa,foa)
            E1 -= 0.5*numpy.einsum('ijij,i,j->',Vb,fob,fob)
            E1 -= numpy.einsum('ijij,i,j->',Vabab,foa,fob)
            Fa,Fb = self.u_fock()
            Fao = Fa.oo - numpy.diag(self.ea[:self.na])
            Fbo = Fb.oo - numpy.diag(self.eb[:self.nb])
            E1 += numpy.einsum('ii->',Fao)
            E1 += numpy.einsum('ii->',Fbo)
            return E1
        else:
            Va,Vb,Vabab = self.u_aint_tot()
            ea,eb = self.u_energies_tot()
            na = ea.shape[0]
            nb = eb.shape[0]
            foa = ft_utils.ff(self.beta, ea, self.mu)
            fob = ft_utils.ff(self.beta, eb, self.mu)
            E1 = -0.5*numpy.einsum('ijij,i,j->',Va,foa,foa)
            E1 = -0.5*numpy.einsum('ijij,i,j->',Vb,fob,fob)
            E1 = -numpy.einsum('ijij,i,j->',Vabab,foa,fob)
            Ia = numpy.identity(na)
            Ib = numpy.identity(nb)
            dena = numpy.einsum('pi,i,qi->pq',Ia,foa,Ia)
            denb = numpy.einsum('pi,i,qi->pq',Ib,fob,Ib)
            JKa = numpy.einsum('prqs,rs->pq',Va,dena)
            JKa += numpy.einsum('prqs,rs->pq',Vabab,denb)
            JKb = numpy.einsum('prqs,rs->pq',Vb,denb)
            JKb += numpy.einsum('prqs,pq->rs',Vabab,dena)
            Ta = self.model.get_tmatS()
            Tb = Ta
            Fa = numpy.einsum('ij,ip,jq->pq',Ta,self.ua,self.ua)
            Fb = numpy.einsum('ij,ip,jq->pq',Tb,self.ub,self.ub)
            Fa += JKa.copy()
            Fb += JKb.copy()
            Fao = Fa - numpy.diag(ea)
            Fbo = Fb - numpy.diag(eb)
            E1 += numpy.einsum('ii,i->',Fao,foa)
            E1 += numpy.einsum('ii,i->',Fbo,fob)
            return E1

    def u_energies_tot(self):
        ea = self.ea
        eb = self.eb
        return ea,eb

    def g_energies_tot(self):
        ea = self.ea
        eb = self.eb
        return numpy.hstack((ea,eb))

    def u_fock_tot(self,direc='f'):
        da,db = self.u_energies_tot()
        nt = self.ti.shape[0]
        na = da.shape[0]
        nb = db.shape[0]
        assert(na == nb)
        Tt = numpy.zeros((nt,na,na),dtype=complex)
        for i,t in enumerate(self.ti):
            dt = t - self.t0
            ex = dt*dt/(2*self.sigma*self.sigma)
            phase = self.A0*numpy.exp(-ex)*numpy.cos(self.omega*dt + self.phi)
            Tt[i] = self.model.get_tmatS(phase=phase)
        foa = ft_utils.ff(self.beta, da, self.mu)
        fob = ft_utils.ff(self.beta, db, self.mu)
        Ia = numpy.identity(na)
        Ib = numpy.identity(nb)
        dena = numpy.einsum('pi,i,qi->pq',Ia,foa,Ia)
        denb = numpy.einsum('pi,i,qi->pq',Ib,fob,Ib)
        Va,Vb,Vabab = self.u_aint_tot()
        JKa = numpy.einsum('prqs,rs->pq',Va,dena)
        JKa += numpy.einsum('prqs,rs->pq',Vabab,denb)
        JKb = numpy.einsum('prqs,rs->pq',Vb,denb)
        JKb += numpy.einsum('prqs,pq->rs',Vabab,dena)
        Fa = numpy.einsum('yij,ip,jq->ypq',Tt,self.ua,self.ua)
        Fb = numpy.einsum('yij,ip,jq->ypq',Tt,self.ub,self.ub)
        Fa += JKa.copy()[None,:,:]
        Fb += JKb.copy()[None,:,:]
        if direc == 'b':
            tempa = Fa.copy()
            tempb = Fb.copy()
            ng = len(self.ti)
            for i in range(ng):
                Fa[i] = tempa[ng - i - 1]
                Fb[i] = tempb[ng - i - 1]
        return Fa,Fb

    def g_fock_tot(self,direc='f'):
        d = self.g_energies_tot()
        n = d.shape[0]
        nt = self.ti.shape[0]
        Tt = numpy.zeros((nt,n,n),dtype=complex)
        for i,t in enumerate(self.ti):
            dt = t - self.t0
            ex = dt*dt/(2*self.sigma*self.sigma)
            phase = self.A0*numpy.exp(-ex)*numpy.cos(self.omega*dt + self.phi)
            Tt[i] = self.model.get_tmat(phase=phase)
        fo = ft_utils.ff(self.beta, d, self.mu)
        I = numpy.identity(n)
        den = numpy.einsum('pi,i,qi->pq',I,fo,I)
        V = self.g_aint_tot()
        JK = numpy.einsum('prqs,rs->pq',V,den)
        Utot = utils.block_diag(self.ua,self.ub)
        Fock = numpy.einsum('yij,ip,jq->ypq',Tt,Utot,Utot)
        Fock += JK[None,:,:]
        if direc == 'b':
            temp = Fock.copy()
            ng = len(self.ti)
            for i in range(ng):
                Fock[i] = temp[ng - i - 1]
        return Fock

    def u_aint_tot(self):
        V = self.model.get_umatS()
        Va = V - V.transpose((0,1,3,2))
        Vabab = self._transform2(V,self.ua,self.ub,self.ua,self.ub)
        Vb = self._transform1(Va,self.ub)
        Va = self._transform1(Va,self.ua)
        return Va,Vb,Vabab

    def g_aint_tot(self):
        na = self.ea.shape[0]
        nb = self.eb.shape[0]
        n = na + nb

        Va,Vb,Vabab = self.u_aint_tot()
        U = numpy.zeros((n,n,n,n))
        U[:na,:na,:na,:na] = Va
        U[na:,na:,na:,na:] = Vb
        U[:na,na:,:na,na:] = Vabab
        U[na:,:na,:na,na:] = -Vabab.transpose((1,0,2,3))
        U[:na,na:,na:,:na] = -Vabab.transpose((0,1,3,2))
        U[na:,:na,na:,:na] = Vabab.transpose((1,0,3,2))
        return U

    def _transform1(self, V, u):
        return self._transform2(V, u, u, u, u)

    def _transform2(self, V, u1, u2, u3, u4):
        Umat2 = numpy.einsum('ijkl,ls->ijks',V,u4)
        Umat1 = numpy.einsum('ijks,kr->ijrs',Umat2,u3)
        Umat2 = numpy.einsum('ijrs,jq->iqrs',Umat1,u2)
        Umat1 = numpy.einsum('iqrs,ip->pqrs',Umat2,u1)
        return Umat1

class HubbardFieldSystem(NeqSystem):
    """Hubbard model system with a time-dependent Pierls phase

    Attributes:
        T (float): Temperature.
        model: Object specifying details of the model.
        Pa: Mean-field alpha density
        Pa: Mean-field beta density
        mu (float): Chemical potential.
        Na (float): Number of alpha electrons.
        Nb (float): Number of beta electrons.
    """
    def __init__(self,T,model,A0,t0,sigma,omega,phi=0.0,
            Pa=None,Pb=None,mu=None,na=None,nb=None,ua=None,ub=None):
        self.A0 = A0
        self.t0 = t0
        self.sigma = sigma
        self.omega = omega
        self.phi = phi
        #self.ti = ti
        self.T = T
        self.model = model
        self.Pa = Pa
        self.Pb = Pb
        self.ot = None
        self.O = None
        if na is None:
            assert(nb is None)
            assert(mu is not None)
            self.mu = mu
            self.na = na
            self.nb = nb
            self.beta = 1.0 / self.T
        else:
            self.na = na
            self.nb = nb
            assert(na > 0)
            assert(nb > 0)
            assert(self.T == 0.0)
            self.mu = None
            self.beta = 1.0e20

        # Build T = 0 fock matrices
        self.orbtype = 'u'
        if Pa is None and Pb is None:
            if ua is None or ub is None:
                raise Exception("No reference provided")
            self.Pa = numpy.einsum('pi,qi->pq',ua[:na,:],ua[:na,:])
            self.Pb = numpy.einsum('pi,qi->pq',ua[:nb,:],ua[:nb,:])
            self.ua = ua
            self.ub = ub
        # build and diagonalize fock matrices
        V = self.model.get_umatS()
        Va = V - V.transpose((0,1,3,2))
        Fa = self.model.get_tmatS()
        Fb = self.model.get_tmatS()
        Fa += numpy.einsum('pqrs,qs->pr',Va,Pa)
        Fa += numpy.einsum('pqrs,qs->pr',V,Pb)
        Fb += numpy.einsum('pqrs,qs->pr',Va,Pb)
        Fb += numpy.einsum('pqrs,pr->qs',V,Pa)
        self.Fa = Fa
        self.Fb = Fb
        if ua is None:
            assert(ub is None)
            self.ea,self.ua = numpy.linalg.eigh(self.Fa)
            self.eb,self.ub = numpy.linalg.eigh(self.Fb)
        else:
            self.ea = numpy.einsum('ij,ip,jq->pq',self.Fa,self.ua,self.ua).diagonal()
            self.eb = numpy.einsum('ij,ip,jq->pq',self.Fb,self.ua,self.ub).diagonal()

    def verify(self,T,mu):
        if not (T == self.T and mu == self.mu):
            return False
        else:
            return True

    def has_g(self): return True
    def has_u(self): return False
    def has_r(self): return False

    def const_energy(self):
        return 0.0

    def get_mp1(self):
        if self.T == 0:
            ea,eb = self.u_energies_tot()
            Va,Vb,Vabab = self.u_aint_tot()
            foa = numpy.zeros(ea.shape)
            fob = numpy.zeros(eb.shape)
            for i in range(self.na):
                foa[i] = 1.0
            for i in range(self.nb):
                fob[i] = 1.0
            E1 = -0.5*numpy.einsum('ijij,i,j->',Va,foa,foa)
            E1 -= 0.5*numpy.einsum('ijij,i,j->',Vb,fob,fob)
            E1 -= numpy.einsum('ijij,i,j->',Vabab,foa,fob)
            Fa,Fb = self.u_fock()
            Fao = Fa.oo - numpy.diag(self.ea[:self.na])
            Fbo = Fb.oo - numpy.diag(self.eb[:self.nb])
            E1 += numpy.einsum('ii->',Fao)
            E1 += numpy.einsum('ii->',Fbo)
            return E1
        else:
            Va,Vb,Vabab = self.u_aint_tot()
            ea,eb = self.u_energies_tot()
            na = ea.shape[0]
            nb = eb.shape[0]
            foa = ft_utils.ff(self.beta, ea, self.mu)
            fob = ft_utils.ff(self.beta, eb, self.mu)
            E1 = -0.5*numpy.einsum('ijij,i,j->',Va,foa,foa)
            E1 = -0.5*numpy.einsum('ijij,i,j->',Vb,fob,fob)
            E1 = -numpy.einsum('ijij,i,j->',Vabab,foa,fob)
            Ia = numpy.identity(na)
            Ib = numpy.identity(nb)
            dena = numpy.einsum('pi,i,qi->pq',Ia,foa,Ia)
            denb = numpy.einsum('pi,i,qi->pq',Ib,fob,Ib)
            JKa = numpy.einsum('prqs,rs->pq',Va,dena)
            JKa += numpy.einsum('prqs,rs->pq',Vabab,denb)
            JKb = numpy.einsum('prqs,rs->pq',Vb,denb)
            JKb += numpy.einsum('prqs,pq->rs',Vabab,dena)
            Ta = self.model.get_tmatS()
            Tb = Ta
            Fa = numpy.einsum('ij,ip,jq->pq',Ta,self.ua,self.ua)
            Fb = numpy.einsum('ij,ip,jq->pq',Tb,self.ub,self.ub)
            Fa += JKa.copy()
            Fb += JKb.copy()
            Fao = Fa - numpy.diag(ea)
            Fbo = Fb - numpy.diag(eb)
            E1 += numpy.einsum('ii,i->',Fao,foa)
            E1 += numpy.einsum('ii,i->',Fbo,fob)
            return E1

    def u_energies_tot(self):
        ea = self.ea
        eb = self.eb
        return ea,eb

    def g_energies_tot(self):
        ea = self.ea
        eb = self.eb
        return numpy.hstack((ea,eb))

    def u_fock_tot(self, t=0):
        da,db = self.u_energies_tot()
        na = da.shape[0]
        nb = db.shape[0]
        assert(na == nb)
        dt = t - self.t0
        ex = dt*dt/(2*self.sigma*self.sigma)
        phase = self.A0*numpy.exp(-ex)*numpy.cos(self.omega*dt + self.phi)
        Tt = self.model.get_tmatS(phase=phase)
        foa = ft_utils.ff(self.beta, da, self.mu)
        fob = ft_utils.ff(self.beta, db, self.mu)
        Ia = numpy.identity(na)
        Ib = numpy.identity(nb)
        dena = numpy.einsum('pi,i,qi->pq',Ia,foa,Ia)
        denb = numpy.einsum('pi,i,qi->pq',Ib,fob,Ib)
        Va,Vb,Vabab = self.u_aint_tot()
        JKa = numpy.einsum('prqs,rs->pq',Va,dena)
        JKa += numpy.einsum('prqs,rs->pq',Vabab,denb)
        JKb = numpy.einsum('prqs,rs->pq',Vb,denb)
        JKb += numpy.einsum('prqs,pq->rs',Vabab,dena)
        Fa = numpy.einsum('ij,ip,jq->pq',Tt,self.ua,self.ua)
        Fb = numpy.einsum('ij,ip,jq->pq',Tt,self.ub,self.ub)
        Fa += JKa.copy()
        Fb += JKb.copy()
        return Fa,Fb

    def g_fock_tot(self, t=0):
        d = self.g_energies_tot()
        n = d.shape[0]
        dt = t - self.t0
        ex = dt*dt/(2*self.sigma*self.sigma)
        phase = self.A0*numpy.exp(-ex)*numpy.cos(self.omega*dt + self.phi)
        Tt = self.model.get_tmat(phase=phase)
        fo = ft_utils.ff(self.beta, d, self.mu)
        I = numpy.identity(n)
        den = numpy.einsum('pi,i,qi->pq',I,fo,I)
        V = self.g_aint_tot()
        JK = numpy.einsum('prqs,rs->pq',V,den)
        Utot = utils.block_diag(self.ua,self.ub)
        Fock = numpy.einsum('ij,ip,jq->pq',Tt,Utot,Utot)
        Fock += JK
        return Fock

    def u_aint_tot(self):
        V = self.model.get_umatS()
        Va = V - V.transpose((0,1,3,2))
        Vabab = self._transform2(V,self.ua,self.ub,self.ua,self.ub)
        Vb = self._transform1(Va,self.ub)
        Va = self._transform1(Va,self.ua)
        return Va,Vb,Vabab

    def g_aint_tot(self):
        na = self.ea.shape[0]
        nb = self.eb.shape[0]
        n = na + nb

        Va,Vb,Vabab = self.u_aint_tot()
        U = numpy.zeros((n,n,n,n))
        U[:na,:na,:na,:na] = Va
        U[na:,na:,na:,na:] = Vb
        U[:na,na:,:na,na:] = Vabab
        U[na:,:na,:na,na:] = -Vabab.transpose((1,0,2,3))
        U[:na,na:,na:,:na] = -Vabab.transpose((0,1,3,2))
        U[na:,:na,na:,:na] = Vabab.transpose((1,0,3,2))
        return U

    def _transform1(self, V, u):
        return self._transform2(V, u, u, u, u)

    def _transform2(self, V, u1, u2, u3, u4):
        Umat2 = numpy.einsum('ijkl,ls->ijks',V,u4)
        Umat1 = numpy.einsum('ijks,kr->ijrs',Umat2,u3)
        Umat2 = numpy.einsum('ijrs,jq->iqrs',Umat1,u2)
        Umat1 = numpy.einsum('iqrs,ip->pqrs',Umat2,u1)
        return Umat1
