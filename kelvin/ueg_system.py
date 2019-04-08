import numpy
from pyscf import lib
from cqcpy import ft_utils
from cqcpy.ov_blocks import one_e_blocks
from cqcpy.ov_blocks import two_e_blocks
from cqcpy.ov_blocks import two_e_blocks_full
from .system import system
from .ueg_utils import ueg_basis
from . import zt_mp
from . import ft_mp

einsum = lib.einsum
#einsum = einsum

class ueg_system(system):
    """The uniform electron gas in a plane-wave basis set.
    
    Attributes: 
        T (float): Temperature.
        L (float): Box-length.
        basis: UEG plane-wave basis set.
        mu (float): Chemical potential.
        Na (float): Number of alpha electrons.
        Nb (float): Number of beta electrons.
        N (float): Total number of electrons.
        den (float): Number density.
        rs (float): Wigner-Seitz radius.
        Ef (float): Fermi-energy (of non-interacting system).
        Tf (float): Redued temperature.
    """
    def __init__(self,T,L,Emax,mu=None,na=None,nb=None,norb=None,orbtype='u',madelung=None):
        self.T = T
        self.L = L
        self.basis = ueg_basis(L,Emax,norb=norb)
        if na is None:
            assert(nb is None)
            assert(mu is not None)
            self.mu = mu
            beta = 1.0 / (self.T + 1e-12)
            en = self.g_energies_tot()
            fo = ft_utils.ff(beta, en, self.mu)
            N = fo.sum()
            self.Na = N/2.0
            self.Nb = self.Na 
        else:
            self.Na = na
            self.Nb = nb
            assert(na > 0)
            assert(nb > 0)
            mua = self.basis.Es[self.Na - 1] + 0.00001
            mub = self.basis.Es[self.Nb - 1] + 0.00001
            assert(mua == mub)
            self.mu = mua
            assert(self.T == 0.0)

        self.N = self.Na + self.Nb
        self.den = self.N/(L*L*L)
        self.rs = (3/(4.0*numpy.pi*self.den))**(1.0/3.0)
        pi2 = numpy.pi*numpy.pi
        self.Ef = 0.5*(3.0*pi2*self.den)**(2.0/3.0)
        self.Tf = self.T / self.Ef
        self.orbtype = orbtype
        self.madelung=madelung
        self._mconst = 2.837297479 / (2*self.L)

    def has_g(self):
        return True

    def has_u(self):
        if self.orbtype == 'g':
            return False
        else:
            return True

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
        if self.madelung == 'const':
            return -(self.Na + self.Nb)*self._mconst
        else:
            return 0.0

    def get_mp1(self):
        if self.has_u():
            if self.T > 0:
                Va,Vb,Vabab = self.u_aint_tot()
                beta = 1.0 / (self.T + 1e-12)
                ea,eb = self.u_energies_tot()
                foa = ft_utils.ff(beta, ea, self.mu)
                fob = ft_utils.ff(beta, eb, self.mu)
                E1 = 0.5*einsum('ijij,i,j->',Va,foa,foa)
                E1 += 0.5*einsum('ijij,i,j->',Vb,fob,fob)
                E1 += einsum('ijij,i,j->',Vabab,foa,fob)
                return E1
            else:
                Va,Vb,Vabab = self.u_aint()
                beta = 1.0 / (self.T + 1e-12)
                #ea,eb = self.u_energies_tot()
                #foa = ft_utils.ff(beta, ea, self.mu)
                #fob = ft_utils.ff(beta, eb, self.mu)
                E1 = 0.5*numpy.einsum('ijij->',Va.oooo)
                E1 += 0.5*numpy.einsum('ijij->',Vb.oooo)
                E1 += numpy.einsum('ijij->',Vabab.oooo)
                return E1
        else:
            if self.T > 0:
                V = self.g_aint_tot()
                beta = 1.0 / (self.T + 1e-12)
                en = self.g_energies_tot()
                fo = ft_utils.ff(beta, en, self.mu)
                return 0.5*einsum('ijij,i,j->',
                    V,fo,fo)
            else:
                V = self.g_aint()
                return 0.5*einsum('ijij->',V.oooo)

    def u_d_mp1(self,dveca,dvecb):
        if self.T > 0:
            Va,Vb,Vabab = self.u_aint_tot()
            beta = 1.0 / (self.T + 1e-12)
            ea,eb = self.u_energies_tot()
            foa = ft_utils.ff(beta, ea, self.mu)
            fva = ft_utils.ffv(beta, ea, self.mu)
            veca = dveca*foa*fva
            fob = ft_utils.ff(beta, eb, self.mu)
            fvb = ft_utils.ffv(beta, eb, self.mu)
            vecb = dvecb*fob*fvb
            D = -einsum('ijij,i,j->',Va,veca,foa)
            D -= einsum('ijij,i,j->',Vb,vecb,fob)
            D -= einsum('ijij,i,j->',Vabab,veca,fob)
            D -= einsum('ijij,i,j->',Vabab,foa,vecb)
            return D
        else:
            print("WARNING: Derivative of MP1 energy is zero at OK")
            return 0.0

    def u_mp1_den(self):
        if self.T > 0:
            Va,Vb,Vabab = self.u_aint_tot()
            beta = 1.0 / (self.T + 1e-12)
            ea,eb = self.u_energies_tot()
            foa = ft_utils.ff(beta, ea, self.mu)
            fva = ft_utils.ffv(beta, ea, self.mu)
            veca = foa*fva
            fob = ft_utils.ff(beta, eb, self.mu)
            fvb = ft_utils.ffv(beta, eb, self.mu)
            vecb = fob*fvb
            Da = -beta*einsum('ijij,i,j->i',Va,veca,foa)
            Db = -beta*einsum('ijij,i,j->i',Vb,vecb,fob)
            Da -= beta*einsum('ijij,i,j->i',Vabab,veca,fob)
            Db -= beta*einsum('ijij,i,j->j',Vabab,foa,vecb)
            return Da,Db
        else:
            print("WARNING: Derivative of MP1 energy is zero at OK")
            return numpy.zeros(ea.shape),numpy.zeros(eb.shape)

    def g_d_mp1(self,dvec):
        if self.T > 0:
            V = self.g_aint_tot()
            beta = 1.0 / (self.T + 1e-12)
            en = self.g_energies_tot()
            fo = ft_utils.ff(beta, en, self.mu)
            fv = ft_utils.ffv(beta, en, self.mu)
            vec = dvec*fo*fv
            return -einsum('ijij,i,j->',V,vec,fo)
        else:
            print("WARNING: Derivative of MP1 energy is zero at OK")
            return 0.0

    def g_mp1_den(self):
        if self.T > 0:
            V = self.g_aint_tot()
            beta = 1.0 / (self.T + 1e-12)
            en = self.g_energies_tot()
            fo = ft_utils.ff(beta, en, self.mu)
            fv = ft_utils.ffv(beta, en, self.mu)
            vec = fo*fv
            return -beta*einsum('ijij,i,j->i',V,vec,fo)
        else:
            print("WARNING: Derivative of MP1 energy is zero at OK")
            return numpy.zeros(self.g_energies_tot.shape)

    def r_energies(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        if self.Na != self.Nb:
            raise Exception("UEG system is not restricted")
        d = self.basis.Es
        na = int(self.Na)
        eo = d[:na]
        ev = d[na:]
        if self.madelung == "orb":
            eo -= self._mconst
        return (eo,ev)

    def u_energies(self):
        d = self.basis.Es
        na = int(self.Na)
        nb = int(self.Nb)
        eoa = numpy.asarray(d[:na])
        eva = numpy.asarray(d[na:])
        eob = numpy.asarray(d[:nb])
        evb = numpy.asarray(d[nb:])
        if self.madelung == "orb":
            eoa -= self._mconst
            eob -= self._mconst
        return (eoa,eva,eob,evb)

    def g_energies(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        d = self.g_energies_tot()
        nbsf = self.basis.get_nbsf()
        na = int(self.Na)
        nb = int(self.Nb)
        eoa = d[:na]
        eva = d[na:nbsf]
        eob = d[nbsf:nbsf+nb]
        evb = d[-(nbsf-nb):]
        eo = numpy.hstack((eoa,eob))
        ev = numpy.hstack((eva,evb))
        if self.madelung == "orb":
            eo -= self._mconst
        return (eo,ev)

    def r_energies_tot(self):
        return numpy.asarray(self.basis.Es)

    def u_energies_tot(self):
        return self.basis.u_build_diag()

    def g_energies_tot(self):
        return self.basis.g_build_diag()
    
    def r_fock(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        F = self.r_hcore()
        d = self.r_energies_tot()
        mu = self.mu
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
        V = self.r_int_tot()
        Vd = V[numpy.ix_(numpy.arange(n),oidx,numpy.arange(n),oidx)]
        Vx = V[numpy.ix_(numpy.arange(n),oidx,oidx,numpy.arange(n))]
        F = F + 2*einsum('piri->pr',Vd) - einsum('piir->pr',Vx)
        Foo = F[numpy.ix_(oidx,oidx)]
        Fvv = F[numpy.ix_(vidx,vidx)]
        Fov = F[numpy.ix_(oidx,vidx)]
        Fvo = F[numpy.ix_(vidx,oidx)]
        return one_e_blocks(Foo,Fov,Fvo,Fvv)

    def u_fock(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        F = self.r_hcore()
        d = self.r_energies_tot()
        mu = self.mu
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
        V = self.r_int_tot()
        Vd = V[numpy.ix_(numpy.arange(n),oidx,numpy.arange(n),oidx)]
        Vx = V[numpy.ix_(numpy.arange(n),oidx,oidx,numpy.arange(n))]
        F = F + 2*einsum('piri->pr',Vd) - einsum('piir->pr',Vx)
        Foo = F[numpy.ix_(oidx,oidx)]
        Fvv = F[numpy.ix_(vidx,vidx)]
        Fov = F[numpy.ix_(oidx,vidx)]
        Fvo = F[numpy.ix_(vidx,oidx)]
        Fa = one_e_blocks(Foo,Fov,Fvo,Fvv)
        Fb = one_e_blocks(Foo,Fov,Fvo,Fvv)
        return Fa,Fb

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
        #Vd = Vb[numpy.ix_(numpy.arange(n),oidx,numpy.arange(n),oidx)]
        #Vx = Vb[numpy.ix_(numpy.arange(n),oidx,oidx,numpy.arange(n))]
        #Vb = Vd - Vx.transpose((0,1,3,2))
        F = F + einsum('piri->pr',V)# - einsum('piir->pr',V.transpose((0,1,3,2)))
        Foo = F[numpy.ix_(oidx,oidx)]
        Fvv = F[numpy.ix_(vidx,vidx)]
        Fov = F[numpy.ix_(oidx,vidx)]
        Fvo = F[numpy.ix_(vidx,oidx)]
        return one_e_blocks(Foo,Fov,Fvo,Fvv)

    def r_fock_tot(self):
        T = self.r_hcore()
        d = self.r_energies_tot()
        n = d.shape[0]
        if self.T > 0.0:
            beta = 1.0 / (self.T + 1e-12)
            fo = ft_utils.ff(beta, d, self.mu)
            I = numpy.identity(n)
            den = einsum('pi,i,qi->pq',I,fo,I)
        else:
            to = numpy.zeros((n,self.N))
            i = 0
            for p in range(n):
                if d[p] < self.mu:
                    to[p,i] = 1.0
                    i = i+1
            den = einsum('pi,qi->pq',to,to)
        V = self.r_int_tot()
        JK = 2*einsum('prqs,rs->pq',V,den) - einsum('prsq,rs->pq',V,den)
        return T + JK

    def u_fock_tot(self):
        Ta,Tb = self.basis.build_u_ke_matrix()
        da,db = self.u_energies_tot()
        na = da.shape[0]
        nb = db.shape[0]
        if self.T > 0.0:
            beta = 1.0 / (self.T + 1e-12)
            foa = ft_utils.ff(beta, da, self.mu)
            fob = ft_utils.ff(beta, db, self.mu)
            Ia = numpy.identity(na)
            Ib = numpy.identity(nb)
            dena = einsum('pi,i,qi->pq',Ia,foa,Ia)
            denb = einsum('pi,i,qi->pq',Ib,fob,Ib)
        else:
            toa = numpy.zeros((na,self.N))
            tob = numpy.zeros((nb,self.N))
            i = 0
            for p in range(na):
                if da[p] < self.mu:
                    toa[p,i] = 1.0
                    i = i+1
            for p in range(nb):
                if db[p] < self.mu:
                    tob[p,i] = 1.0
                    i = i+1
            dena = einsum('pi,qi->pq',toa,toa)
            denb = einsum('pi,qi->pq',tob,tob)
        Va,Vb,Vabab = self.u_aint_tot()
        JKa = einsum('prqs,rs->pq',Va,dena)
        JKa += einsum('prqs,rs->pq',Vabab,denb)
        JKb = einsum('prqs,rs->pq',Vb,denb)
        JKb += einsum('prqs,rs->pq',Vabab,dena)
        return (Ta + JKa),(Tb + JKb)

    def g_fock_tot(self):
        T = self.basis.build_g_ke_matrix()
        d = self.g_energies_tot()
        n = d.shape[0]
        if self.T > 0.0:
            beta = 1.0 / (self.T + 1e-12)
            fo = ft_utils.ff(beta, d, self.mu)
            I = numpy.identity(n)
            den = einsum('pi,i,qi->pq',I,fo,I)
        else:
            to = numpy.zeros((n,self.N))
            i = 0
            for p in range(n):
                if d[p] < self.mu:
                    to[p,i] = 1.0
                    i = i+1
            den = einsum('pi,qi->pq',to,to)
        V = self.g_aint_tot()
        JK = einsum('prqs,rs->pq',V,den)
        return T + JK

    def u_fock_d_tot(self,dveca,dvecb):
        da,db = self.u_energies_tot()
        na = da.shape[0]
        nb = db.shape[0]
        if self.T == 0.0:
            print("WARNING: Occupations derivatives are zero at 0K")
            return numpy.zeros((n,n)),numpy.zeros((n,n))
        beta = 1.0 / (self.T + 1e-12)
        foa = ft_utils.ff(beta, da, self.mu)
        fva = ft_utils.ffv(beta, da, self.mu)
        veca = dveca*foa*fva
        fob = ft_utils.ff(beta, db, self.mu)
        fvb = ft_utils.ffv(beta, db, self.mu)
        vecb = dvecb*fob*fvb
        Ia = numpy.identity(na)
        Ib = numpy.identity(nb)
        dena = einsum('pi,i,qi->pq',Ia,veca,Ia)
        denb = einsum('pi,i,qi->pq',Ib,vecb,Ib)
        Va,Vb,Vabab = self.u_aint_tot()
        JKa = einsum('prqs,rs->pq',Va,dena)
        JKa += einsum('prqs,rs->pq',Vabab,denb)
        JKb = einsum('prqs,rs->pq',Vb,denb)
        JKb += einsum('prqs,pq->rs',Vabab,dena)
        return -JKa,-JKb

    def u_fock_d_den(self):
        da,db = self.u_energies_tot()
        na = da.shape[0]
        nb = db.shape[0]
        if self.T == 0.0:
            print("WARNING: Occupations derivatives are zero at 0K")
            return numpy.zeros((na,na)),numpy.zeros((na,na))
        beta = 1.0 / (self.T + 1e-12)
        foa = ft_utils.ff(beta, da, self.mu)
        fva = ft_utils.ffv(beta, da, self.mu)
        veca = foa*fva
        fob = ft_utils.ff(beta, db, self.mu)
        fvb = ft_utils.ffv(beta, db, self.mu)
        vecb = fob*fvb
        #Ia = numpy.identity(na)
        #Ib = numpy.identity(nb)
        #dena = einsum('pi,i,qi->pq',Ia,veca,Ia)
        #denb = einsum('pi,i,qi->pq',Ib,vecb,Ib)
        Va,Vb,Vabab = self.u_aint_tot()
        JKaa = einsum('piqi,i->pqi',Va,veca)
        JKab = einsum('piqi,i->pqi',Vabab,vecb)
        JKbb = einsum('piqi,i->pqi',Vb,vecb)
        JKba = einsum('iris,i->rsi',Vabab,veca)
        return JKaa,JKab,JKbb,JKba

    def g_fock_d_tot(self,dvec):
        d = self.g_energies_tot()
        n = d.shape[0]
        if self.T == 0.0:
            print("WARNING: Occupations derivatives are zero at 0K")
            return numpy.zeros((n,n))
        beta = 1.0 / (self.T + 1e-12)
        fo = ft_utils.ff(beta, d, self.mu)
        fv = ft_utils.ffv(beta, d, self.mu)
        vec = dvec*fo*fv
        I = numpy.identity(n)
        den = einsum('pi,i,qi->pq',I,vec,I)
        V = self.g_aint_tot()
        JK = einsum('prqs,rs->pq',V,den)
        return -JK

    def g_fock_d_den(self):
        d = self.g_energies_tot()
        n = d.shape[0]
        if self.T == 0.0:
            print("WARNING: Occupations derivatives are zero at 0K")
            return numpy.zeros((n,n))
        beta = 1.0 / (self.T + 1e-12)
        fo = ft_utils.ff(beta, d, self.mu)
        fv = ft_utils.ffv(beta, d, self.mu)
        vec = fo*fv
        #I = numpy.identity(n)
        #den = einsum('pi,i,qi->pq',I,vec,I)
        V = self.g_aint_tot()
        JK = einsum('piqi,i->pqi',V,vec)
        return JK

    def r_hcore(self):
        return numpy.diag(self.r_energies_tot())

    def g_hcore(self):
        return self.basis.build_g_ke_matrix()

    def u_aint(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        da,db = self.u_energies_tot()
        na = da.shape[0]
        nb = db.shape[0]
        occa = []
        vira = []
        for p in range(na):
            if da[p] < self.mu:
                occa.append(p)
            if da[p] > self.mu:
                vira.append(p)
        occb = []
        virb = []
        for p in range(nb):
            if db[p] < self.mu:
                occb.append(p)
            if db[p] > self.mu:
                virb.append(p)
        Va,Vb,Vabab = self.u_aint_tot()
        oaidx = numpy.r_[occa]
        vaidx = numpy.r_[vira]
        obidx = numpy.r_[occb]
        vbidx = numpy.r_[virb]

        Vvvvv = Va[numpy.ix_(vaidx,vaidx,vaidx,vaidx)]
        Vvvvo = Va[numpy.ix_(vaidx,vaidx,vaidx,oaidx)] 
        Vvovv = Va[numpy.ix_(vaidx,oaidx,vaidx,vaidx)]
        Vvvoo = Va[numpy.ix_(vaidx,vaidx,oaidx,oaidx)]
        Vvovo = Va[numpy.ix_(vaidx,oaidx,vaidx,oaidx)]
        Voovv = Va[numpy.ix_(oaidx,oaidx,vaidx,vaidx)]
        Vvooo = Va[numpy.ix_(vaidx,oaidx,oaidx,oaidx)]
        Vooov = Va[numpy.ix_(oaidx,oaidx,oaidx,vaidx)]
        Voooo = Va[numpy.ix_(oaidx,oaidx,oaidx,oaidx)]
        Va = two_e_blocks(
            vvvv=Vvvvv,vvvo=Vvvvo,
            vovv=Vvovv,vvoo=Vvvoo,
            vovo=Vvovo,oovv=Voovv,
            vooo=Vvooo,ooov=Vooov,
            oooo=Voooo)
        Vvvvv = Vb[numpy.ix_(vbidx,vbidx,vbidx,vbidx)]
        Vvvvo = Vb[numpy.ix_(vbidx,vbidx,vbidx,obidx)] 
        Vvovv = Vb[numpy.ix_(vbidx,obidx,vbidx,vbidx)]
        Vvvoo = Vb[numpy.ix_(vbidx,vbidx,obidx,obidx)]
        Vvovo = Vb[numpy.ix_(vbidx,obidx,vbidx,obidx)]
        Voovv = Vb[numpy.ix_(obidx,obidx,vbidx,vbidx)]
        Vvooo = Vb[numpy.ix_(vbidx,obidx,obidx,obidx)]
        Vooov = Vb[numpy.ix_(obidx,obidx,obidx,vbidx)]
        Voooo = Vb[numpy.ix_(obidx,obidx,obidx,obidx)]
        Vb = two_e_blocks(
            vvvv=Vvvvv,vvvo=Vvvvo,
            vovv=Vvovv,vvoo=Vvvoo,
            vovo=Vvovo,oovv=Voovv,
            vooo=Vvooo,ooov=Vooov,
            oooo=Voooo)

        Vvvvv = Vabab[numpy.ix_(vaidx,vbidx,vaidx,vbidx)]
        Vvvvo = Vabab[numpy.ix_(vaidx,vbidx,vaidx,obidx)] 
        Vvvov = Vabab[numpy.ix_(vaidx,vbidx,oaidx,vbidx)] 
        Vvovv = Vabab[numpy.ix_(vaidx,obidx,vaidx,vbidx)]
        Vovvv = Vabab[numpy.ix_(oaidx,vbidx,vaidx,vbidx)]
        Vvvoo = Vabab[numpy.ix_(vaidx,vbidx,oaidx,obidx)]
        Vvoov = Vabab[numpy.ix_(vaidx,obidx,oaidx,vbidx)]
        Vvovo = Vabab[numpy.ix_(vaidx,obidx,vaidx,obidx)]
        Vovvo = Vabab[numpy.ix_(oaidx,vbidx,vaidx,obidx)]
        Vovov = Vabab[numpy.ix_(oaidx,vbidx,oaidx,vbidx)]
        Voovv = Vabab[numpy.ix_(oaidx,obidx,vaidx,vbidx)]
        Vvooo = Vabab[numpy.ix_(vaidx,obidx,oaidx,obidx)]
        Vovoo = Vabab[numpy.ix_(oaidx,vbidx,oaidx,obidx)]
        Voovo = Vabab[numpy.ix_(oaidx,obidx,vaidx,obidx)]
        Vooov = Vabab[numpy.ix_(oaidx,obidx,oaidx,vbidx)]
        Voooo = Vabab[numpy.ix_(oaidx,obidx,oaidx,obidx)]
        Vabab = two_e_blocks_full(vvvv=Vvvvv,
                vvvo=Vvvvo,vvov=Vvvov,
                vovv=Vvovv,ovvv=Vovvv,
                vvoo=Vvvoo,vovo=Vvovo,
                ovvo=Vovvo,voov=Vvoov,
                ovov=Vovov,oovv=Voovv,
                vooo=Vvooo,ovoo=Vovoo,
                oovo=Voovo,ooov=Vooov,
                oooo=Voooo)
        return Va,Vb,Vabab

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

    def u_aint_tot(self):
        return self.basis.build_u2e_matrix()

    def g_aint_tot(self):
        return self.basis.build_g2e_matrix()

    def r_int_tot(self):
        return self.basis.build_r2e_matrix()

    def g_int_tot(self):
        return self.basis.build_g2e_matrix(anti=False)
