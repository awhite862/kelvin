import numpy
import lattice
from cqcpy import ft_utils
from cqcpy.ov_blocks import one_e_blocks
from cqcpy.ov_blocks import two_e_blocks
from cqcpy.ov_blocks import two_e_blocks_full
from cqcpy import utils
from . import zt_mp
from .system import system

einsum = numpy.einsum

class HubbardSystem(system):
    """Hubbard model system in a mean-field basis

    Attributes:
        T (float): Temperature.
        model: Object specifying details of the model.
        Pa: Mean-field alpha density
        Pa: Mean-field beta density
        mu (float): Chemical potential.
        Na (float): Number of alpha electrons.
        Nb (float): Number of beta electrons.
    """
    def __init__(self,T,model,Pa=None,Pb=None,mu=None,na=None,nb=None,ua=None,ub=None,orbtype='u'):
        self.T = T
        self.model = model
        self.Pa = Pa
        self.Pb = Pb
        self.orbtype = orbtype
        if na is None:
            assert(nb is None)
            assert(mu is not None)
            self.mu = mu
            self.na = na
            self.nb = nb
            self.beta = 1.0 / self.T if self.T > 0.0 else 1.0e20
        else:
            self.na = na
            self.nb = nb
            assert(na > 0)
            assert(nb > 0)
            assert(self.T == 0.0)
            self.beta = 1.0e20
            self.mu = None

        # Build T = 0 fock matrices
        if Pa is None and Pb is None:
            if ua is None or ub is None :
                raise Exception("No reference provided")
            if na is None or nb is None :
                raise Exception("No reference provided")
            self.Pa = numpy.einsum('pi,qi->pq',ua[:na,:],ua[:na,:])
            self.Pb = numpy.einsum('pi,qi->pq',ua[:nb,:],ua[:nb,:])
            self.ua = ua
            self.ub = ub
        # build and diagonalize fock matrices
        V = self.model.get_umatS()
        Va = V - V.transpose((0,1,3,2))
        Fa = self.r_hcore()
        Fb = self.r_hcore()
        Fa += numpy.einsum('pqrs,qs->pq',Va,Pa)
        Fa += numpy.einsum('pqrs,qs->pq',V,Pb)
        Fb += numpy.einsum('pqrs,qs->pq',Va,Pb)
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

    def has_g(self):
        return True

    def has_u(self):
        if self.orbtype == 'g':
            return False
        else:
            return True

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

    # TODO: clean this up
    def get_mp1(self):
        if self.T == 0:
            # orbital energies
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
            foa = ft_utils.ff(self.beta, ea, self.mu)
            fob = ft_utils.ff(self.beta, eb, self.mu)
            E1 = -0.5*numpy.einsum('ijij,i,j->',Va,foa,foa)
            E1 = -0.5*numpy.einsum('ijij,i,j->',Vb,fob,fob)
            E1 = -numpy.einsum('ijij,i,j->',Vabab,foa,fob)
            Fa,Fb = self.u_fock_tot()
            Fao = Fa - numpy.diag(ea)
            Fbo = Fb - numpy.diag(eb)
            E1 += numpy.einsum('ii,i->',Fao,foa)
            E1 += numpy.einsum('ii,i->',Fbo,fob)
            return E1

    def u_d_mp1(self,dveca,dvecb):
        if self.T > 0:
            Va,Vb,Vabab = self.u_aint_tot()
            ea,eb = self.u_energies_tot()
            foa = ft_utils.ff(self.beta, ea, self.mu)
            fva = ft_utils.ffv(self.beta, ea, self.mu)
            veca = dveca*foa*fva
            fob = ft_utils.ff(self.beta, eb, self.mu)
            fvb = ft_utils.ffv(self.beta, eb, self.mu)
            vecb = dvecb*fob*fvb
            Fa,Fb = self.u_fock_tot()
            D = -einsum('ii,i->',Fa - numpy.diag(ea),veca)
            D += -einsum('ii,i->',Fb - numpy.diag(eb),vecb)
            D += einsum('ijij,i,j->',Va,veca,foa)
            D += einsum('ijij,i,j->',Vb,vecb,fob)
            D += einsum('ijij,i,j->',Vabab,veca,fob)
            D += einsum('ijij,i,j->',Vabab,foa,vecb)
            Fa,Fb = self.u_fock_d_tot(dveca,dvecb)
            D += einsum('ii,i->',Fa,foa)
            D += einsum('ii,i->',Fb,fob)
            return D
        else:
            print("WARNING: Derivative of MP1 energy is zero at OK")
            return 0.0

    def g_d_mp1(self,dvec):
        if self.T > 0:
            V = self.g_aint_tot()
            en = self.g_energies_tot()
            fo = ft_utils.ff(self.beta, en, self.mu)
            fv = ft_utils.ffv(self.beta, en, self.mu)
            vec = dvec*fo*fv
            F = self.g_fock_tot()
            D = -einsum('ii,i->',F - numpy.diag(en),vec)
            D += einsum('ijij,i,j->',V,vec,fo)
            F = self.g_fock_d_tot(dvec)
            D += einsum('ii->i',F,fo)
            return D
        else:
            print("WARNING: Derivative of MP1 energy is zero at OK")
            return 0.0

    def u_mp1_den(self):
        if self.T > 0:
            Va,Vb,Vabab = self.u_aint_tot()
            beta = self.beta
            ea,eb = self.u_energies_tot()
            foa = ft_utils.ff(beta, ea, self.mu)
            fva = ft_utils.ffv(beta, ea, self.mu)
            veca = foa*fva
            fob = ft_utils.ff(beta, eb, self.mu)
            fvb = ft_utils.ffv(beta, eb, self.mu)
            vecb = fob*fvb
            tmat = self.r_hcore()
            T = self.model.get_tmatS()
            Ta = numpy.einsum('ij,ip,jq->pq',T,self.ua,self.ua)
            Tb = numpy.einsum('ij,ip,jq->pq',T,self.ub,self.ub)
            Da = -beta*einsum('ii,i->i',Ta - numpy.diag(ea),veca)
            Db = -beta*einsum('ii,i->i',Tb - numpy.diag(eb),vecb)
            Da += -beta*einsum('ijij,i,j->i',Va,veca,foa)
            Db += -beta*einsum('ijij,i,j->i',Vb,vecb,fob)
            Da += -beta*einsum('ijij,i,j->i',Vabab,veca,fob)
            Db += -beta*einsum('ijij,i,j->j',Vabab,foa,vecb)
            return Da,Db
        else:
            print("WARNING: Derivative of MP1 energy is zero at OK")
            return 0.0

    def g_mp1_den(self):
        if self.T > 0:
            V = self.g_aint_tot()
            beta = self.beta
            en = self.g_energies_tot()
            T = self.model.get_tmat()
            Utot = utils.block_diag(self.ua,self.ub)
            T = numpy.einsum('ij,ip,jq->pq',T,Utot,Utot)
            fo = ft_utils.ff(beta, en, self.mu)
            fv = ft_utils.ffv(beta, en, self.mu)
            vec = fo*fv
            D = -beta*numpy.einsum('ii,i->i',T - numpy.diag(en),vec)
            D += -beta*numpy.einsum('ijij,i,j->i',V,vec,fo)
            return D
        else:
            print("WARNING: Derivative of MP1 energy is zero at OK")
            return 0.0

    def r_energies(self):
        raise Exception("Restricted energies are not definted")

    def u_energies(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        mu = self.mu
        if mu is None:
            eoa = self.ea[:self.na]
            eob = self.eb[:self.nb]
            eva = self.ea[self.na:]
            evb = self.eb[self.nb:]
            return (eoa,eva,eob,evb)
        else:
            ea = self.ea
            eb = self.eb
            return (ea[ea<mu],ea[ea>mu],eb[eb<mu],eb[eb>mu])

    def g_energies(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        mu = self.mu
        if mu is None:
            ea = self.ea
            eb = self.eb
            eoa = ea[0:self.na]
            eob = eb[0:self.nb]
            eva = ea[self.na:]
            evb = ea[self.nb:]
            return (numpy.hstack((eoa,eob)),numpy.hstack((eva,evb)))
        else:
            dtot = self.g_energies_tot()
            return (dtot[dtot<mu],dtot[dtot>mu])

    def r_energies_tot(self):
        raise Exception("Unrestricted reference")

    def u_energies_tot(self):
        ea = self.ea
        eb = self.eb
        return ea,eb

    def g_energies_tot(self):
        ea = self.ea
        eb = self.eb
        return numpy.hstack((ea,eb))

    def r_fock(self):
        raise Exception("Restricted mean-field is not implemented")

    def u_fock(self):
        oa,va,ob,vb = self._u_get_ov()
        oidxa = numpy.r_[oa]
        vidxa = numpy.r_[va]
        oidxb = numpy.r_[ob]
        vidxb = numpy.r_[vb]
        foa = numpy.zeros(self.ea.shape)
        fob = numpy.zeros(self.eb.shape)
        for i in range(self.na):
            foa[i] = 1.0
        for i in range(self.nb):
            fob[i] = 1.0
        Va,Vb,Vab = self.u_aint_tot()
        T = self.model.get_tmatS()
        Fa = numpy.einsum('ij,ip,jq->pq',T,self.ua,self.ua)
        Fb = numpy.einsum('ij,ip,jq->pq',T,self.ub,self.ub)
        Fa += numpy.einsum('pqrs,q,s->pr',Va,foa,foa)
        Fa += numpy.einsum('pqrs,q,s->pr',Vab,fob,fob)
        Fb += numpy.einsum('pqrs,q,s->pr',Vb,fob,fob)
        Fb += numpy.einsum('pqrs,p,r->qs',Vab,foa,foa)
        Fooa = Fa[numpy.ix_(oidxa,oidxa)]
        Fova = Fa[numpy.ix_(oidxa,vidxa)]
        Fvoa = Fa[numpy.ix_(vidxa,oidxa)]
        Fvva = Fa[numpy.ix_(vidxa,vidxa)]
        Foob = Fb[numpy.ix_(oidxb,oidxb)]
        Fovb = Fb[numpy.ix_(oidxb,vidxb)]
        Fvob = Fb[numpy.ix_(vidxb,oidxb)]
        Fvvb = Fb[numpy.ix_(vidxb,vidxb)]
        Fa_blocks = one_e_blocks(Fooa,Fova,Fvoa,Fvva)
        Fb_blocks = one_e_blocks(Foob,Fovb,Fvob,Fvvb)
        return Fa_blocks,Fb_blocks

    def g_fock(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        o,v = self._get_ov()
        T = self.model.get_tmatS()
        Tg = utils.block_diag(T,T)
        V = self.g_aint_tot()
        n = self.ea.shape[0] + self.eb.shape[0]
        do = numpy.zeros((n))
        for io in o:
            do[io] = 1.0
        utot = utils.block_diag(self.ua,self.ub)
        F = numpy.einsum('ij,ip,jq->pq',Tg,utot,utot)
        F += numpy.einsum('pqrs,q,s->pr',V,do,do)
        oidx = numpy.r_[o]
        vidx = numpy.r_[v]
        Foo = F[numpy.ix_(oidx,oidx)]
        Fvv = F[numpy.ix_(vidx,vidx)]
        Fov = F[numpy.ix_(oidx,vidx)]
        Fvo = F[numpy.ix_(vidx,oidx)]
        return one_e_blocks(Foo,Fov,Fvo,Fvv)

    def r_fock_tot(self):
        raise Exception("Restricted Fock operator not defined")
        return self.model.get_tmatS()
            
    def g_fock_tot(self):
        T = self.model.get_tmat()
        d = self.g_energies_tot()
        n = d.shape[0]
        if self.T > 0.0:
            fo = ft_utils.ff(self.beta, d, self.mu)
            I = numpy.identity(n)
            den = numpy.einsum('pi,i,qi->pq',I,fo,I)
        else:
            to = numpy.zeros((n,self.N))
            o,v = self._get_ov()
            for i,io in enumerate(o):
                to[i,io] = 1.0
            den = numpy.einsum('pi,qi->pq',to,to)
        V = self.g_aint_tot()
        JK = numpy.einsum('prqs,rs->pq',V,den)
        Utot = utils.block_diag(self.ua,self.ub)
        return JK + numpy.einsum('ij,ip,jq->pq',T,Utot,Utot)

    def u_fock_tot(self):
        Ta = self.model.get_tmatS()
        Tb = Ta
        da,db = self.u_energies_tot()
        na = da.shape[0]
        nb = db.shape[0]
        if self.T > 0.0:
            foa = ft_utils.ff(self.beta, da, self.mu)
            fob = ft_utils.ff(self.beta, db, self.mu)
            Ia = numpy.identity(na)
            Ib = numpy.identity(nb)
            dena = numpy.einsum('pi,i,qi->pq',Ia,foa,Ia)
            denb = numpy.einsum('pi,i,qi->pq',Ib,fob,Ib)
        else:
            N = self.ea.shape[0]
            toa = numpy.zeros((na,N))
            tob = numpy.zeros((nb,N))
            oa,va,ob,vb = self._u_get_ov()
            for i,io in enumerate(oa):
                toa[i,io] = 1.0
            for i,io in enumerate(ob):
                tob[i,io] = 1.0
            dena = numpy.einsum('pi,qi->pq',toa,toa)
            denb = numpy.einsum('pi,qi->pq',tob,tob)
        Va,Vb,Vabab = self.u_aint_tot()
        JKa = numpy.einsum('prqs,rs->pq',Va,dena)
        JKa += numpy.einsum('prqs,rs->pq',Vabab,denb)
        JKb = numpy.einsum('prqs,rs->pq',Vb,denb)
        JKb += numpy.einsum('prqs,pq->rs',Vabab,dena)
        Fa = JKa.copy()
        Fb = JKb.copy()
        Fa += numpy.einsum('ij,ip,jq->pq',Ta,self.ua,self.ua)
        Fb += numpy.einsum('ij,ip,jq->pq',Tb,self.ub,self.ub)
        return Fa,Fb

    def u_fock_d_tot(self,dveca,dvecb):
        Ta = self.model.get_tmatS()
        Tb = Ta
        da,db = self.u_energies_tot()
        na = da.shape[0]
        nb = db.shape[0]
        if self.T == 0.0:
            print("WARNING: Occupation derivatives are zero at 0K")
            return numpy.zeros((na,na)),numpy.zeros((nb,nb))
        foa = ft_utils.ff(self.beta, da, self.mu)
        fob = ft_utils.ff(self.beta, db, self.mu)
        fva = ft_utils.ffv(self.beta, da, self.mu)
        fvb = ft_utils.ffv(self.beta, db, self.mu)
        veca = dveca*foa*fva
        vecb = dvecb*fob*fvb
        Ia = numpy.identity(na)
        Ib = numpy.identity(nb)
        dena = numpy.einsum('pi,i,qi->pq',Ia,veca,Ia)
        denb = numpy.einsum('pi,i,qi->pq',Ib,vecb,Ib)
        Va,Vb,Vabab = self.u_aint_tot()
        JKa = numpy.einsum('prqs,rs->pq',Va,dena)
        JKa += numpy.einsum('prqs,rs->pq',Vabab,denb)
        JKb = numpy.einsum('prqs,rs->pq',Vb,denb)
        JKb += numpy.einsum('prqs,pq->rs',Vabab,dena)
        Fa = -JKa
        Fb = -JKb
        return Fa,Fb

    def g_fock_d_tot(self,dvec):
        d = self.g_energies_tot()
        n = d.shape[0]
        if self.T == 0.0:
            print("WARNING: Occupations derivatives are zero at 0K")
            return numpy.zeros((n,n))
        fo = ft_utils.ff(self.beta, d, self.mu)
        fv = ft_utils.ffv(self.beta, d, self.mu)
        vec = dvec*fo*fv
        I = numpy.identity(n)
        den = einsum('pi,i,qi->pq',I,vec,I)
        V = self.g_aint_tot()
        JK = einsum('prqs,rs->pq',V,den)
        return -JK

    def u_fock_d_den(self):
        da,db = self.u_energies_tot()
        na = da.shape[0]
        nb = db.shape[0]
        if self.T == 0.0:
            print("WARNING: Occupation derivatives are zero at 0K")
            return (numpy.zeros((na,na,na)),
                    numpy.zeros((na,na,nb)),
                    numpy.zeros((nb,nb,na)),
                    numpy.zeros((nb,nb,nb)))
        foa = ft_utils.ff(self.beta, da, self.mu)
        fob = ft_utils.ff(self.beta, db, self.mu)
        fva = ft_utils.ffv(self.beta, da, self.mu)
        fvb = ft_utils.ffv(self.beta, db, self.mu)
        veca = foa*fva
        vecb = fob*fvb
        Va,Vb,Vabab = self.u_aint_tot()
        JKaa = numpy.einsum('piqi,i->pqi',Va,veca)
        JKab = numpy.einsum('piqi,i->pqi',Vabab,vecb)
        JKbb = numpy.einsum('piqi,i->pqi',Vb,vecb)
        JKba = numpy.einsum('iris,i->rsi',Vabab,veca)
        return JKaa,JKab,JKbb,JKba

    def g_fock_d_den(self):
        d = self.g_energies_tot()
        n = d.shape[0]
        if self.T == 0.0:
            print("WARNING: Occupations derivatives are zero at 0K")
            return numpy.zeros((n,n))
        fo = ft_utils.ff(self.beta, d, self.mu)
        fv = ft_utils.ffv(self.beta, d, self.mu)
        vec = fo*fv
        V = self.g_aint_tot()
        JK = einsum('piqi,i->pqi',V,vec)
        return JK

    def r_hcore(self):
        return self.model.get_tmatS()

    def g_hcore(self):
        return utils.block_diag(self.model.get_tmat(),self.model.get_tmat())

    def u_aint(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        oa,va,ob,vb = self._u_get_ov()
        oidxa = numpy.r_[oa]
        vidxa = numpy.r_[va]
        oidxb = numpy.r_[ob]
        vidxb = numpy.r_[vb]
        na = self.na
        nb = self.nb
        Va,Vb,Vabab = self.u_aint_tot()

        Vvvvv = Va[numpy.ix_(vidxa,vidxa,vidxa,vidxa)]
        Vvvvo = Va[numpy.ix_(vidxa,vidxa,vidxa,oidxa)]
        Vvovv = Va[numpy.ix_(vidxa,oidxa,vidxa,vidxa)]
        Vvvoo = Va[numpy.ix_(vidxa,vidxa,oidxa,oidxa)]
        Vvovo = Va[numpy.ix_(vidxa,oidxa,vidxa,oidxa)]
        Voovv = Va[numpy.ix_(oidxa,oidxa,vidxa,vidxa)]
        Vvooo = Va[numpy.ix_(vidxa,oidxa,oidxa,oidxa)]
        Vooov = Va[numpy.ix_(oidxa,oidxa,oidxa,vidxa)]
        Voooo = Va[numpy.ix_(oidxa,oidxa,oidxa,oidxa)]
        Va = two_e_blocks(
            vvvv=Vvvvv,vvvo=Vvvvo,
            vovv=Vvovv,vvoo=Vvvoo,
            vovo=Vvovo,oovv=Voovv,
            vooo=Vvooo,ooov=Vooov,
            oooo=Voooo)
        Vvvvv = Vb[numpy.ix_(vidxb,vidxb,vidxb,vidxb)]
        Vvvvo = Vb[numpy.ix_(vidxb,vidxb,vidxb,oidxb)]
        Vvovv = Vb[numpy.ix_(vidxb,oidxb,vidxb,vidxb)]
        Vvvoo = Vb[numpy.ix_(vidxb,vidxb,oidxb,oidxb)]
        Vvovo = Vb[numpy.ix_(vidxb,oidxb,vidxb,oidxb)]
        Voovv = Vb[numpy.ix_(oidxb,oidxb,vidxb,vidxb)]
        Vvooo = Vb[numpy.ix_(vidxb,oidxb,oidxb,oidxb)]
        Vooov = Vb[numpy.ix_(oidxb,oidxb,oidxb,vidxb)]
        Voooo = Vb[numpy.ix_(oidxb,oidxb,oidxb,oidxb)]
        Vb = two_e_blocks(
            vvvv=Vvvvv,vvvo=Vvvvo,
            vovv=Vvovv,vvoo=Vvvoo,
            vovo=Vvovo,oovv=Voovv,
            vooo=Vvooo,ooov=Vooov,
            oooo=Voooo)

        Vvvvv = Vabab[numpy.ix_(vidxa,vidxb,vidxa,vidxb)]
        Vvvvo = Vabab[numpy.ix_(vidxa,vidxb,vidxa,oidxb)]
        Vvvov = Vabab[numpy.ix_(vidxa,vidxb,oidxa,vidxb)]
        Vvovv = Vabab[numpy.ix_(vidxa,oidxb,vidxa,vidxb)]
        Vovvv = Vabab[numpy.ix_(oidxa,vidxb,vidxa,vidxb)]
        Vvvoo = Vabab[numpy.ix_(vidxa,vidxb,oidxa,oidxb)]
        Vvoov = Vabab[numpy.ix_(vidxa,oidxb,oidxa,vidxb)]
        Vvovo = Vabab[numpy.ix_(vidxa,oidxb,vidxa,oidxb)]
        Vovvo = Vabab[numpy.ix_(oidxa,vidxb,vidxa,oidxb)]
        Vovov = Vabab[numpy.ix_(oidxa,vidxb,oidxa,vidxb)]
        Voovv = Vabab[numpy.ix_(oidxa,oidxb,vidxa,vidxb)]
        Vvooo = Vabab[numpy.ix_(vidxa,oidxb,oidxa,oidxb)]
        Vovoo = Vabab[numpy.ix_(oidxa,vidxb,oidxa,oidxb)]
        Voovo = Vabab[numpy.ix_(oidxa,oidxb,vidxa,oidxb)]
        Vooov = Vabab[numpy.ix_(oidxa,oidxb,oidxa,vidxb)]
        Voooo = Vabab[numpy.ix_(oidxa,oidxb,oidxa,oidxb)]
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
        Utot = utils.block_diag(self.ua,self.ub)
        Umat = self.g_aint_tot()
        o,v = self._get_ov()
        Vvvvv = None
        Vvvvo = None
        Vvovv = None
        Vvvoo = None
        Vvovo = None
        Voovv = None
        Vvooo = None
        Vooov = None
        Voooo = None
        oidx = numpy.r_[o]
        vidx = numpy.r_[v]
        if code == 0 or code == 1:
            Vvvvv = Umat[numpy.ix_(vidx,vidx,vidx,vidx)]
        if code == 0 or code == 2:
            Vvvvo = Umat[numpy.ix_(vidx,vidx,vidx,oidx)]
        if code == 0 or code == 3:
            Vvovv = Umat[numpy.ix_(vidx,oidx,vidx,vidx)]
        if code == 0 or code == 4:
            Vvvoo = Umat[numpy.ix_(vidx,vidx,oidx,oidx)]
        if code == 0 or code == 5:
            Vvovo = Umat[numpy.ix_(vidx,oidx,vidx,oidx)]
        if code == 0 or code == 6:
            Voovv = Umat[numpy.ix_(oidx,oidx,vidx,vidx)]
        if code == 0 or code == 7:
            Vvooo = Umat[numpy.ix_(vidx,oidx,oidx,oidx)]
        if code == 0 or code == 8:
            Vooov = Umat[numpy.ix_(oidx,oidx,oidx,vidx)]
        if code == 0 or code == 9:
            Voooo = Umat[numpy.ix_(oidx,oidx,oidx,oidx)]
        return two_e_blocks(
            vvvv=Vvvvv,vvvo=Vvvvo,
            vovv=Vvovv,vvoo=Vvvoo,
            vovo=Vvovo,oovv=Voovv,
            vooo=Vvooo,ooov=Vooov,
            oooo=Voooo)

    def u_aint_tot(self):
        V = self.model.get_umatS()
        Va = V - V.transpose((0,1,3,2))
        Vabab = self._transform2(V,self.ua,self.ub,self.ua,self.ub)
        Vb = self._transform1(Va,self.ub)
        Va = self._transform1(Va,self.ua)
        return Va,Vb,Vabab

    def g_aint_tot(self):
        Us = self.model.get_umatS()
        n,n1,n2,n3 = Us.shape
        assert(n == n1)
        assert(n == n2)
        assert(n == n3)

        U = numpy.zeros((2*n,2*n,2*n,2*n))
        U[n:,:n,n:,:n] = Us
        U[:n,n:,n:,:n] = -Us
        U[:n,n:,:n,n:] = Us
        U[n:,:n,:n,n:] = -Us
        utot = utils.block_diag(self.ua,self.ub)
        U = self._transform1(U,utot)
        return U
        
    def r_int_tot(self):
        raise Exception("Restricted MOs not implemented")
        return None

    def g_int_tot(self):
        U = self.model.get_umat()
        utot = utils.block_diag(self.ua,self.ub)
        U = self._transform1(U,utot)
        return U

    def _get_ov(self):
        """Get occupied and virtual indices in the general orbital space"""
        if self.mu is None:
            e = self.g_energies_tot()
            es = numpy.argsort(e)
            N = self.na + self.nb
            occ = es[:N]
            vir = es[N:]
            return (occ,vir)
        else:
            d = self.g_energies_tot()
            occ = []
            vir = []
            mu = self.mu
            for i in range(d.shape[0]):
                if d[i] < mu:
                    occ.append(i)
                else:
                    vir.append(i)
            return (occ,vir)

    def _u_get_ov(self):
        """Get occupied and virtual indices in the general orbital space"""
        if self.mu is None:
            ea,eb = self.u_energies_tot()
            esa = numpy.argsort(ea)
            esb = numpy.argsort(eb)
            na = self.na
            nb = self.nb
            occa = esa[:na]
            occb = esb[:nb]
            vira = esa[na:]
            virb = esb[nb:]
            return occa,vira,occb,virb
        else:
            da,db = self.u_energies_tot()
            occa = []
            vira = []
            mu = self.mu
            for i in range(da.shape[0]):
                if da[i] < mu:
                    occa.append(i)
                else:
                    vira.append(i)
            for i in range(db.shape[0]):
                if db[i] < mu:
                    occb.append(i)
                else:
                    virb.append(i)
            return occa,vira,occb,virb

    def _transform1(self, V, u):
        return self._transform2(V, u, u, u, u)

    def _transform2(self, V, u1, u2, u3, u4):
        Umat2 = numpy.einsum('ijkl,ls->ijks',V,u4)
        Umat1 = numpy.einsum('ijks,kr->ijrs',Umat2,u3)
        Umat2 = numpy.einsum('ijrs,jq->iqrs',Umat1,u2)
        Umat1 = numpy.einsum('iqrs,ip->pqrs',Umat2,u1)
        return Umat1
