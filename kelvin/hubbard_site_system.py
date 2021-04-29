import numpy
from cqcpy import ft_utils
from cqcpy.ov_blocks import one_e_blocks
from cqcpy.ov_blocks import two_e_blocks
from cqcpy.ov_blocks import two_e_blocks_full
from cqcpy import utils
from .system import system

einsum = numpy.einsum

class hubbard_site_system(system):
    """Hubbard model system in the site basis

    Attributes:
        T (float): Temperature.
        model: Object specifying details of the model.
        Pa: Mean-field alpha density
        Pa: Mean-field beta density
        mu (float): Chemical potential.
        Na (float): Number of alpha electrons.
        Nb (float): Number of beta electrons.
    """
    def __init__(self, T, model, Pa=None, Pb=None,
                 mu=None, na=None, nb=None):
        self.T = T
        self.model = model
        self.Pa = Pa
        self.Pb = Pb
        if na is None:
            assert(nb is None)
            assert(mu is not None)
            self.mu = mu
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
            self.Fa = self.r_hcore()
            self.Fb = self.Fa
            self.orbtype = 'r'
        else:
            # build fock matrices
            self.orbtype = 'u'
            V = self.model.get_umatS()
            Va = V - V.transpose((0,1,3,2))
            Fa = self.r_hcore()
            Fb = self.r_hcore()
            Fa += einsum('pqrs,qs->pq', Va, Pa)
            Fa += einsum('pqrs,qs->pq', V, Pb)
            Fb += einsum('pqrs,qs->pq', Va, Pb)
            Fb += einsum('pqrs,pr->qs', V, Pa)
            self.Fa = Fa
            self.Fb = Fb

    def has_g(self):
        return True

    def has_u(self):
        return True

    def has_r(self):
        if self.orbtype == 'r':
            return True
        else:
            return False

    def verify(self, T, mu):
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
        if self.Pa is None:
            return 0.0
        if self.T == 0:
            # orbital energies
            eao,eav,ebo,evb = self.u_energies()
            E0 = eao.sum() + ebo.sum()
            hcore = self.r_hcore()
            ptot = self.Pa + self.Pb
            EHF = 0.5*numpy.trace(hcore*ptot)
            EHF += 0.5*numpy.trace(self.Fa*self.Pa)
            EHF += 0.5*numpy.trace(self.Fb*self.Pb)
            return EHF - E0
        else:
            Va,Vb,Vabab = self.u_aint_tot()
            ea,eb = self.u_energies_tot()
            foa = ft_utils.ff(self.beta, ea, self.mu)
            fob = ft_utils.ff(self.beta, eb, self.mu)
            E1 = 0.5*einsum('ijij,i,j->', Va, foa, foa)
            E1 += 0.5*einsum('ijij,i,j->', Vb, fob, fob)
            E1 += einsum('ijij,i,j->', Vabab, foa, fob)
            return E1

    def r_energies(self):
        raise Exception("Restricted energies are not definted")

    def u_energies(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        mu = self.mu
        if mu is None:
            oa,va,ob,vb = self._u_get_ov()
            oidxa = numpy.r_[oa]
            vidxa = numpy.r_[va]
            oidxb = numpy.r_[ob]
            vidxb = numpy.r_[vb]
            ea = self.Fa.diagonal()
            eb = self.Fb.diagonal()
            eoa = ea[numpy.ix_(oidxa)]
            eob = eb[numpy.ix_(oidxb)]
            eva = ea[numpy.ix_(vidxa)]
            evb = eb[numpy.ix_(vidxb)]
            return (eoa,eva,eob,evb)
        else:
            ea,eb = self.u_energies_tot()
            return (ea[ea < mu],ea[ea > mu],eb[eb < mu],eb[eb > mu])

    def g_energies(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        mu = self.mu
        if mu is None:
            e = self.g_energies_tot()
            o,v = self._get_ov()
            oidx = numpy.r_[o]
            vidx = numpy.r_[v]
            eo = e[numpy.ix_(oidx)]
            ev = e[numpy.ix_(vidx)]
            return (eo,ev)
        else:
            dtot = self.g_energies_tot()
            return (dtot[dtot < mu],dtot[dtot > mu])

    def r_energies_tot(self):
        if self.orbtype == 'r':
            return self.Fa.diagonal()
        else:
            raise Exception("Unrestricted reference")

    def u_energies_tot(self):
        ea = self.Fa.diagonal()
        eb = self.Fb.diagonal()
        return ea,eb

    def g_energies_tot(self):
        ea = self.Fa.diagonal()
        eb = self.Fb.diagonal()
        return numpy.hstack((ea,eb))

    def r_fock(self):
        raise Exception("Restricted mean-field is not implemented")

    def u_fock(self):
        oa,va,ob,vb = self._u_get_ov()
        oidxa = numpy.r_[oa]
        vidxa = numpy.r_[va]
        oidxb = numpy.r_[ob]
        vidxb = numpy.r_[vb]
        Fooa = self.Fa[numpy.ix_(oidxa,oidxa)]
        Fova = self.Fa[numpy.ix_(oidxa,vidxa)]
        Fvoa = self.Fa[numpy.ix_(vidxa,oidxa)]
        Fvva = self.Fa[numpy.ix_(vidxa,vidxa)]
        Foob = self.Fb[numpy.ix_(oidxb,oidxb)]
        Fovb = self.Fb[numpy.ix_(oidxb,vidxb)]
        Fvob = self.Fb[numpy.ix_(vidxb,oidxb)]
        Fvvb = self.Fb[numpy.ix_(vidxb,vidxb)]
        Fa = one_e_blocks(Fooa,Fova,Fvoa,Fvva)
        Fb = one_e_blocks(Foob,Fovb,Fvob,Fvvb)
        return Fa,Fb

    def g_fock(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        F = utils.block_diag(self.Fa,self.Fb)
        o,v = self._get_ov()
        oidx = numpy.r_[o]
        vidx = numpy.r_[v]
        Foo = F[numpy.ix_(oidx,oidx)]
        Fvv = F[numpy.ix_(vidx,vidx)]
        Fov = F[numpy.ix_(oidx,vidx)]
        Fvo = F[numpy.ix_(vidx,oidx)]
        return one_e_blocks(Foo,Fov,Fvo,Fvv)

    def r_fock_tot(self):
        return self.model.get_tmatS()

    def g_fock_tot(self):
        T = self.model.tmat()
        d = self.g_energies_tot()
        n = d.shape[0]
        if self.T > 0.0:
            fo = ft_utils.ff(self.beta, d, self.mu)
            I = numpy.identity(n)
            den = einsum('pi,i,qi->pq', I, fo, I)
        else:
            to = numpy.zeros((n,self.N))
            o,v = self._get_ov()
            for i,io in enumerate(o):
                to[i,io] = 1.0
            den = einsum('pi,qi->pq', to, to)
        V = self.g_aint_tot()
        JK = einsum('prqs,rs->pq', V, den)
        return T + JK

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
            dena = einsum('pi,i,qi->pq', Ia, foa, Ia)
            denb = einsum('pi,i,qi->pq', Ib, fob, Ib)
        else:
            toa = numpy.zeros((na,self.N))
            tob = numpy.zeros((nb,self.N))
            oa,va,ob,vb = self._get_u_ov()
            for i,io in enumerate(oa):
                toa[i,io] = 1.0
            for i,io in enumerate(ob):
                tob[i,io] = 1.0
            dena = einsum('pi,qi->pq', toa, toa)
            denb = einsum('pi,qi->pq', tob, tob)
        Va,Vb,Vabab = self.u_aint_tot()
        JKa = einsum('prqs,rs->pq', Va, dena)
        JKa += einsum('prqs,rs->pq', Vabab, denb)
        JKb = einsum('prqs,rs->pq', Vb, denb)
        JKb += einsum('prqs,rs->pq', Vabab, dena)
        return (Ta + JKa),(Tb + JKb)

    #def u_fock_d_tot(self,dveca,dvecb):
    #def g_fock_d_tot(self,dvec):

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

    def g_aint(self, code=0):
        Umat = self.model.get_u_tot()
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
        return Va,Va,V

    def g_aint_tot(self):
        U = self.model.get_umat()
        return U - U.transpose((0,1,3,2))

    def r_int_tot(self):
        return self.model.get_umatS()

    def g_int_tot(self):
        return self.model.get_umat()

    def _get_ov(self):
        """Get occupied and virtual indices in the general orbital space"""
        if self.mu is None:
            ea,eb = self.u_energies_tot()
            N = ea.shape[0]
            esa = numpy.argsort(ea)
            esb = numpy.argsort(eb)
            na = self.na
            nb = self.nb
            occ = numpy.hstack((esa[:na],esb[:nb] + N))
            vir = numpy.hstack((esa[na:],esb[nb:] + N))
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
