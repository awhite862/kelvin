import logging
import numpy
from pyscf import lib
from cqcpy import ft_utils
from cqcpy.ov_blocks import one_e_blocks
from cqcpy.ov_blocks import two_e_blocks
from cqcpy.ov_blocks import two_e_blocks_full
from .system import System
from .ueg_utils import UEGBasis

einsum = lib.einsum
#einsum = einsum


class UEGSCFSystem(System):
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
    def __init__(self, T, L, Emax, mu=None, na=None, nb=None,
                 norb=None, orbtype='u', madelung=None, naref=None):
        self.T = T
        self.L = L
        self.basis = UEGBasis(L, Emax, norb=norb)

        # compute mu if N is specified
        if na is not None:
            self.Na = na
            self.Nb = nb
            assert(na > 0)
            assert(nb > 0)
            mua = self.basis.Es[self.Na - 1] + 0.00001
            mub = self.basis.Es[self.Nb - 1] + 0.00001
            assert(mua == mub)
            self.mu = mua
            assert(self.T == 0.0)
        else:
            assert(nb is None)
            assert(mu is not None)
            self.mu = mu

        # store orbital occupations as numpy ranges
        d0 = numpy.asarray(self.basis.Es)
        n = d0.shape[0]
        if naref is not None:
            occ = [i for i in range(naref)]
            vir = [i + naref for i in range(n - naref)]
            self.oidx = numpy.r_[occ]
            self.vidx = numpy.r_[vir]
        else:
            occ = []
            vir = []
            for p, d in enumerate(d0):
                if d < self.mu:
                    occ.append(p)
                if d > self.mu:
                    vir.append(p)
            self.oidx = numpy.r_[occ]
            self.vidx = numpy.r_[vir]
            for p, d in enumerate(d0):
                if d < self.mu:
                    occ.append(p + n)
                if d > self.mu:
                    vir.append(p + n)
            self.goidx = numpy.r_[occ]
            self.gvidx = numpy.r_[vir]

        # now get real occupations if necessary
        if na is None:
            assert(nb is None)
            en = self.g_energies_tot()
            beta = 1.0 / self.T if self.T > 0.0 else 1.0e20
            fo = ft_utils.ff(beta, en, self.mu)
            N = fo.sum()
            self.Na = N/2.0
            self.Nb = self.Na

        # save some parameters
        self.N = self.Na + self.Nb
        self.den = self.N/(L*L*L)
        self.rs = (3/(4.0*numpy.pi*self.den))**(1.0/3.0)
        pi2 = numpy.pi*numpy.pi
        self.Ef = 0.5*(3.0*pi2*self.den)**(2.0/3.0)
        self.Tf = self.T / self.Ef
        self.orbtype = orbtype
        self.madelung = madelung
        self._mconst = 2.837297479 / (2*self.L)

    def has_g(self):
        return True

    def has_u(self):
        return (False if self.orbtype == 'g' else True)

    def has_r(self):
        return (True if self.orbtype == 'r' else False)

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
        if self.madelung == 'const':
            return -(self.Na + self.Nb)*self._mconst
        else:
            return 0.0

    def get_mp1(self):
        if self.has_u():
            if self.T > 0:
                Va, Vb, Vabab = self.u_aint_tot()
                beta = 1.0 / self.T
                ea, eb = self.u_energies_tot()
                foa = ft_utils.ff(beta, ea, self.mu)
                fob = ft_utils.ff(beta, eb, self.mu)
                tmat = self.r_hcore()
                E1_1 = einsum('ii,i->', tmat - numpy.diag(ea), foa)
                E1_1 += einsum('ii,i->', tmat - numpy.diag(eb), fob)
                E1_2 = 0.5*einsum('ijij,i,j->', Va, foa, foa)
                E1_2 += 0.5*einsum('ijij,i,j->', Vb, fob, fob)
                E1_2 += einsum('ijij,i,j->', Vabab, foa, fob)
                return E1_2 + E1_1
            else:
                Va, Vb, Vabab = self.u_aint()
                E1 = -0.5*numpy.einsum('ijij->', Va.oooo)
                E1 -= 0.5*numpy.einsum('ijij->', Vb.oooo)
                E1 -= numpy.einsum('ijij->', Vabab.oooo)
                return E1
        else:
            if self.T > 0:
                V = self.g_aint_tot()
                beta = 1.0 / self.T
                en = self.g_energies_tot()
                tmat = self.g_hcore()
                fo = ft_utils.ff(beta, en, self.mu)
                E1_1 = einsum('ii,i->', tmat - numpy.diag(en), fo)
                E1_2 = 0.5*einsum('ijij,i,j->', V, fo, fo)
                return E1_2 + E1_1
            else:
                V = self.g_aint()
                return -0.5*einsum('ijij->', V.oooo)

    def u_d_mp1(self, dveca, dvecb):
        if self.T > 0:
            Va, Vb, Vabab = self.u_aint_tot()
            beta = 1.0 / self.T
            ea, eb = self.u_energies_tot()
            foa = ft_utils.ff(beta, ea, self.mu)
            fva = ft_utils.ffv(beta, ea, self.mu)
            veca = dveca*foa*fva
            fob = ft_utils.ff(beta, eb, self.mu)
            fvb = ft_utils.ffv(beta, eb, self.mu)
            vecb = dvecb*fob*fvb
            tmat = self.r_hcore()
            D = -einsum('ii,i->', tmat - numpy.diag(ea), veca)
            D += -einsum('ii,i->', tmat - numpy.diag(eb), vecb)
            D += -einsum('ijij,i,j->', Va, veca, foa)
            D += -einsum('ijij,i,j->', Vb, vecb, fob)
            D += -einsum('ijij,i,j->', Vabab, veca, fob)
            D += -einsum('ijij,i,j->', Vabab, foa, vecb)
            return D
        else:
            logging.warning("Derivative of MP1 energy is zero at OK")
            return 0.0

    def u_mp1_den(self):
        if self.T > 0:
            Va, Vb, Vabab = self.u_aint_tot()
            beta = 1.0 / self.T
            ea, eb = self.u_energies_tot()
            foa = ft_utils.ff(beta, ea, self.mu)
            fva = ft_utils.ffv(beta, ea, self.mu)
            veca = foa*fva
            fob = ft_utils.ff(beta, eb, self.mu)
            fvb = ft_utils.ffv(beta, eb, self.mu)
            vecb = fob*fvb
            tmat = self.r_hcore()
            Da = -beta*einsum('ii,i->i', tmat - numpy.diag(ea), veca)
            Db = -beta*einsum('ii,i->i', tmat - numpy.diag(eb), vecb)
            Da += -beta*einsum('ijij,i,j->i', Va, veca, foa)
            Db += -beta*einsum('ijij,i,j->i', Vb, vecb, fob)
            Da += -beta*einsum('ijij,i,j->i', Vabab, veca, fob)
            Db += -beta*einsum('ijij,i,j->j', Vabab, foa, vecb)
            return Da, Db
        else:
            logging.warning("Derivative of MP1 energy is zero at OK")
            return 0.0

    def g_d_mp1(self, dvec):
        if self.T > 0:
            V = self.g_aint_tot()
            beta = 1.0 / self.T
            en = self.g_energies_tot()
            fo = ft_utils.ff(beta, en, self.mu)
            fv = ft_utils.ffv(beta, en, self.mu)
            vec = dvec*fo*fv
            tmat = self.g_hcore()
            E1_2 = -einsum('ijij,i,j->', V, vec, fo)
            E1_1 = -einsum('ii,i->', tmat - numpy.diag(en), vec)
            return E1_2 + E1_1
        else:
            logging.warning("Derivative of MP1 energy is zero at OK")
            return 0.0

    def g_mp1_den(self):
        if self.T > 0:
            V = self.g_aint_tot()
            beta = 1.0 / self.T
            en = self.g_energies_tot()
            fo = ft_utils.ff(beta, en, self.mu)
            fv = ft_utils.ffv(beta, en, self.mu)
            vec = fo*fv
            tmat = self.g_hcore()
            E1_2 = -beta*einsum('ijij,i,j->i', V, vec, fo)
            E1_1 = -beta*einsum('ii,i->i', tmat - numpy.diag(en), vec)
            return E1_1 + E1_2
        else:
            logging.warning("Derivative of MP1 energy is zero at OK")
            return numpy.zeros((self.g_energies_tot().shape))

    def r_energies(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        if self.Na != self.Nb:
            raise Exception("UEG system is not restricted")
        F = self.r_fock()
        na = int(self.Na)
        assert(na == int(self.Nb))
        eo = F.oo.diagonal()
        ev = F.vv.diagonal()
        if self.madelung == "orb":
            eo -= self._mconst
        return (eo, ev)

    def u_energies(self):
        fa, fb = self.u_fock()
        eoa = fa.oo.diagonal()
        eva = fa.vv.diagonal()
        eob = fb.oo.diagonal()
        evb = fb.vv.diagonal()
        if self.madelung == "orb":
            eoa -= self._mconst
            eob -= self._mconst
        return (eoa, eva, eob, evb)

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
        eo = numpy.hstack((eoa, eob))
        ev = numpy.hstack((eva, evb))
        if self.madelung == "orb":
            eo -= self._mconst
        return (eo, ev)

    def r_energies_tot(self):
        e = numpy.asarray(self.basis.Es)
        n = e.shape[0]
        V = self.r_int_tot()
        Vd = V[numpy.ix_(numpy.arange(n), self.oidx, numpy.arange(n), self.oidx)]
        Vx = V[numpy.ix_(numpy.arange(n), self.oidx, self.oidx, numpy.arange(n))]
        e += 2*numpy.einsum('pipi->p', Vd) - numpy.einsum('piip->p', Vx)
        return e

    def u_energies_tot(self):
        e = self.r_energies_tot()
        return e, e.copy()

    def g_energies_tot(self):
        ea, eb = self.u_energies_tot()
        return numpy.hstack((ea, eb))

    def r_fock(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        F = self.r_hcore()
        V = self.r_int_tot()
        n = F.shape[0]
        Vd = V[numpy.ix_(numpy.arange(n), self.oidx, numpy.arange(n), self.oidx)]
        Vx = V[numpy.ix_(numpy.arange(n), self.oidx, self.oidx, numpy.arange(n))]
        F = F + 2*einsum('piri->pr', Vd) - einsum('piir->pr', Vx)
        Foo = F[numpy.ix_(self.oidx, self.oidx)]
        Fvv = F[numpy.ix_(self.vidx, self.vidx)]
        Fov = F[numpy.ix_(self.oidx, self.vidx)]
        Fvo = F[numpy.ix_(self.vidx, self.oidx)]
        return one_e_blocks(Foo, Fov, Fvo, Fvv)

    def u_fock(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        F = self.r_hcore()
        n = F.shape[0]
        oidx = self.oidx
        vidx = self.vidx
        V = self.r_int_tot()
        Vd = V[numpy.ix_(numpy.arange(n), oidx, numpy.arange(n), oidx)]
        Vx = V[numpy.ix_(numpy.arange(n), oidx, oidx, numpy.arange(n))]
        F = F + 2*einsum('piri->pr', Vd) - einsum('piir->pr', Vx)
        Foo = F[numpy.ix_(oidx, oidx)]
        Fvv = F[numpy.ix_(vidx, vidx)]
        Fov = F[numpy.ix_(oidx, vidx)]
        Fvo = F[numpy.ix_(vidx, oidx)]
        Fa = one_e_blocks(Foo, Fov, Fvo, Fvv)
        Fb = one_e_blocks(Foo, Fov, Fvo, Fvv)
        return Fa, Fb

    def g_fock(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        F = self.g_hcore()
        n = F.shape[0]
        goidx = self.goidx
        gvidx = self.gvidx
        V = self.g_aint_tot()
        V = V[numpy.ix_(numpy.arange(n), goidx, numpy.arange(n), goidx)]
        F = F + einsum('piri->pr', V)
        Foo = F[numpy.ix_(goidx, goidx)]
        Fvv = F[numpy.ix_(gvidx, gvidx)]
        Fov = F[numpy.ix_(goidx, gvidx)]
        Fvo = F[numpy.ix_(gvidx, goidx)]
        return one_e_blocks(Foo, Fov, Fvo, Fvv)

    def r_fock_tot(self):
        T = self.r_hcore()
        d = self.r_energies_tot()
        n = d.shape[0]
        if self.T > 0.0:
            beta = 1.0 / self.T
            fo = ft_utils.ff(beta, d, self.mu)
            I = numpy.identity(n)
            den = einsum('pi,i,qi->pq', I, fo, I)
        else:
            to = numpy.zeros((n, self.N))
            i = 0
            for p in range(n):
                if d[p] < self.mu:
                    to[p, i] = 1.0
                    i = i+1
            den = einsum('pi,qi->pq', to, to)
        V = self.r_int_tot()
        JK = 2*einsum('prqs,rs->pq', V, den) - einsum('prsq,rs->pq', V, den)
        return T + JK

    def u_fock_tot(self):
        Ta, Tb = self.basis.build_u_ke_matrix()
        da, db = self.u_energies_tot()
        na = da.shape[0]
        nb = db.shape[0]
        if self.T > 0.0:
            beta = 1.0 / self.T
            foa = ft_utils.ff(beta, da, self.mu)
            fob = ft_utils.ff(beta, db, self.mu)
            Ia = numpy.identity(na)
            Ib = numpy.identity(nb)
            dena = einsum('pi,i,qi->pq', Ia, foa, Ia)
            denb = einsum('pi,i,qi->pq', Ib, fob, Ib)
        else:
            dena = numpy.zeros((na, na))
            denb = numpy.zeros((nb, nb))
            for i in range(self.oidx):
                dena[i, i] = 1.0
                denb[i, i] = 1.0
        Va, Vb, Vabab = self.u_aint_tot()
        JKa = einsum('prqs,rs->pq', Va, dena)
        JKa += einsum('prqs,rs->pq', Vabab, denb)
        JKb = einsum('prqs,rs->pq', Vb, denb)
        JKb += einsum('prqs,rs->pq', Vabab, dena)
        return (Ta + JKa), (Tb + JKb)

    def g_fock_tot(self):
        T = self.basis.build_g_ke_matrix()
        d = self.g_energies_tot()
        n = d.shape[0]
        if self.T > 0.0:
            beta = 1.0 / self.T
            fo = ft_utils.ff(beta, d, self.mu)
            I = numpy.identity(n)
            den = einsum('pi,i,qi->pq', I, fo, I)
        else:
            to = numpy.zeros((n, self.N))
            i = 0
            for p in range(n):
                if d[p] < self.mu:
                    to[p, i] = 1.0
                    i = i+1
            den = einsum('pi,qi->pq', to, to)
        V = self.g_aint_tot()
        JK = einsum('prqs,rs->pq', V, den)
        return T + JK

    def u_fock_d_tot(self, dveca, dvecb):
        da, db = self.u_energies_tot()
        na = da.shape[0]
        nb = db.shape[0]
        if self.T == 0.0:
            logging.warning("Occupations derivatives are zero at 0K")
            return numpy.zeros((na, na)), numpy.zeros((nb, nb))
        beta = 1.0 / self.T
        foa = ft_utils.ff(beta, da, self.mu)
        fva = ft_utils.ffv(beta, da, self.mu)
        veca = dveca*foa*fva
        fob = ft_utils.ff(beta, db, self.mu)
        fvb = ft_utils.ffv(beta, db, self.mu)
        vecb = dvecb*fob*fvb
        Ia = numpy.identity(na)
        Ib = numpy.identity(nb)
        dena = einsum('pi,i,qi->pq', Ia, veca, Ia)
        denb = einsum('pi,i,qi->pq', Ib, vecb, Ib)
        Va, Vb, Vabab = self.u_aint_tot()
        JKa = einsum('prqs,rs->pq', Va, dena)
        JKa += einsum('prqs,rs->pq', Vabab, denb)
        JKb = einsum('prqs,rs->pq', Vb, denb)
        JKb += einsum('prqs,pq->rs', Vabab, dena)
        return -JKa, -JKb

    def u_fock_d_den(self):
        da, db = self.u_energies_tot()
        na = da.shape[0]
        nb = db.shape[0]
        if self.T == 0.0:
            logging.warning("Occupations derivatives are zero at 0K")
            return numpy.zeros((na, na)), numpy.zeros((nb, nb))
        beta = 1.0 / self.T
        foa = ft_utils.ff(beta, da, self.mu)
        fva = ft_utils.ffv(beta, da, self.mu)
        veca = foa*fva
        fob = ft_utils.ff(beta, db, self.mu)
        fvb = ft_utils.ffv(beta, db, self.mu)
        vecb = fob*fvb
        Va, Vb, Vabab = self.u_aint_tot()
        JKaa = einsum('piqi,i->pqi', Va, veca)
        JKab = einsum('piqi,i->pqi', Vabab, vecb)
        JKbb = einsum('piqi,i->pqi', Vb, vecb)
        JKba = einsum('iris,i->rsi', Vabab, veca)
        return JKaa, JKab, JKbb, JKba

    def g_fock_d_tot(self, dvec):
        d = self.g_energies_tot()
        n = d.shape[0]
        if self.T == 0.0:
            logging.warning("Occupations derivatives are zero at 0K")
            return numpy.zeros((n, n))
        beta = 1.0 / self.T
        fo = ft_utils.ff(beta, d, self.mu)
        fv = ft_utils.ffv(beta, d, self.mu)
        vec = dvec*fo*fv
        I = numpy.identity(n)
        den = einsum('pi,i,qi->pq', I, vec, I)
        V = self.g_aint_tot()
        JK = einsum('prqs,rs->pq', V, den)
        return -JK

    def g_fock_d_den(self):
        d = self.g_energies_tot()
        n = d.shape[0]
        if self.T == 0.0:
            logging.warning("Occupations derivatives are zero at 0K")
            return numpy.zeros((n, n))
        beta = 1.0 / self.T
        fo = ft_utils.ff(beta, d, self.mu)
        fv = ft_utils.ffv(beta, d, self.mu)
        vec = fo*fv
        V = self.g_aint_tot()
        JK = einsum('piqi,i->pqi', V, vec)
        return JK

    def r_hcore(self):
        return numpy.diag(numpy.asarray(self.basis.Es))

    def g_hcore(self):
        return self.basis.build_g_ke_matrix()

    def u_aint(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        Va, Vb, Vabab = self.u_aint_tot()
        oaidx = self.oidx
        vaidx = self.vidx
        obidx = self.oidx
        vbidx = self.vidx

        Vvvvv = Va[numpy.ix_(vaidx, vaidx, vaidx, vaidx)]
        Vvvvo = Va[numpy.ix_(vaidx, vaidx, vaidx, oaidx)]
        Vvovv = Va[numpy.ix_(vaidx, oaidx, vaidx, vaidx)]
        Vvvoo = Va[numpy.ix_(vaidx, vaidx, oaidx, oaidx)]
        Vvovo = Va[numpy.ix_(vaidx, oaidx, vaidx, oaidx)]
        Voovv = Va[numpy.ix_(oaidx, oaidx, vaidx, vaidx)]
        Vvooo = Va[numpy.ix_(vaidx, oaidx, oaidx, oaidx)]
        Vooov = Va[numpy.ix_(oaidx, oaidx, oaidx, vaidx)]
        Voooo = Va[numpy.ix_(oaidx, oaidx, oaidx, oaidx)]
        Va = two_e_blocks(
            vvvv=Vvvvv, vvvo=Vvvvo,
            vovv=Vvovv, vvoo=Vvvoo,
            vovo=Vvovo, oovv=Voovv,
            vooo=Vvooo, ooov=Vooov,
            oooo=Voooo)
        Vvvvv = Vb[numpy.ix_(vbidx, vbidx, vbidx, vbidx)]
        Vvvvo = Vb[numpy.ix_(vbidx, vbidx, vbidx, obidx)]
        Vvovv = Vb[numpy.ix_(vbidx, obidx, vbidx, vbidx)]
        Vvvoo = Vb[numpy.ix_(vbidx, vbidx, obidx, obidx)]
        Vvovo = Vb[numpy.ix_(vbidx, obidx, vbidx, obidx)]
        Voovv = Vb[numpy.ix_(obidx, obidx, vbidx, vbidx)]
        Vvooo = Vb[numpy.ix_(vbidx, obidx, obidx, obidx)]
        Vooov = Vb[numpy.ix_(obidx, obidx, obidx, vbidx)]
        Voooo = Vb[numpy.ix_(obidx, obidx, obidx, obidx)]
        Vb = two_e_blocks(
            vvvv=Vvvvv, vvvo=Vvvvo,
            vovv=Vvovv, vvoo=Vvvoo,
            vovo=Vvovo, oovv=Voovv,
            vooo=Vvooo, ooov=Vooov,
            oooo=Voooo)

        Vvvvv = Vabab[numpy.ix_(vaidx, vbidx, vaidx, vbidx)]
        Vvvvo = Vabab[numpy.ix_(vaidx, vbidx, vaidx, obidx)]
        Vvvov = Vabab[numpy.ix_(vaidx, vbidx, oaidx, vbidx)]
        Vvovv = Vabab[numpy.ix_(vaidx, obidx, vaidx, vbidx)]
        Vovvv = Vabab[numpy.ix_(oaidx, vbidx, vaidx, vbidx)]
        Vvvoo = Vabab[numpy.ix_(vaidx, vbidx, oaidx, obidx)]
        Vvoov = Vabab[numpy.ix_(vaidx, obidx, oaidx, vbidx)]
        Vvovo = Vabab[numpy.ix_(vaidx, obidx, vaidx, obidx)]
        Vovvo = Vabab[numpy.ix_(oaidx, vbidx, vaidx, obidx)]
        Vovov = Vabab[numpy.ix_(oaidx, vbidx, oaidx, vbidx)]
        Voovv = Vabab[numpy.ix_(oaidx, obidx, vaidx, vbidx)]
        Vvooo = Vabab[numpy.ix_(vaidx, obidx, oaidx, obidx)]
        Vovoo = Vabab[numpy.ix_(oaidx, vbidx, oaidx, obidx)]
        Voovo = Vabab[numpy.ix_(oaidx, obidx, vaidx, obidx)]
        Vooov = Vabab[numpy.ix_(oaidx, obidx, oaidx, vbidx)]
        Voooo = Vabab[numpy.ix_(oaidx, obidx, oaidx, obidx)]
        Vabab = two_e_blocks_full(
            vvvv=Vvvvv, vvvo=Vvvvo,
            vvov=Vvvov, vovv=Vvovv,
            ovvv=Vovvv, vvoo=Vvvoo,
            vovo=Vvovo, ovvo=Vovvo,
            voov=Vvoov, ovov=Vovov,
            oovv=Voovv, vooo=Vvooo,
            ovoo=Vovoo, oovo=Voovo,
            ooov=Vooov, oooo=Voooo)
        return Va, Vb, Vabab

    def g_aint(self, code=0):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
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
        goidx = self.goidx
        gvidx = self.gvidx
        if code == 0 or code == 1:
            Vvvvv = V[numpy.ix_(gvidx, gvidx, gvidx, gvidx)]
        if code == 0 or code == 2:
            Vvvvo = V[numpy.ix_(gvidx, gvidx, gvidx, goidx)]
        if code == 0 or code == 3:
            Vvovv = V[numpy.ix_(gvidx, goidx, gvidx, gvidx)]
        if code == 0 or code == 4:
            Vvvoo = V[numpy.ix_(gvidx, gvidx, goidx, goidx)]
        if code == 0 or code == 5:
            Vvovo = V[numpy.ix_(gvidx, goidx, gvidx, goidx)]
        if code == 0 or code == 6:
            Voovv = V[numpy.ix_(goidx, goidx, gvidx, gvidx)]
        if code == 0 or code == 7:
            Vvooo = V[numpy.ix_(gvidx, goidx, goidx, goidx)]
        if code == 0 or code == 8:
            Vooov = V[numpy.ix_(goidx, goidx, goidx, gvidx)]
        if code == 0 or code == 9:
            Voooo = V[numpy.ix_(goidx, goidx, goidx, goidx)]
        return two_e_blocks(
            vvvv=Vvvvv, vvvo=Vvvvo,
            vovv=Vvovv, vvoo=Vvvoo,
            vovo=Vvovo, oovv=Voovv,
            vooo=Vvooo, ooov=Vooov,
            oooo=Voooo)

    def u_aint_tot(self):
        return self.basis.build_u2e_matrix()

    def g_aint_tot(self):
        return self.basis.build_g2e_matrix()

    def r_int_tot(self):
        return self.basis.build_r2e_matrix()

    def g_int_tot(self):
        return self.basis.build_g2e_matrix(anti=False)


class ueg_scf_system(UEGSCFSystem):
    def __init__(self, T, L, Emax, mu=None, na=None, nb=None,
                 norb=None, orbtype='u', madelung=None, naref=None):
        logging.warning("This class is deprecated, use UEGSCFSystem instead")
        UEGSCFSystem.__init__(
            self, T, L, Emax, mu=mu, na=na, nb=nb, norb=norb,
            orbtype=orbtype, madelung=madelung, naref=naref)
