import numpy
import functools
from pyscf import gto, scf
from cqcpy import ft_utils
from cqcpy import integrals
from cqcpy.ov_blocks import one_e_blocks
from cqcpy.ov_blocks import two_e_blocks
from cqcpy.ov_blocks import two_e_blocks_full
from cqcpy.ov_blocks import make_two_e_blocks
from cqcpy.ov_blocks import make_two_e_blocks_full
from cqcpy.integrals import eri_blocks
from cqcpy import utils
from . import scf_utils
from . import zt_mp
from . import ft_mp
from .system import system

class scf_system(system):
    """Object representing a molecular mean-field system.

    Attributes:
        mf (pyscf.scf): SCF object (not necessarily converged).
        T (float): Temperature.
        mu (float): Chemical potential.
    """
    def __init__(self,mf,T,mu,orbtype='u'):
        self.mf = mf
        self.T = T
        self.mu = mu
        self.orbtype = orbtype

    def verify(self,T,mu):
        if not (T == self.T and mu == self.mu):
            return False
        else:
            return True

    def has_g(self):
        return True

    def has_u(self):
        return (False if self.orbtype == 'g' else True)

    def const_energy(self):
        return self.mf.mol.energy_nuc()

    def get_mp1(self):
        hcore = self.mf.get_hcore(self.mf.mol)
        h = utils.block_diag(hcore,hcore)
        if self.T == 0:
            fao = scf_utils.get_ao_fock(self.mf)
            pao = scf_utils.get_ao_den(self.mf)
            return zt_mp.mp1(pao, fao, h)
        else:
            beta = 1.0 / self.T
            en = self.g_energies_tot()
            fo = ft_utils.ff(beta, en, self.mu)
            p = scf_utils.get_ao_ft_den(self.mf, fo)
            f0 = scf_utils.get_ao_fock(self.mf)
            fao = scf_utils.get_ao_ft_fock(self.mf, fo)
            return ft_mp.mp1(p,2*f0 - fao,h)

    # TODO: Do this with Fock build
    def g_d_mp1(self,dvec):
        assert(self.T > 0.0)
        beta = 1.0 / self.T
        en = self.g_energies_tot()
        fo = ft_utils.ff(beta, en, self.mu)
        fv = ft_utils.ffv(beta, en, self.mu)
        fov = dvec*fo*fv
        hcore = self.mf.get_hcore(self.mf.mol)
        hmo = scf_utils.mo_tran_1e(self.mf, hcore)
        eri = self.g_aint_tot()

        d1 = -numpy.einsum('ii,i->',hmo - numpy.diag(en),fov)
        d2 = -numpy.einsum('ijij,i,j->',eri,fov,fo)
        return (d1 + d2)

    def g_mp1_den(self):
        assert(self.T > 0.0)
        beta = 1.0 / self.T
        en = self.g_energies_tot()
        fo = ft_utils.ff(beta, en, self.mu)
        fv = ft_utils.ffv(beta, en, self.mu)
        fov = fo*fv
        hcore = self.mf.get_hcore(self.mf.mol)
        hmo = scf_utils.mo_tran_1e(self.mf, hcore)
        eri = self.g_aint_tot()

        d1 = -numpy.einsum('ii,i->i',hmo - numpy.diag(en),fov)
        d2 = -numpy.einsum('ijij,i,j->i',eri,fov,fo)
        na = en.shape[0]//2
        return beta*(d1 + d2)

    # TODO: Do this with Fock build
    def u_d_mp1(self,dveca,dvecb):
        assert(self.T > 0.0)
        beta = 1.0 / self.T
        ea,eb = self.u_energies_tot()
        hcore = self.mf.get_hcore(self.mf.mol)
        ha,hb = scf_utils.u_mo_tran_1e(self.mf, hcore)
        foa = ft_utils.ff(beta, ea, self.mu)
        fva = ft_utils.ffv(beta, ea, self.mu)
        fob = ft_utils.ff(beta, eb, self.mu)
        fvb = ft_utils.ffv(beta, eb, self.mu)
        fova = dveca*foa*fva
        fovb = dvecb*fob*fvb
        Ia,Ib,Iabab = self.u_aint_tot()
        d1 = -numpy.einsum('ii,i->',ha - numpy.diag(ea),fova)
        d1 -= numpy.einsum('ii,i->',hb - numpy.diag(eb),fovb)
        d2 = -numpy.einsum('ijij,i,j->',Ia,fova,foa)
        d2 -= numpy.einsum('ijij,i,j->',Ib,fovb,fob)
        d2 -= numpy.einsum('ijij,i,j->',Iabab,fova,fob)
        d2 -= numpy.einsum('ijij,i,j->',Iabab,foa,fovb)
        return (d1 + d2)

    def u_mp1_den(self):
        assert(self.T > 0.0)
        beta = 1.0 / self.T
        ea,eb = self.u_energies_tot()
        hcore = self.mf.get_hcore(self.mf.mol)
        ha,hb = scf_utils.u_mo_tran_1e(self.mf, hcore)
        foa = ft_utils.ff(beta, ea, self.mu)
        fva = ft_utils.ffv(beta, ea, self.mu)
        fob = ft_utils.ff(beta, eb, self.mu)
        fvb = ft_utils.ffv(beta, eb, self.mu)
        fova = foa*fva
        fovb = fob*fvb
        Ia,Ib,Iabab = self.u_aint_tot()
        d1a = -numpy.einsum('ii,i->i',ha - numpy.diag(ea),fova)
        d1b = -numpy.einsum('ii,i->i',hb - numpy.diag(eb),fovb)
        d2a = -numpy.einsum('ijij,i,j->i',Ia,fova,foa)
        d2b = -numpy.einsum('ijij,i,j->i',Ib,fovb,fob)
        d2a -= numpy.einsum('ijij,i,j->i',Iabab,fova,fob)
        d2b -= numpy.einsum('ijij,i,j->j',Iabab,foa,fovb)
        return beta*(d1a + d2a),beta*(d1b + d2b)

    def r_energies(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        return scf_utils.get_r_orbital_energies(self.mf)

    def u_energies(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        return scf_utils.get_u_orbital_energies(self.mf)

    def g_energies(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        return scf_utils.get_orbital_energies(self.mf)

    def r_energies_tot(self):
        return scf_utils.get_r_orbital_energies_tot(self.mf)

    def u_energies_tot(self):
        return scf_utils.get_u_orbital_energies_tot(self.mf)

    def g_energies_tot(self):
        return scf_utils.get_orbital_energies_gen(self.mf)
    
    def r_fock(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        F = scf_utils.r_fock_blocks(self.mf)
        return one_e_blocks(F.oo,F.ov,F.vo,F.vv)

    def u_fock(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        Fa = scf_utils.r_fock_blocks(self.mf,orb='a')
        Fb = scf_utils.r_fock_blocks(self.mf,orb='b')
        return (one_e_blocks(Fa.oo,Fa.ov,Fa.vo,Fa.vv),one_e_blocks(Fb.oo,Fb.ov,Fb.vo,Fb.vv))

    def g_fock(self):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        F = scf_utils.g_fock_blocks(self.mf)
        return one_e_blocks(F.oo,F.ov,F.vo,F.vv)

    def r_fock_tot(self):
        beta = 1.0 / self.T
        en = self.r_energies_tot()
        fo = ft_utils.ff(beta, en, self.mu)
        return scf_utils.get_r_ft_fock(self.mf, fo)

    def u_fock_tot(self):
        beta = 1.0 / self.T if self.T > 0 else 1.0e20
        ea,eb = self.u_energies_tot()
        foa = ft_utils.ff(beta, ea, self.mu)
        fob = ft_utils.ff(beta, eb, self.mu)
        return scf_utils.get_u_ft_fock(self.mf, foa, fob)

    def g_fock_tot(self):
        beta = 1.0 / self.T if self.T > 0 else 1.0e20
        en = self.g_energies_tot()
        fo = ft_utils.ff(beta, en, self.mu)
        return scf_utils.get_mo_ft_fock(self.mf, fo)

    def u_fock_d_tot(self,dveca,dvecb):
        beta = 1.0 / self.T if self.T > 0 else 1.0e20
        ea,eb = self.u_energies_tot()
        foa = ft_utils.ff(beta, ea, self.mu)
        fva = ft_utils.ffv(beta, ea, self.mu)
        fob = ft_utils.ff(beta, eb, self.mu)
        fvb = ft_utils.ffv(beta, eb, self.mu)
        return scf_utils.u_mo_d_ft_fock(self.mf, foa, fva, fob, fvb, dveca, dvecb)

    def u_fock_d_den(self):
        beta = 1.0 / self.T if self.T > 0 else 1.0e20
        ea,eb = self.u_energies_tot()
        foa = ft_utils.ff(beta, ea, self.mu)
        fva = ft_utils.ffv(beta, ea, self.mu)
        fob = ft_utils.ff(beta, eb, self.mu)
        fvb = ft_utils.ffv(beta, eb, self.mu)
        veca = foa*fva
        vecb = fob*fvb
        Va,Vb,Vabab = self.u_aint_tot()
        JKaa = numpy.einsum('piqi,i->pqi',Va,veca)
        JKab = numpy.einsum('piqi,i->pqi',Vabab,vecb)
        JKbb = numpy.einsum('piqi,i->pqi',Vb,vecb)
        JKba = numpy.einsum('iris,i->rsi',Vabab,veca)
        return JKaa,JKab,JKbb,JKba

    def g_fock_d_tot(self,dvec):
        beta = 1.0 / self.T if self.T > 0 else 1.0e20
        en = self.g_energies_tot()
        fo = ft_utils.ff(beta, en, self.mu)
        fv = ft_utils.ffv(beta, en, self.mu)
        return scf_utils.get_mo_d_ft_fock(self.mf, fo, fv, dvec)

    def g_fock_d_den(self):
        beta = 1.0 / self.T if self.T > 0 else 1.0e20
        en = self.g_energies_tot()
        fo = ft_utils.ff(beta, en, self.mu)
        fv = ft_utils.ffv(beta, en, self.mu)
        I = self.g_aint_tot()
        return numpy.einsum('piqi,i->pqi',I,fo*fv)

    def r_hcore(self):
        hcore = self.mf.get_hcore()
        h1e = list(functools.reduce(numpy.dot, (self.mf.mo_coeff.T, hcore, self.mf.mo_coeff)))
        return h1e

    #def g_hcore(self):
    #    hcore = self.mf.get_hcore()
    #    return scf_utils.mo_tran_1e(self.mf,hcore)

    def u_aint(self):
        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        _Ia,_Ib,_Iabab = self.u_aint_tot()
        if len(mo_occ.shape) == 1:
            noa = mo_occ[mo_occ>0].size
            nva = mo_occ[mo_occ==0].size
            nob = noa
            nvb = nva
        elif len(mo_occ.shape) == 2:
            mo_occa = mo_occ[0]
            mo_occb = mo_occ[1]
            noa = mo_occa[mo_occa>0].size
            nva = mo_occa[mo_occa==0].size
            nob = mo_occb[mo_occb>0].size
            nvb = mo_occb[mo_occb==0].size
        Ia = make_two_e_blocks(_Ia,noa,nva,noa,nva,noa,nva,noa,nva)
        Ib = make_two_e_blocks(_Ib,nob,nvb,nob,nvb,nob,nvb,nob,nvb)
        Iabab = make_two_e_blocks_full(_Iabab,
                noa,nva,nob,nvb,noa,nva,nob,nvb)

        return Ia, Ib, Iabab

    def g_aint(self,code=0):
        if self.T > 0.0:
            raise Exception("Undefined ov blocks at FT")
        I = eri_blocks(self.mf, code=code)
        if code == 0:
            return two_e_blocks(
                vvvv=I.vvvv, vvvo=I.vvvo,
                vovv=I.vovv, vvoo=I.vvoo,
                vovo=I.vovo, oovv=I.oovv,
                vooo=I.vooo, ooov=I.ooov,
                oooo=I.oooo)
        elif code == 4:
            return two_e_blocks(vvoo=I.vvoo)
        elif code == 9:
            return two_e_blocks(oooo=I.oooo)
        else:
            raise Exception(
                "Unrecognized integral code in scf_system::g_aint")

    def g_aint_tot(self):
        return integrals.get_phys_antiu_all_gen(self.mf)

    def u_aint_tot(self):
        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        if len(mo_occ.shape) == 1:
            moa = self.mf.mo_coeff
            mob = moa
        elif len(mo_occ.shape) == 2:
            moa = self.mf.mo_coeff[0]
            mob = self.mf.mo_coeff[1]

        #mol = self.mf.mol
        mf = self.mf
        Ia = integrals.get_phys_gen(mf,moa,moa,moa,moa,anti=True)
        Ib = integrals.get_phys_gen(mf,mob,mob,mob,mob,anti=True)
        Iabab = integrals.get_phys_gen(mf,moa,mob,moa,mob,anti=False)

        return Ia,Ib,Iabab

    def g_int_tot(self):
        return integrals.get_physu_all_gen(self.mf,anti=False)
