import numpy
from pyscf import gto, scf
from cqcpy import ft_utils
from cqcpy import integrals
from cqcpy.ov_blocks import one_e_blocks
from cqcpy.ov_blocks import two_e_blocks
from cqcpy.ov_blocks import two_e_blocks_full
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
        if self.orbtype == 'g':
            return False
        else:
            return True

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
            beta = 1.0 / (self.T + 1e-12)
            en = self.g_energies_tot()
            fo = ft_utils.ff(beta, en, self.mu)
            p = scf_utils.get_ao_ft_den(self.mf, fo)
            f0 = scf_utils.get_ao_fock(self.mf)
            fao = scf_utils.get_ao_ft_fock(self.mf, fo)
            return ft_mp.mp1(p,2*f0 - fao,h)

    # TODO: Do this with Fock build
    def g_d_mp1(self,dvec):
        assert(self.T > 0.0)
        beta = 1.0 / (self.T + 1e-12)
        en = self.g_energies_tot()
        fo = ft_utils.ff(beta, en, self.mu)
        fv = ft_utils.ffv(beta, en, self.mu)
        fov = dvec*fo*fv
        hcore = self.mf.get_hcore(self.mf.mol)
        hmo = scf_utils.mo_tran_1e(self.mf, hcore)
        eri = self.g_aint_tot()

        d1 = numpy.einsum('ii,i->',hmo - numpy.diag(en),fov)
        d2 = numpy.einsum('ijij,i,j->',eri,fov,fo)
        return (d1 + d2)

    # TODO: Do this with Fock build
    def u_d_mp1(self,dveca,dvecb):
        assert(self.T > 0.0)
        beta = 1.0 / (self.T + 1e-12)
        ea,eb = self.u_energies_tot()
        hcore = self.mf.get_hcore(self.mf.mol)
        ha,hb = scf_utils.u_mo_tran_1e(self.mf, hcore)
        foa = ft_utils.ff(beta, ea, self.mu)
        fva = ft_utils.ffv(beta, eb, self.mu)
        fob = ft_utils.ff(beta, ea, self.mu)
        fvb = ft_utils.ffv(beta, eb, self.mu)
        fova = dveca*foa*fva
        fovb = dvecb*fob*fvb
        Ia,Ib,Iabab = self.u_aint_tot()
        d1 = numpy.einsum('ii,i->',ha - numpy.diag(ea),fova)
        d1 += numpy.einsum('ii,i->',hb - numpy.diag(eb),fovb)
        d2 = numpy.einsum('ijij,i,j->',Ia,fova,foa)
        d2 += numpy.einsum('ijij,i,j->',Ib,fovb,fob)
        d2 += numpy.einsum('ijij,i,j->',Iabab,fova,fob)
        d2 += numpy.einsum('ijij,i,j->',Iabab,foa,fovb)
        return (d1 + d2)

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
        beta = 1.0 / (self.T + 1e-12)
        en = self.r_energies_tot()
        fo = ft_utils.ff(beta, en, self.mu)
        return scf_utils.get_r_ft_fock(self.mf, fo)

    def u_fock_tot(self):
        beta = 1.0 / (self.T + 1e-12)
        ea,eb = self.u_energies_tot()
        foa = ft_utils.ff(beta, ea, self.mu)
        fob = ft_utils.ff(beta, eb, self.mu)
        return scf_utils.get_u_ft_fock(self.mf, foa, fob)

    def g_fock_tot(self):
        beta = 1.0 / (self.T + 1e-12)
        en = self.g_energies_tot()
        fo = ft_utils.ff(beta, en, self.mu)
        return scf_utils.get_mo_ft_fock(self.mf, fo)

    def u_fock_d_tot(self,dveca,dvecb):
        beta = 1.0 / (self.T + 1e-12)
        ea,eb = self.u_energies_tot()
        foa = ft_utils.ff(beta, ea, self.mu)
        fva = ft_utils.ffv(beta, ea, self.mu)
        fob = ft_utils.ff(beta, eb, self.mu)
        fvb = ft_utils.ffv(beta, eb, self.mu)
        return scf_utils.u_mo_d_ft_fock(self.mf, foa, fva, fob, fvb, dveca, dvecb)

    def g_fock_d_tot(self,dvec):
        beta = 1.0 / (self.T + 1e-12)
        en = self.g_energies_tot()
        fo = ft_utils.ff(beta, en, self.mu)
        fv = ft_utils.ffv(beta, en, self.mu)
        return scf_utils.get_mo_d_ft_fock(self.mf, fo, fv, dvec)

    def r_hcore(self):
        hcore = self.mf.get_hcore()
        h1e = reduce(numpy.dot, (self.mf.mo_coeff.T, hcore, self.mf.mo_coeff))
        return h1e

    #def g_hcore(self):
    #    hcore = self.mf.get_hcore()
    #    return scf_utils.mo_tran_1e(self.mf,hcore)

    def u_aint(self):
        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        if len(mo_occ.shape) == 1:
            oa = self.mf.mo_coeff[:,mo_occ>0]
            va = self.mf.mo_coeff[:,mo_occ==0]
            ob = oa
            vb = va
        elif len(mo_occ.shape) == 2:
            mo_occa = mo_occ[0]
            mo_occb = mo_occ[1]
            oa = (self.mf.mo_coeff[0])[:,mo_occa>0]
            va = (self.mf.mo_coeff[0])[:,mo_occa==0]
            ob = (self.mf.mo_coeff[1])[:,mo_occb>0]
            vb = (self.mf.mo_coeff[1])[:,mo_occb==0]

        mol = self.mf.mol

        # build alpha integrals
        vvvv = integrals.get_phys_anti(mol,va,va,va,va)
        vvvo = integrals.get_phys_anti(mol,va,va,va,oa)
        vovv = integrals.get_phys_anti(mol,va,oa,va,va)
        vvoo = integrals.get_phys_anti(mol,va,va,oa,oa)
        vovo = integrals.get_phys_anti(mol,va,oa,va,oa)
        oovv = integrals.get_phys_anti(mol,oa,oa,va,va)
        vooo = integrals.get_phys_anti(mol,va,oa,oa,oa)
        ooov = integrals.get_phys_anti(mol,oa,oa,oa,va)
        oooo = integrals.get_phys_anti(mol,oa,oa,oa,oa)
        Ia = two_e_blocks(
                vvvv=vvvv, vvvo=vvvo,
                vovv=vovv, vvoo=vvoo,
                vovo=vovo, oovv=oovv,
                vooo=vooo, ooov=ooov,
                oooo=oooo)

        # build beta integrals
        vvvv = integrals.get_phys_anti(mol,vb,vb,vb,vb)
        vvvo = integrals.get_phys_anti(mol,vb,vb,vb,ob)
        vovv = integrals.get_phys_anti(mol,vb,ob,vb,vb)
        vvoo = integrals.get_phys_anti(mol,vb,vb,ob,ob)
        vovo = integrals.get_phys_anti(mol,vb,ob,vb,ob)
        oovv = integrals.get_phys_anti(mol,ob,ob,vb,vb)
        vooo = integrals.get_phys_anti(mol,vb,ob,ob,ob)
        ooov = integrals.get_phys_anti(mol,ob,ob,ob,vb)
        oooo = integrals.get_phys_anti(mol,ob,ob,ob,ob)
        Ib = two_e_blocks(
                vvvv=vvvv, vvvo=vvvo,
                vovv=vovv, vvoo=vvoo,
                vovo=vovo, oovv=oovv,
                vooo=vooo, ooov=ooov,
                oooo=oooo)

        # build abab integrals
        vvvv = integrals.get_phys(mol,va,vb,va,vb)
        vvvo = integrals.get_phys(mol,va,vb,va,ob)
        vvov = integrals.get_phys(mol,va,vb,oa,vb)
        vovv = integrals.get_phys(mol,va,ob,va,vb)
        ovvv = integrals.get_phys(mol,oa,vb,va,vb)
        vvoo = integrals.get_phys(mol,va,vb,oa,ob)
        vovo = integrals.get_phys(mol,va,ob,va,ob)
        ovvo = integrals.get_phys(mol,oa,vb,va,ob)
        ovov = integrals.get_phys(mol,oa,vb,oa,vb)
        oovv = integrals.get_phys(mol,oa,ob,va,vb)
        voov = integrals.get_phys(mol,va,ob,oa,vb)
        vooo = integrals.get_phys(mol,va,ob,oa,ob)
        ovoo = integrals.get_phys(mol,oa,vb,oa,ob)
        oovo = integrals.get_phys(mol,oa,ob,va,ob)
        ooov = integrals.get_phys(mol,oa,ob,oa,vb)
        oooo = integrals.get_phys(mol,oa,ob,oa,ob)
        Iabab = two_e_blocks_full(vvvv=vvvv,
            vvvo=vvvo,vvov=vvov,
            vovv=vovv,ovvv=ovvv,
            vvoo=vvoo,vovo=vovo,
            ovov=ovov,voov=voov,
            ovvo=ovvo,oovv=oovv,
            vooo=vooo,ovoo=ovoo,
            oovo=oovo,ooov=ooov,
            oooo=oooo)

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

        mol = self.mf.mol
        Ia = integrals.get_phys_anti(mol,moa,moa,moa,moa)
        Ib = integrals.get_phys_anti(mol,mob,mob,mob,mob)
        Iabab = integrals.get_phys(mol,moa,mob,moa,mob)

        return Ia,Ib,Iabab

    def g_int_tot(self):
        return integrals.get_phys_antiu_all_gen(self.mf,anti=False)
