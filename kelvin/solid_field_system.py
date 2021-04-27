import numpy
from cqcpy import ft_utils
from cqcpy import utils
from cqcpy import integrals

#from .neq_system import *
from .system import *
from . import scf_utils

class solid_field_system(system):

    def __init__(self, T, mf, ti, A0, t0, sigma, omega, mu=0.0):
        self.A0 = A0
        self.t0 = t0
        self.sigma = sigma
        self.omega = omega
        self.ti = ti
        self.T = T
        self.mu = mu
        self.mf = mf
        self.ot = None

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
            beta = 1.0 / self.T
            ea,eb = self.u_energies_tot()
            na = ea.shape[0]
            nb = eb.shape[0]
            foa = ft_utils.ff(beta, ea, self.mu)
            fob = ft_utils.ff(beta, eb, self.mu)
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
            hcore = self.mf.get_hcore(self.mf.mol)
            hmoa,hmob = scf_utils.u_mo_tran_1e(self.mf, hcore)
            Fa = hmoa.copy()#numpy.einsum('ij,ip,jq->pq',Ta,self.ua,self.ua)
            Fb = hmob.copy()#numpy.einsum('ij,ip,jq->pq',Tb,self.ub,self.ub)
            Fa += JKa.copy()
            Fb += JKb.copy()
            Fao = Fa - numpy.diag(ea)
            Fbo = Fb - numpy.diag(eb)
            E1 += numpy.einsum('ii,i->',Fao,foa)
            E1 += numpy.einsum('ii,i->',Fbo,fob)
            return E1

    def u_energies_tot(self):
        return scf_utils.get_u_orbital_energies_tot(self.mf)

    def g_energies_tot(self):
        return scf_utils.get_orbital_energies_gen(self.mf)

    def u_fock_tot(self,direc='f'):
        beta = 1.0 / self.T if self.T > 0 else 1.0e20
        ea,eb = self.u_energies_tot()
        foa = ft_utils.ff(beta, ea, self.mu)
        fob = ft_utils.ff(beta, eb, self.mu)
        fa,fb = scf_utils.get_u_ft_fock(self.mf, foa, fob)
        nt = self.ti.shape[0]
        na = ea.shape[0]
        nb = eb.shape[0]
        assert(na == nb)
        px,py,pz = self.mf.cell.pbc_intor('int1e_ipovlp', hermi=0, comp=3)
        px = -1.j*px.conj().transpose((1,0))
        py = -1.j*py.conj().transpose((1,0))
        pz = -1.j*pz.conj().transpose((1,0))
        pax,pbx = scf_utils.u_mo_tran_1e(self.mf, px)
        pay,pby = scf_utils.u_mo_tran_1e(self.mf, py)
        paz,pbz = scf_utils.u_mo_tran_1e(self.mf, pz)
        Tta = numpy.zeros((nt,na,na),dtype=complex)
        Ttb = numpy.zeros((nt,nb,nb),dtype=complex)
        for i,t in enumerate(self.ti):
            dt = t - self.t0
            ex = dt*dt/(2*self.sigma*self.sigma)
            phase = self.A0*numpy.exp(-ex)*numpy.cos(self.omega*dt)
            Tta[i] = phase*paz
            Ttb[i] = phase*pbz
        Fa = Tta.copy()
        Fb = Ttb.copy()
        Fa += fa[None,:,:]
        Fb += fb[None,:,:]
        if direc == 'b':
            tempa = Fa.copy()
            tempb = Fb.copy()
            ng = len(self.ti)
            for i in range(ng):
                Fa[i] = tempa[ng - i - 1]
                Fb[i] = tempb[ng - i - 1]
        return Fa,Fb

    def g_fock_tot(self,direc='f'):
        Fa,Fb = self.u_fock_tot()
        nt,na1,na2 = Fa.shape
        assert(na1 == na2)
        Fock = numpy.zeros((nt,2*na1,2*na2),dtype=complex)
        for i in range(nt):
            Fock[i] = utils.block_diag(Fa[i],Fb[i])
        return Fock

    def u_aint_tot(self):
        mo_occ = self.mf.mo_occ
        if len(mo_occ.shape) == 1:
            moa = self.mf.mo_coeff
            mob = moa
        elif len(mo_occ.shape) == 2:
            moa = self.mf.mo_coeff[0]
            mob = self.mf.mo_coeff[1]

        mf = self.mf
        Ia = integrals.get_phys_gen(mf,moa,moa,moa,moa,anti=True)
        Ib = integrals.get_phys_gen(mf,mob,mob,mob,mob,anti=True)
        Iabab = integrals.get_phys_gen(mf,moa,mob,moa,mob,anti=False)

        return Ia,Ib,Iabab

    def g_aint_tot(self):
        return integrals.get_phys_antiu_all_gen(self.mf)
