import numpy
from cqcpy import ft_utils
from . import cc_utils
from . import ft_cc_energy
from . import ft_cc_equations
from . import ft_mp
from . import quadrature

class neq_ccsd(object):
    """Non-equilibrium coupled cluster singles and doubles"""
    def __init__(self, sys, T, mu=0.0, tmax=0.0, econv=1e-8,
        max_iter=40, damp=0.0, ngr=100, ngi=10, iprint=1):

        self.T = T
        self.mu = mu
        self.econv = econv
        self.max_iter = max_iter
        self.alpha = damp
        self.tmax = tmax
        self.ngr = ngr
        self.ngi = ngi
        self.iprint = iprint
        if not sys.verify(self.T,self.mu):
            raise Exception("Sytem temperature inconsistent with CC temp")
        self.sys = sys

    def run(self,T1=None,T2=None):
        """Run CCSD calculation."""
        return self._neq_ccsd(T1in=T1,T2in=T2)

    def _neq_ccsd(self,T1in=None,T2in=None):

        T = self.T
        beta = 1.0 / (T + 1e-12)
        tmax = self.tmax
        mu = self.mu

        # get time-grid
        ngr = self.ngr
        ngi = self.ngi
        deltar = tmax/(ngr)
        deltai = beta/(ngi - 1)
        tir = numpy.asarray([float(i)*deltar + deltar/2 for i in range(ngr)])
        tii = numpy.asarray([float(i)*deltai for i in range(ngi)])
        Gi = quadrature.get_G(ngi,deltai)
        gi = quadrature.get_gint(ngi, deltai)
        Gr = quadrature.get_G_midpoint(ngr,deltar)
        gr = quadrature.get_g_midpoint(ngr, deltar)
        self.deltar = deltar
        self.deltai = deltai
        self.gr = gr
        self.Gr = Gr
        self.gi = gi
        self.Gi = Gi
        self.tir = tir
        self.tii = tii

        # get orbital energies
        en = self.sys.g_energies_tot()

        # get 0th and 1st order contributions
        En = self.sys.const_energy()
        g0 = ft_utils.GP0(beta, en, mu)
        E0 = ft_mp.mp0(g0) + En
        E1 = self.sys.get_mp1()
        E01 = E0 + E1

        # get scaled integrals
        F,Ff,Fb,I = cc_utils.get_ft_integrals_neq(self.sys, en, beta, mu)
        ot = self.sys.ot

        # get energy differences
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
                - en[None,None,:,None] - en[None,None,None,:]

        # get MP2 T-amplitudes
        if T1in is not None and T2in is not None:
            T1oldf = T1in[0:ngr]
            T1oldb = T1in[ngr:ngr+ngi]
            T1oldi = T1in[ngr+ngi:]
            T2oldf = T2in[0:ngr]
            T2oldb = T2in[ngr:ngr+ngi]
            T2oldi = T2in[ngr+ngi:]
        else:
            T1oldf = -Ff.vo.copy()
            T1oldb = -Fb.vo.copy()
            Idr = numpy.ones((ngr))
            Idi = numpy.ones((ngi))
            T1oldi = -numpy.einsum('v,ai->vai',Idi,F.vo)
            T2oldb = -numpy.einsum('v,abij->vabij',Idr,I.vvoo)
            T2oldf = T2oldb.copy() 
            T2oldi = -numpy.einsum('v,abij->vabij',Idi,I.vvoo)

            T1oldf,T1oldb,T1oldi = quadrature.int_tbar1_keldysh(
                ngr,ngi,T1oldf,T1oldb,T1oldi,tir,tii,D1,Gr,Gi)
            T2oldf,T2oldb,T2oldi = quadrature.int_tbar2_keldysh(
                ngr,ngi,T2oldf,T2oldb,T2oldi,tir,tii,D2,Gr,Gi)

        Ei = ft_cc_energy.ft_cc_energy_neq(
            T1oldf,T1oldb,T1oldi,T2oldf,T2oldb,T2oldi,
            Ff.ov,Fb.ov,F.ov,I.oovv,gr,gi,beta,Qterm=False)
        if self.iprint > 0:
            print('MP2 Energy {:.10f}'.format(Ei))

        converged = False
        thresh = self.econv
        max_iter = self.max_iter
        alpha = self.alpha
        i = 0
        Eold = 888888888.888888888
        nl1 = numpy.linalg.norm(T1oldf) + 0.0001
        nl1 += numpy.linalg.norm(T1oldb)
        nl1 += numpy.linalg.norm(T1oldi)
        nl2 = numpy.linalg.norm(T2oldf) + 0.0001
        nl2 += numpy.linalg.norm(T2oldb)
        nl2 += numpy.linalg.norm(T2oldi)
        while i < max_iter and not converged:
            # form new T1 and T2
            T1f,T1b,T1i,T2f,T2b,T2i = \
                ft_cc_equations.neq_ccsd_stanton(Ff,Fb,F,I,T1oldf,T1oldb,T1oldi,
                        T2oldf,T2oldb,T2oldi,D1,D2,tir,tii,ngr,ngi,Gr,Gi)

            res1 = numpy.linalg.norm(T1f - T1oldf) / nl1
            res1 += numpy.linalg.norm(T1b - T1oldb) / nl1
            res1 += numpy.linalg.norm(T1i - T1oldi) / nl1
            res2 = numpy.linalg.norm(T2f - T2oldf) / nl2
            res2 += numpy.linalg.norm(T2b - T2oldb) / nl2
            res2 += numpy.linalg.norm(T2i - T2oldi) / nl2
            # damp new T-amplitudes
            T1oldf = alpha*T1oldf + (1.0 - alpha)*T1f
            T1oldb = alpha*T1oldb + (1.0 - alpha)*T1b
            T1oldi = alpha*T1oldi + (1.0 - alpha)*T1i
            T2oldf = alpha*T2oldf + (1.0 - alpha)*T2f
            T2oldb = alpha*T2oldb + (1.0 - alpha)*T2b
            T2oldi = alpha*T2oldi + (1.0 - alpha)*T2i
            nl1 = numpy.linalg.norm(T1oldf) + 0.0001
            nl1 += numpy.linalg.norm(T1oldb)
            nl1 += numpy.linalg.norm(T1oldi)
            nl2 = numpy.linalg.norm(T2oldf) + 0.0001
            nl2 += numpy.linalg.norm(T2oldb)
            nl2 += numpy.linalg.norm(T2oldi)

            # compute energy
            E = ft_cc_energy.ft_cc_energy_neq(
                T1oldf,T1oldb,T1oldi,T2oldf,T2oldb,T2oldi,
                Ff.ov,Fb.ov,F.ov,I.oovv,gr,gi,beta)

            # determine convergence
            if self.iprint > 0:
                print(' %2d  (%.8f,%.8f)   %.4E' % (i+1,E.real,E.imag,res1+res2))
            i = i + 1
            if numpy.abs(E - Eold) < thresh:
                converged = True
            Eold = E

        if not converged:
            print("WARNING: {} did not converge!".format(method))

        self.T1f = T1oldf
        self.T1b = T1oldb
        self.T1i = T1oldi
        self.T2f = T2oldf
        self.T2b = T2oldb
        self.T2i = T2oldi

        return (Eold + E01,Eold)

    def _neq_ccsd_lambda(self, L1=None, L2=None):
        """Solve FT-CCSD Lambda equations."""
        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu

        # get time-grid
        ngr = self.ngr
        ngi = self.ngi
        tir = self.tir
        tii = self.tii
        Gi = self.Gi
        gi = self.gi
        Gr = self.Gr
        gr = self.gr

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()

        En = self.sys.const_energy()
        g0 = ft_utils.GP0(beta, en, mu)
        E0 = ft_mp.mp0(g0) + En
        E1 = self.sys.get_mp1()
        E01 = E0 + E1

        # get scaled integrals
        F,Ff,Fb,I = cc_utils.get_ft_integrals_neq(self.sys, en, beta, mu)

        # get energy differences
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
                - en[None,None,:,None] - en[None,None,None,:]

        if L2 is None:
            # Use T^{\dagger} as a guess for Lambda
            L1oldf = numpy.transpose(self.T1f,(0,2,1))
            L1oldb = numpy.transpose(self.T1b,(0,2,1))
            L1oldi = numpy.transpose(self.T1i,(0,2,1))
            L2oldf = numpy.transpose(self.T2f,(0,3,4,1,2))
            L2oldb = numpy.transpose(self.T2b,(0,3,4,1,2))
            L2oldi = numpy.transpose(self.T2i,(0,3,4,1,2))
        else:
            L2oldf = L2[0]
            L2oldb = L2[1]
            L2oldi = L2[2]
            if L1 is None:
                L1oldf = numpy.zeros(self.T1f.shape)
                L1oldb = numpy.zeros(self.T1b.shape)
                L1oldi = numpy.zeros(self.T1i.shape)
            else:
                L1oldf = L1[0]
                L1oldb = L1[1]
                L1oldi = L1[2]

        # run lambda iterations
        thresh = self.econv
        max_iter = self.max_iter
        alpha = self.alpha
        i = 0
        Eold = 888888888.888888888
        nl1 = numpy.linalg.norm(L1oldf) + 0.0001
        nl1 += numpy.linalg.norm(L1oldb)
        nl1 += numpy.linalg.norm(L1oldi)
        nl2 = numpy.linalg.norm(L2oldf) + 0.0001
        nl2 += numpy.linalg.norm(L2oldb)
        nl2 += numpy.linalg.norm(L2oldi)
        converged = False
        while i < max_iter and not converged:
            # form new T1 and T2
            L1f,L1b,L1i,L2f,L2b,L2i = \
                ft_cc_equations.neq_lambda_opt(Ff,Fb,F,I,L1oldf,L1oldb,L1oldi,
                        L2oldf,L2oldb,L2oldi,self.T1f,self.T1b,self.T1i,
                        self.T2f,self.T2b,self.T2i,D1,D2,tir,tii,ngr,ngi,gr,gi,Gr,Gi)

            res1 = numpy.linalg.norm(L1f - L1oldf) / nl1
            res1 += numpy.linalg.norm(L1b - L1oldb) / nl1
            res1 += numpy.linalg.norm(L1i - L1oldi) / nl1
            res2 = numpy.linalg.norm(L2f - L2oldf) / nl2
            res2 += numpy.linalg.norm(L2b - L2oldb) / nl2
            res2 += numpy.linalg.norm(L2i - L2oldi) / nl2
            # damp new L-amplitudes
            L1oldf = alpha*L1oldf + (1.0 - alpha)*L1f
            L1oldb = alpha*L1oldb + (1.0 - alpha)*L1b
            L1oldi = alpha*L1oldi + (1.0 - alpha)*L1i
            L2oldf = alpha*L2oldf + (1.0 - alpha)*L2f
            L2oldb = alpha*L2oldb + (1.0 - alpha)*L2b
            L2oldi = alpha*L2oldi + (1.0 - alpha)*L2i
            nl1 = numpy.linalg.norm(L1oldf) + 0.0001
            nl1 += numpy.linalg.norm(L1oldb)
            nl1 += numpy.linalg.norm(L1oldi)
            nl2 = numpy.linalg.norm(L2oldf) + 0.0001
            nl2 += numpy.linalg.norm(L2oldb)
            nl2 += numpy.linalg.norm(L2oldi)

            # determine convergence
            if self.iprint > 0:
                print(' %2d  %.6E' % (i+1,res1+res2))
            i = i + 1
            if res1 + res2 < thresh:
                converged = True

        # save lambda amplitudes
        self.L1f = L1f
        self.L1b = L1b
        self.L1i = L1i
        self.L2f = L2f
        self.L2b = L2b
        self.L2i = L2i

    def _neq_1rdm(self):
        if self.L2f is None or self.T2f is None:
            raise Exception("Cannot compute density without Lambda!")

        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu

        # get time-grid
        ngr = self.ngr
        ngi = self.ngi
        tir = self.tir
        tii = self.tii
        Gi = self.Gi
        gi = self.gi
        Gr = self.Gr
        gr = self.gr


        # get energies and occupation numbers
        en = self.sys.g_energies_tot()

        # get energy differences
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
                - en[None,None,:,None] - en[None,None,None,:]

        pia,pba,pji,pai = ft_cc_equations.neq_1rdm(
                self.T1f,self.T1b,self.T1i,
                self.T2f,self.T2b,self.T2i,
                self.L1f,self.L1b,self.L1i,
                self.L2f,self.L2b,self.L2i,
                D1,D2,tir,tii,ngr,ngi,gr,gi,Gr,Gi)

        self.dia = pia
        self.dba = pba
        self.dji = pji
        self.dai = pai

    def compute_prop(self, A, t):
        pia = self.dia[t]
        pba = self.dba[t]
        pji = self.dji[t]
        pai = self.dai[t]

        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)
        ngr = self.ngr
        deltar = self.deltar
        tir = self.tir
        tii = self.tii
        Gi = self.Gi
        gi = self.gi
        Gr = self.Gr
        gr = self.gr

        # compute first order part
        prop = numpy.einsum('ii,i->',A,fo)

        # compute higher order contribution
        E21 = numpy.einsum('ai,ia,i,a->',A,pia,fo,fv)/(deltar)
        E22 = numpy.einsum('ab,ba,a->',A,pba,fv)/(deltar)
        E23 = numpy.einsum('ij,ji,j->',A,pji,fo)/(deltar)
        E24 = numpy.einsum('ia,ai->',A,pai)/(deltar)
        #print(E21,E22,E23,E24)
        E2 = E21 + E22 + E23 + E24
        #E2 += 1.j*0.2*numpy.einsum('ai,ia,i,a->',A,pia,fo,fv)/gr[t]
        #E2 += 1.j*0.2*numpy.einsum('ab,ba,a,b->',A,pba,fv,fv)/gr[t]
        #E2 += 1.j*0.2*numpy.einsum('ij,ji,j,i->',A,pji,fo,fo)/gr[t]
        #E2 += 1.j*0.2*numpy.einsum('ia,ai,a,i->',A,pai,fv,fo)/gr[t]
        prop += E2

        return prop

    def compute_2e_prop(self, A, t):

        # compute 2-rdm on the fly
        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)
        ngr = self.ngr
        deltar = self.deltar
        tir = self.tir
        tii = self.tii
        Gi = self.Gi
        gi = self.gi
        Gr = self.Gr
        gr = self.gr

        # compute first order part
        prop = numpy.einsum('ijij,i,j->',A,fo,fo)
        prop -= numpy.einsum('ijji,i,j->',A,fo,fo)

        # compute higher order contribution
        A1 = 0.25*einsum('cdab,abcd,a,b,c,d->',self.P2[0],A,fv,fv,fv,fv)
        A2 = 0.5*einsum('ciab,abci,i,a,b,c->',self.P2[1],A,fo,fv,fv,fv)
        A3 = 0.5*einsum('bcai,aibc,i,a,b,c->',self.P2[2],A,fo,fv,fv,fv)
        A4 = 0.25*einsum('ijab,abij,i,j,a,b->',self.P2[3],A,fo,fo,fv,fv)
        A5 = 1.0*einsum('bjai,aibj,i,j,a,b->',self.P2[4],A,fo,fo,fv,fv)
        A6 = 0.25*einsum('abij,ijab,i,j,a,b->',self.P2[5],A,fo,fo,fv,fv)
        A7 = 0.5*einsum('jkai,aijk,i,j,k,a->',self.P2[6],A,fo,fo,fo,fv)
        A8 = 0.5*einsum('kaij,ijka,i,j,k,a->',self.P2[7],A,fo,fo,fo,fv)
        A9 = 0.25*einsum('klij,ijkl,i,j,k,l->',self.P2[8],A,fo,fo,fo,fo)
        E2 = A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9
        prop += E2
        return prop
