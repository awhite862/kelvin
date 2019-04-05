import numpy
import time
from pyscf import lib
from cqcpy import cc_energy
from cqcpy import cc_equations
from cqcpy import ft_utils
from . import zt_mp
from . import ft_mp
from . import cc_utils
from . import ft_cc_energy
from . import ft_cc_equations
from . import quadrature

einsum = lib.einsum
#einsum = numpy.einsum

class ccsd(object):
    """Coupled cluster singles and doubles (CCSD) driver.

    Attributes:
        sys: System object.
        T (float): Temperature.    
        mu (float): Chemical potential.
        iprint (int): Print level.
        singles (bool): Include singles (False -> CCD).
        econv (float): Energy difference convergence threshold.
        tconv (float): Amplitude difference convergence threshold.
        max_iter (int): Max number of iterations.
        damp (float): Mixing parameter to damp iterations.
        ngrid (int): Number of grid points.
        realtime (bool): Force time-dependent formulation for zero T
        athresh (float): Threshold for ignoring small occupations
        quad (string): Transformation for quadrature rule
        rt_iter (string): "all" or "point" for all-at-once or pointwise convergence
        T1: Saved T1 amplitudes
        T2: Saved T2 amplitudes
        L1: Saved L1 amplitudes
        L2: Saved L2 amplitudes
    """
    def __init__(self, sys, T=0.0, mu=0.0, iprint=0,
        singles=True, econv=1e-8, tconv=None, max_iter=40,
        damp=0.0, ngrid=10, realtime=False, athresh=0.0, 
        quad='lin', rt_iter="all"):

        self.T = T
        self.mu = mu
        self.finite_T = False if T == 0 else True
        self.iprint = iprint
        self.singles = singles
        self.econv = econv
        self.tconv = tconv if tconv is not None else 1000.0*econv
        self.max_iter = max_iter
        self.damp = damp
        self.ngrid = ngrid
        self.realtime = realtime
        self.athresh = athresh
        self.quad = quad
        self.rt_iter = rt_iter
        if self.finite_T:
            self.realtime = True
        if not sys.verify(self.T,self.mu):
            raise Exception("Sytem temperature inconsistent with CC temp")
        if self.realtime:
            if self.finite_T:
                beta_max = 1.0/(T + 1e-12)
            else:
                beta_max = 80
                self._beta_max = 80
            ng = self.ngrid
            self.ti,self.g,self.G = quadrature.ft_quad(self.ngrid,beta_max,self.quad)
        self.sys = sys
        self.T1 = None
        self.T2 = None
        self.L1 = None
        self.L2 = None
        self.dia = None
        self.dba = None
        self.dji = None
        self.dai = None
        self.P2 = None

    def run(self,T1=None,T2=None):
        """Run CCSD calculation."""
        if self.finite_T:
            if self.iprint > 0:
                print('Running CCSD at an electronic temperature of %f K'
                    % ft_utils.HtoK(self.T))
            if self.sys.has_u():
                return self._ft_uccsd(T1in=T1,T2in=T2)
            else:
                return self._ft_ccsd(T1in=T1,T2in=T2)
        else:
            if self.iprint > 0:
                print('Running CCSD at zero Temperature')
            if self.realtime:
                return self._ft_ccsd()
            else:
                if self.sys.has_u():
                    return self._uccsd(T1in=T1,T2in=T2)
                else:
                    return self._ccsd()

    def compute_ESN(self,L1=None,L2=None,gderiv=True):
        """Compute energy, entropy, particle number."""
        if not self.finite_T:
            N = self.sys.g_energies()[0].shape[0]
            print("T = 0: ")
            print('  E = {}'.format(self.Etot))
            print('  S = {}'.format(0.0))
            print('  N = {}'.format(N))
        else:
            if self.L1 is None:
                if self.sys.has_u():
                    self._ft_uccsd_lambda(L1=L1,L2=L2)
                    ti = time.time()
                    self._u_ft_1rdm()
                    self._u_ft_2rdm()
                    tf = time.time()
                    if self.iprint > 0:
                        print("RDM construction time: {} s".format(tf - ti))
                else:
                    self._ft_ccsd_lambda(L1=L1,L2=L2)
                    self._g_ft_1rdm()
                    self._g_ft_2rdm()
            if self.sys.has_u():
                ti = time.time()
                self._u_ft_ESN(L1,L2,gderiv=gderiv)
                tf = time.time()
                if self.iprint > 0:
                    print("Total derivative time: {} s".format(tf - ti))
            else:
                self._g_ft_ESN(L1,L2,gderiv=gderiv)

    #def compute_1eprop(self,A,L1=None,L2=None):
    #    if not self.finite_T:
    #        raise Exception("Not implemented")
    #    else:
    #        if self.L1 is None:
    #            if self.sys.has_u():
    #                self._ft_uccsd_lambda(L1=L1,L2=L2)
    #                self._u_ft_1rdm()
    #                self._u_ft_2rdm()
    #            else:
    #                self._ft_ccsd_lambda(L1=L1,L2=L2)
    #                self._g_ft_1rdm()
    #                self._g_ft_2rdm()
    #        if self.sys.has_u():
    #            raise Exception("Not yet implemented")
    #        else:
    #            # temperature info
    #            T = self.T
    #            beta = 1.0 / (T + 1e-12)
    #            mu = self.mu

    #            # zero order contributions
    #            en = self.sys.g_energies_tot()
    #            fo = ft_utils.ff(beta, en, mu)
    #            A0 = numpy.einsum('i,i->',fo,A.diagonal())
    #            A1,Acc = self._g_nocc_deriv(A.diagonal()*beta)

    #            D1 = en[:,None] - en[None,:]
    #            D2 = en[:,None,None,None] + en[None,:,None,None] \
    #                - en[None,None,:,None] - en[None,None,None,:]
    #            F,I = cc_utils.ft_integrals(self.sys, en, beta, mu)
    #            Gnew = self.G.copy()
    #            m = Gnew.shape[0]
    #            n = Gnew.shape[0]
    #            for i in range(m):
    #                for j in range(n):
    #                    Gnew[i,j] *= (self.ti[j] - self.ti[i])

    #            ng = self.ti.shape[0]
    #            T1temp,T2temp = ft_cc_equations.ccsd_stanton(F,I,self.T1,self.T2,
    #                    D1,D2,self.ti,ng,Gnew)
    #            DA1 = A.diagonal()[:,None] - A.diagonal()[None,:]
    #            DA2 = A.diagonal()[:,None,None,None] + A.diagonal()[None,:,None,None] \
    #                    - A.diagonal()[None,None,:,None] - A.diagonal()[None,None,None,:]
    #            T1temp *= DA1
    #            T2temp *= DA2

    #            At1 = (1.0/beta)*einsum('via,vai->v',self.L1, T1temp)
    #            At2 = (1.0/beta)*0.25*einsum('vijab,vabij->v',self.L2, T2temp)
    #            At1g = einsum('v,v->',At1,self.g)
    #            At2g = einsum('v,v->',At2,self.g)
    #            return A0,A1,(Acc + At1g + At2g)

    def _g_ft_ESN(self,L1=None,L2=None,gderiv=True):
            # temperature info
            T = self.T
            beta = 1.0 / (T + 1e-12)
            mu = self.mu

            # zero order contributions
            en = self.sys.g_energies_tot()
            fo = ft_utils.ff(beta, en, mu)
            B0 = ft_utils.dGP0(beta, en, mu)
            N0 = fo.sum()
            E0 = beta*B0.sum() + mu*N0 + self.G0

            # higher order contributions
            dvec = -beta*numpy.ones(en.shape) # mu derivative
            N1,Ncc = self._g_nocc_deriv(dvec)
            N1 *= -1.0 # N = - dG/dmu
            Ncc *= -1.0
            dvec = en - mu # beta derivative
            B1,Bcc = self._g_nocc_deriv(dvec)

            # compute other contributions to CC derivative
            Bcc -= self.Gcc/(beta) # derivative from factors of 1/beta
            if gderiv:
                dg,dG = self._g_nocc_gderiv()
                Bcc += dG + dg
            else:
                Bcc += self._g_gderiv_approx()

            # E = beta*dG/dbeta + G + mu*N
            E1 = beta*B1 + mu*N1 + self.G1
            Ecc = beta*Bcc + mu*Ncc + self.Gcc

            self.N0 = N0
            self.N1 = N1
            self.Ncc = Ncc
            self.N = Ncc + N0 + N1
            self.E0 = E0
            self.E1 = E1
            self.Ecc = Ecc
            self.E = E0 + E1 + Ecc
            self.S = -beta*(self.Gtot - self.E + mu*self.N)
            self.S0 = -beta*(self.G0 - self.E0 + mu*self.N0)
            self.S1 = -beta*(self.G1 - self.E1 + mu*self.N1)
            self.Scc = self.S - self.S0 - self.S1

    def _u_ft_ESN(self,L1=None,L2=None,gderiv=True):
            # temperature info
            T = self.T
            beta = 1.0 / (T + 1e-12)
            mu = self.mu

            # zero order contributions
            en = self.sys.g_energies_tot()
            ea,eb = self.sys.u_energies_tot()
            foa = ft_utils.ff(beta, ea, mu)
            fob = ft_utils.ff(beta, eb, mu)
            B0 = ft_utils.dGP0(beta, en, mu)
            B0a = ft_utils.dGP0(beta, ea, mu)
            B0b = ft_utils.dGP0(beta, eb, mu)
            N0 = foa.sum() + fob.sum()
            E0 = beta*(B0a.sum() + B0b.sum()) + mu*N0 + self.G0

            # higher order contributions
            dveca = -beta*numpy.ones(ea.shape) # mu derivative
            dvecb = -beta*numpy.ones(eb.shape) # mu derivative
            N1,Ncc = self._u_nocc_deriv(dveca,dvecb)
            N1 *= -1.0 # N = - dG/dmu
            Ncc *= -1.0
            dveca = ea - mu
            dvecb = eb - mu
            B1,Bcc = self._u_nocc_deriv(dveca,dvecb)

            # compute other contributions to CC derivative
            Bcc -= self.Gcc/(beta)
            if gderiv:
                dg,dG = self._u_nocc_gderiv()
                Bcc += dG + dg
            else:
                Bcc += self._u_gderiv_approx()

            # E = beta*dG/dbeta + G + mu*N
            E1 = beta*B1 + mu*N1 + self.G1
            Ecc = beta*Bcc + mu*Ncc + self.Gcc

            self.N0 = N0
            self.N1 = N1
            self.Ncc = Ncc
            self.N = Ncc + N0 + N1
            self.E0 = E0
            self.E1 = E1
            self.Ecc = Ecc
            self.E = E0 + E1 + Ecc
            self.S0 = -beta*(self.G0 - self.E0 + mu*self.N0)
            self.S1 = -beta*(self.G1 - self.E1 + mu*self.N1)
            self.S = -beta*(self.Gtot - self.E + mu*self.N)

    def _ccsd(self):
        """Simple CCSD implementation at zero temperature."""
        # create energies and denominators in spin-orbital basis
        tbeg = time.time()
        eo,ev = self.sys.g_energies()
        Dov = 1.0/(eo[:,None] - ev[None,:])
        Doovv = 1.0/(eo[:,None,None,None] + eo[None,:,None,None]
            - ev[None,None,:,None] - ev[None,None,None,:])

        # get HF energy 
        t1 = time.time()
        En = self.sys.const_energy()
        E0 = zt_mp.mp0(eo) + En
        E1 = self.sys.get_mp1()
        Ehf = E0 + E1
        t2 = time.time()
    
        # compute required memory
        no = eo.shape[0] 
        nv = ev.shape[0] 
        noa = no//2
        nva = nv//2
        mem1e = no*no + 5*no*nv + nv*nv  # include memory for D1
        mem2e = 6*no*no*nv*nv + nv*nv*nv*nv + 2*nv*nv*nv*no + \
                2*nv*no*no*no + no*no*no*no # include memory for D2
        mem_mb = 2.0*(mem1e + mem2e)*8.0/1024.0/1024.0
        if self.iprint > 0:
            print('  CCSD will use %f mb' % mem_mb)

        # get Fock matrix
        t3 = time.time()
        F = self.sys.g_fock()
        F.oo = F.oo - numpy.diag(eo) # subtract diagonal
        F.vv = F.vv - numpy.diag(ev) # subtract diagonal

        # get ERIs
        I = self.sys.g_aint()
        t4 = time.time()

        # get MP2 T-amplitudes
        T1old = einsum('ai,ia->ai',F.vo,Dov)
        T2old = einsum('abij,ijab->abij',I.vvoo,Doovv)
        Es = cc_energy.cc_energy_s1(T1old, F.vo.transpose(1,0))
        Ed = cc_energy.cc_energy_d(T2old, I.vvoo.transpose(2,3,0,1))
        Emp2 = Es + Ed
        t5 = time.time()
        if self.iprint > 0:
            print('MP2 Energy {:.10f}'.format(Emp2))

        # run CC iterations
        converged = False
        max_iter = self.max_iter
        i = 0
        Eold = 1000000
        nl1 = numpy.sqrt(T1old.size)
        nl2 = numpy.sqrt(T2old.size)
        alpha = self.damp
        while i < max_iter and not converged:
            if self.singles:
                T1,T2 = cc_equations.ccsd_stanton(F,I,T1old,T2old)
            else:
                T1 = numpy.zeros(F.vo.shape)
                T2 = cc_equations.ccd_simple(F,I,T2old)
            T1 = einsum('ai,ia->ai',T1,Dov)
            T2 = einsum('abij,ijab->abij',T2,Doovv)
            res1 = numpy.linalg.norm(T1 - T1old) / nl1
            res2 = numpy.linalg.norm(T2 - T2old) / nl2
            T1old = alpha*T1old + (1.0 - alpha)*T1
            T2old = alpha*T2old + (1.0 - alpha)*T2
            E = cc_energy.cc_energy(T1old,T2old,F.ov,I.oovv)
            if self.iprint > 0:
                print(' {:2d}  {:.10f}  {:.10f}'.format(i+1,E,res1+res2))
            i = i + 1
            if numpy.abs(E - Eold) < self.econv and res1+res2 < self.tconv:
                converged = True
            Eold = E
            T1old = T1
            T2old = T2
            T1 = None
            T2 = None

        # save and return
        self.Ecor = Eold
        self.Etot = Ehf + Eold
        self.T1 = T1old
        self.T2 = T2old
        tend = time.time()
        tmp1 = t2 - t1
        tint = t4 - t3
        tmp2 = t5 - t4
        return (Eold+Ehf,Eold)

    def _uccsd(self,T1in=None,T2in=None):
        """Simple UCCSD implementation at zero temperature."""
        # create energies and denominators in spin-orbital basis
        tbeg = time.time()
        eoa,eva,eob,evb = self.sys.u_energies()
        Dova = 1.0/(eoa[:,None] - eva[None,:])
        Dovb = 1.0/(eob[:,None] - evb[None,:])
        Doovvaa = 1.0/(eoa[:,None,None,None] + eoa[None,:,None,None]
            - eva[None,None,:,None] - eva[None,None,None,:])
        Doovvbb = 1.0/(eob[:,None,None,None] + eob[None,:,None,None]
            - evb[None,None,:,None] - evb[None,None,None,:])
        Doovvab = 1.0/(eoa[:,None,None,None] + eob[None,:,None,None]
            - eva[None,None,:,None] - evb[None,None,None,:])

        # get HF energy
        t1 = time.time()
        En = self.sys.const_energy()
        E0 = zt_mp.ump0(eoa,eob) + En
        E1 = self.sys.get_mp1()
        Ehf = E0 + E1
        t2 = time.time()

        # compute required memory
        noa = eoa.shape[0]
        nva = eva.shape[0]
        nob = eob.shape[0]
        nvb = evb.shape[0]
        no = noa + nob
        nv = nva + nvb
        mem1e = no*no + 5*no*nv + nv*nv  # include memory for D1
        mem2e = 6*no*no*nv*nv + nv*nv*nv*nv + 2*nv*nv*nv*no + \
                2*nv*no*no*no + no*no*no*no # include memory for D2
        mem_mb = 2.0*(mem1e + mem2e)*8.0/1024.0/1024.0
        if self.iprint > 0:
            print('  CCSD will use %f mb' % mem_mb)

        # get Fock matrix
        t3 = time.time()
        Fa,Fb = self.sys.u_fock()
        Fa.oo = Fa.oo - numpy.diag(eoa) # subtract diagonal
        Fa.vv = Fa.vv - numpy.diag(eva) # subtract diagonal
        Fb.oo = Fb.oo - numpy.diag(eob) # subtract diagonal
        Fb.vv = Fb.vv - numpy.diag(evb) # subtract diagonal

        # get ERIs
        Ia, Ib, Iabab = self.sys.u_aint()

        t4 = time.time()

        # TODO: Add CCD
        if not self.singles:
            raise Exception("UCCD is not implemented")
        # get MP2 T-amplitudes
        T1aold = einsum('ai,ia->ai',Fa.vo,Dova)
        T1bold = einsum('ai,ia->ai',Fb.vo,Dovb)
        T2aaold = einsum('abij,ijab->abij',Ia.vvoo,Doovvaa)
        T2abold = einsum('abij,ijab->abij',Iabab.vvoo,Doovvab)
        T2bbold = einsum('abij,ijab->abij',Ib.vvoo,Doovvbb)
        T1olds = (T1aold,T1bold)
        T2olds = (T2aaold,T2abold,T2bbold)
        Emp2 = cc_energy.ump2_energy(T1olds,T2olds,
                Fa.ov,Fb.ov,Ia.oovv,Ib.oovv,Iabab.oovv)

        # start with custom guess if provided
        if T1in is not None:
            T1aold = T1in[0]
            T1bold = T1in[1]
        if T2in is not None:
            T2aaold = T2in[0]
            T2abold = T2in[1]
            T2bbold = T2in[2]
        T1olds = (T1aold,T1bold)
        T2olds = (T2aaold,T2abold,T2bbold)
        t5 = time.time()
        if self.iprint > 0:
            print('MP2 Energy {:.10f}'.format(Emp2))

        # run CC iterations
        converged = False
        max_iter = self.max_iter
        i = 0
        Eold = 1000000
        alpha = self.damp
        nl1 = numpy.sqrt(T1aold.size)
        nl2 = numpy.sqrt(T2aaold.size)
        while i < max_iter and not converged:
            (T1a,T1b),(T2aa,T2ab,T2bb) = cc_equations.uccsd_stanton(Fa, Fb, Ia, Ib, Iabab, T1olds, T2olds)
            T1a = einsum('ai,ia->ai',T1a,Dova)
            T1b = einsum('ai,ia->ai',T1b,Dovb)
            T2aa = einsum('abij,ijab->abij',T2aa,Doovvaa)
            T2ab = einsum('abij,ijab->abij',T2ab,Doovvab)
            T2bb = einsum('abij,ijab->abij',T2bb,Doovvbb)
            E = cc_energy.ucc_energy((T1a,T1b),(T2aa,T2ab,T2bb),
                    Fa.ov,Fb.ov,Ia.oovv,Ib.oovv,Iabab.oovv)
            res1 = numpy.linalg.norm(T1olds[0] - T1a)/nl1
            res1 += numpy.linalg.norm(T1olds[1] - T1b)/nl1
            res2 = numpy.linalg.norm(T2olds[0] - T2aa)/nl2
            res2 += numpy.linalg.norm(T2olds[1] - T2ab)/nl2
            res2 += numpy.linalg.norm(T2olds[2] - T2bb)/nl2
            T1a = alpha*T1olds[0] + (1.0 - alpha)*T1a
            T1b = alpha*T1olds[1] + (1.0 - alpha)*T1b
            T2aa = alpha*T2olds[0] + (1.0 - alpha)*T2aa
            T2ab = alpha*T2olds[1] + (1.0 - alpha)*T2ab
            T2bb = alpha*T2olds[2] + (1.0 - alpha)*T2bb
            if self.iprint > 0:
                print(' %2d  %.10f' % (i+1,E))
            i = i + 1
            if numpy.abs(E - Eold) < self.econv and res1+res2 < self.tconv:
                converged = True
            Eold = E

            T1olds = (T1a,T1b)
            T2olds = (T2aa,T2ab,T2bb)
            T1a = None
            T1b = None
            T2aa = None
            T2ab = None
            T2bb = None

        # save and return
        self.Ecor = Eold
        self.Etot = Ehf + Eold
        self.T1 = T1olds
        self.T2 = T2olds
        tend = time.time()
        tmp1 = t2 - t1
        tint = t4 - t3
        tmp2 = t5 - t4
        return (Eold+Ehf,Eold)

    def _ccsd_lambda(self):
        """Solve CCSD Lambda equations at zero temperature."""
        # create energies and denominators in spin-orbital basis
        tbeg = time.time()
        eo,ev = self.sys.g_energies()
        Dov = 1.0/(eo[:,None] - ev[None,:])
        Doovv = 1.0/(eo[:,None,None,None] + eo[None,:,None,None]
            - ev[None,None,:,None] - ev[None,None,None,:])
        Nov = 1.0/Dov
        Noovv = 1.0/Doovv

        # get Fock matrix
        t3 = time.time()
        F = self.sys.g_fock()
        F.oo = F.oo - numpy.diag(eo) # subtract diagonal
        F.vv = F.vv - numpy.diag(ev) # subtract diagonal

        # get ERIs
        I = self.sys.g_aint()
        t4 = time.time()

        # Initialize Lambda amplitudes and compute intermediates
        if self.singles:
            L1old = self.T1.transpose((1,0))
        else:
            L1old = numpy.zeros(F.ov.shape)
        L2old = self.T2.transpose((2,3,0,1))
        if self.singles:
            intor = cc_equations.lambda_int(F,I,self.T1,self.T2)

        # run CC-Lambda iterations
        converged = False
        nl1 = numpy.sqrt(L1old.size)
        nl2 = numpy.sqrt(L2old.size)
        max_iter = self.max_iter
        i = 0
        while i < max_iter and not converged:
            if self.singles:
                L1,L2 = cc_equations.ccsd_lambda_opt_int(F,I,L1old,L2old,self.T1,self.T2,intor)
            else:
                L1 = numpy.zeros(F.ov.shape)
                L2 = cc_equations.ccd_lambda_simple(F,I,L2old,self.T2)
            L1 = einsum('ia,ia->ia',L1,Dov)
            L2 = einsum('ijab,ijab->ijab',L2,Doovv)
            res1 = numpy.linalg.norm(L1 - L1old) / nl1
            res2 = numpy.linalg.norm(L2 - L2old) / nl2
            if self.iprint > 0:
                print(' %2d  %.10f' % (i+1,res1 + res2))
            i = i + 1
            if res1 + res2 < self.tconv:
                converged = True
            L1old = L1
            L2old = L2
            L1 = None
            L2 = None

        # save Lambdas
        self.L1 = L1old
        self.L2 = L2old
        tend = time.time()

    def _uccsd_lambda(self):
        """Solve CCSD Lambda equations at zero temperature."""
        # create energies and denominators in spin-orbital basis
        tbeg = time.time()
        eoa,eva,eob,evb = self.sys.u_energies()
        Dova = 1.0/(eoa[:,None] - eva[None,:])
        Dovb = 1.0/(eob[:,None] - evb[None,:])
        Doovvaa = 1.0/(eoa[:,None,None,None] + eoa[None,:,None,None]
            - eva[None,None,:,None] - eva[None,None,None,:])
        Doovvbb = 1.0/(eob[:,None,None,None] + eob[None,:,None,None]
            - evb[None,None,:,None] - evb[None,None,None,:])
        Doovvab = 1.0/(eoa[:,None,None,None] + eob[None,:,None,None]
            - eva[None,None,:,None] - evb[None,None,None,:])
        noa = eoa.shape[0]
        nva = eva.shape[0]
        nob = eob.shape[0]
        nvb = evb.shape[0]

        # get Fock matrix
        t3 = time.time()
        Fa,Fb = self.sys.u_fock()
        Fa.oo = Fa.oo - numpy.diag(eoa) # subtract diagonal
        Fa.vv = Fa.vv - numpy.diag(eva) # subtract diagonal
        Fb.oo = Fb.oo - numpy.diag(eob) # subtract diagonal
        Fb.vv = Fb.vv - numpy.diag(evb) # subtract diagonal

        # get ERIs
        Ia, Ib, Iabab = self.sys.u_aint()
        t4 = time.time()

        T1olds = self.T1
        T2olds = self.T2

        # Initialize Lambda amplitudes and compute intermediates
        if self.singles:
            L1aold = T1olds[0].transpose((1,0))
            L1bold = T1olds[1].transpose((1,0))
        else:
            raise Exception("UCCD Lambdas not implemented")
            #L1old = numpy.zeros(F.ov.shape)
        L2aaold = T2olds[0].transpose((2,3,0,1))
        L2abold = T2olds[1].transpose((2,3,0,1))
        L2bbold = T2olds[2].transpose((2,3,0,1))

        # run CC-Lambda iterations
        converged = False
        max_iter = self.max_iter
        nl1 = numpy.sqrt(L1aold.size + L1bold.size)
        nl2 = numpy.sqrt(L2aaold.size + L2abold.size + L2bbold.size)
        i = 0
        L1olds = (L1aold,L1bold)
        L2olds = (L2aaold,L2abold,L2bbold)
        while i < max_iter and not converged:
            if self.singles:
                L1s,L2s = cc_equations.uccsd_lambda_opt(Fa, Fb, Ia, Ib, Iabab, L1olds, L2olds, T1olds, T2olds)
                L1a,L1b = L1s
                L2aa,L2ab,L2bb = L2s
            else:
                raise Exception("UCCD Lambdas not implemented")
                L1 = numpy.zeros(F.ov.shape)
                L2 = cc_equations.ccd_lambda_simple(F,I,L2old,self.T2)
            L1a = einsum('ia,ia->ia',L1a,Dova)
            L1b = einsum('ia,ia->ia',L1b,Dovb)
            L2aa = einsum('ijab,ijab->ijab',L2aa,Doovvaa)
            L2ab = einsum('ijab,ijab->ijab',L2ab,Doovvab)
            L2bb = einsum('ijab,ijab->ijab',L2bb,Doovvbb)
            res1 = numpy.linalg.norm(L1a - L1aold) / nl1
            res1 += numpy.linalg.norm(L1b - L1bold) / nl1
            res2 = numpy.linalg.norm(L2aa - L2olds[0]) / nl2
            res2 += numpy.linalg.norm(L2ab - L2olds[1]) / nl2
            res2 += numpy.linalg.norm(L2bb - L2olds[2]) / nl2
            if self.iprint > 0:
                print(' %2d  %.10f' % (i+1,res1 + res2))
            i = i + 1
            if res1 + res2 < self.tconv:
                converged = True
            L1aold = L1a
            L1bold = L1b
            L2aaold = L2aa
            L2abold = L2ab
            L2bbold = L2bb
            L1olds = (L1aold,L1bold)
            L2olds = (L2aaold,L2abold,L2bbold)
            L1a = None
            L1b = None
            L2aa = None
            L2ab = None
            L2bb = None

        # save Lambdas
        self.L1 = (L1aold,L1bold)
        self.L2 = (L2aaold,L2abold,L2bbold)
        tend = time.time()

    def _ft_ccsd(self,T1in=None,T2in=None):
        """Solve finite temperature coupled cluster equations."""
        #T = self.T if not self.realtime else 1.0e-5
        beta = 1.0 / (self.T + 1e-12) if self.finite_T else self._beta_max
        mu = self.mu if self.finite_T else None

        # get time-grid
        ng = self.ngrid
        ti = self.ti
        G = self.G
        g = self.g

        if not self.finite_T:
            # create energies in spin-orbital basis
            eo,ev = self.sys.g_energies()
            no = eo.shape[0]
            nv = ev.shape[0]
            D1 = (ev[:,None] - eo[None,:])
            D2 = (ev[:,None,None,None] + ev[None,:,None,None]
                - eo[None,None,:,None] - eo[None,None,None,:])

            # get HF energy
            En = self.sys.const_energy()
            E0 = zt_mp.mp0(eo) + En
            E1 = self.sys.get_mp1()
            E01 = E0 + E1

            # get Fock matrix
            F = self.sys.g_fock()
            F.oo = F.oo - numpy.diag(eo) # subtract diagonal
            F.vv = F.vv - numpy.diag(ev) # subtract diagonal

            # get ERIs
            I = self.sys.g_aint()
        else:
            # get orbital energies
            en = self.sys.g_energies_tot()
            fo = ft_utils.ff(beta, en, mu)
            fv = ft_utils.ffv(beta, en, mu)
            n = en.shape[0]

            # get 0th and 1st order contributions
            En = self.sys.const_energy()
            g0 = ft_utils.GP0(beta, en, mu)
            E0 = ft_mp.mp0(g0) + En
            E1 = self.sys.get_mp1()
            E01 = E0 + E1
            if self.athresh > 0.0:
                athresh = self.athresh
                focc = [x for x in fo if x > athresh]
                fvir = [x for x in fv if x > athresh]
                iocc = [i for i,x in enumerate(fo) if x > athresh]
                ivir = [i for i,x in enumerate(fv) if x > athresh]
                nocc = len(focc)
                nvir = len(fvir)
                nact = nocc + nvir - n
                ncor = nocc - nact
                nvvv = nvir - nact
                if self.iprint > 0:
                    print("FT-CCSD orbital info:")
                    print('  nocc: {:d}'.format(nocc))
                    print('  nvir: {:d}'.format(nvir))
                    print('  nact: {:d}'.format(nact))

                # get scaled active space integrals
                F,I = cc_utils.ft_active_integrals(self.sys, en, focc, fvir, iocc, ivir)

                # get exponentials
                D1 = en[:,None] - en[None,:]
                D2 = en[:,None,None,None] + en[None,:,None,None] \
                        - en[None,None,:,None] - en[None,None,None,:]
                D1 = D1[numpy.ix_(ivir,iocc)]
                D2 = D2[numpy.ix_(ivir,ivir,iocc,iocc)]

            else:
                # get scaled integrals
                F,I = cc_utils.ft_integrals(self.sys, en, beta, mu)

                # get energy differences
                D1 = en[:,None] - en[None,:]
                D2 = en[:,None,None,None] + en[None,:,None,None] \
                        - en[None,None,:,None] - en[None,None,None,:]

        method = "CCSD" if self.singles else "CCD"
        conv_options = {
                "econv":self.econv,
                "tconv":self.tconv,
                "max_iter":self.max_iter,
                "damp":self.damp}
        if self.rt_iter[0] == 'a' or T2in is not None:
            if self.rt_iter[0] != 'a' and iprint > 0:
                print("WARNING: Converngece scheme ({}) is being ignored.".format(self.rt_iter))
            # get MP2 T-amplitudes
            if T1in is not None and T2in is not None:
                T1old = T1in if self.singles else numpy.zeros((ng,n,n))
                T2old = T2in
            else:
                if self.singles:
                    Id = numpy.ones((ng))
                    T1old = -einsum('v,ai->vai',Id,F.vo)
                else:
                    T1old = numpy.zeros((ng,n,n))
                Id = numpy.ones((ng))
                T2old = -einsum('v,abij->vabij',Id,I.vvoo)
                T1old = quadrature.int_tbar1(ng,T1old,ti,D1,G)
                T2old = quadrature.int_tbar2(ng,T2old,ti,D2,G)
            E2 = ft_cc_energy.ft_cc_energy(T1old,T2old,
                F.ov,I.oovv,g,beta,Qterm=False)
            if self.iprint > 0:
                print('MP2 Energy: {:.10f}'.format(E2))

            # run CC iterations
            Eccn,T1,T2 = cc_utils.ft_cc_iter(method, T1old, T2old, F, I, D1, D2, g, G,
                    beta, ng, ti, self.iprint, conv_options)
        else:
            T1,T2 = cc_utils.ft_cc_iter_extrap(method, F, I, D1, D2, g, G, beta, ng, ti,
                    self.iprint, conv_options)
            Eccn = ft_cc_energy.ft_cc_energy(T1,T2,
                F.ov,I.oovv,g,beta)

        # save T amplitudes
        self.T1 = T1
        self.T2 = T2
        self.G0 = E0
        self.G1 = E1
        self.Gcc = Eccn
        self.Gtot = E0 + E1 + Eccn

        return (Eccn+E01,Eccn)

    def _ft_uccsd(self,T1in=None,T2in=None):
        """Solve finite temperature coupled cluster equations."""
        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu

        # get time-grid
        ng = self.ngrid
        ti = self.ti
        G = self.G
        g = self.g

        # get orbital energies
        ea,eb = self.sys.u_energies_tot()
        na = ea.shape[0]
        nb = eb.shape[0]

        # get 0th and 1st order contributions
        En = self.sys.const_energy()
        g0 = ft_utils.uGP0(beta, ea, eb, mu)
        E0 = ft_mp.ump0(g0[0],g0[1]) + En
        E1 = self.sys.get_mp1()
        E01 = E0 + E1

        if self.athresh > 0.0:
            athresh = self.athresh
            foa = ft_utils.ff(beta, ea, mu)
            fva = ft_utils.ffv(beta, ea, mu)
            fob = ft_utils.ff(beta, eb, mu)
            fvb = ft_utils.ffv(beta, eb, mu)
            focca = [x for x in foa if x > athresh]
            fvira = [x for x in fva if x > athresh]
            iocca = [i for i,x in enumerate(foa) if x > athresh]
            ivira = [i for i,x in enumerate(fva) if x > athresh]
            foccb = [x for x in fob if x > athresh]
            fvirb = [x for x in fvb if x > athresh]
            ioccb = [i for i,x in enumerate(fob) if x > athresh]
            ivirb = [i for i,x in enumerate(fvb) if x > athresh]
            nocca = len(focca)
            nvira = len(fvira)
            noccb = len(foccb)
            nvirb = len(fvirb)
            nacta = nocca + nvira - na
            nactb = noccb + nvirb - nb
            if self.iprint > 0:
                print("FT-UCCSD orbital info:")
                print('  nocca: {:d}'.format(nocca))
                print('  nvira: {:d}'.format(nvira))
                print('  nacta: {:d}'.format(nacta))
                print('  noccb: {:d}'.format(nocca))
                print('  nvirb: {:d}'.format(nvira))
                print('  nactb: {:d}'.format(nacta))

            # get energy differences
            D1a = ea[:,None] - ea[None,:]
            D1b = eb[:,None] - eb[None,:]
            D2aa = ea[:,None,None,None] + ea[None,:,None,None] \
                    - ea[None,None,:,None] - ea[None,None,None,:]
            D2ab = ea[:,None,None,None] + eb[None,:,None,None] \
                    - ea[None,None,:,None] - eb[None,None,None,:]
            D2bb = eb[:,None,None,None] + eb[None,:,None,None] \
                    - eb[None,None,:,None] - eb[None,None,None,:]
            D1a = D1a[numpy.ix_(ivira,iocca)]
            D1b = D1b[numpy.ix_(ivirb,ioccb)]
            D2aa = D2aa[numpy.ix_(ivira,ivira,iocca,iocca)]
            D2ab = D2ab[numpy.ix_(ivira,ivirb,iocca,ioccb)]
            D2bb = D2bb[numpy.ix_(ivirb,ivirb,ioccb,ioccb)]

            # get scaled integrals
            Fa,Fb,Ia,Ib,Iabab = cc_utils.uft_active_integrals(
                    self.sys, ea, eb, focca, fvira, foccb, fvirb, iocca, ivira, ioccb, ivirb)

            T1ashape = (ng,nvira,nocca)
            T2bshape = (ng,nvirb,noccb)

        else:
            # get scaled integrals
            Fa,Fb,Ia,Ib,Iabab = cc_utils.uft_integrals(self.sys, ea, eb, beta, mu)

            # get energy differences
            D1a = ea[:,None] - ea[None,:]
            D1b = eb[:,None] - eb[None,:]
            D2aa = ea[:,None,None,None] + ea[None,:,None,None] \
                    - ea[None,None,:,None] - ea[None,None,None,:]
            D2ab = ea[:,None,None,None] + eb[None,:,None,None] \
                    - ea[None,None,:,None] - eb[None,None,None,:]
            D2bb = eb[:,None,None,None] + eb[None,:,None,None] \
                    - eb[None,None,:,None] - eb[None,None,None,:]
            T1ashape = (ng,na,na)
            T2bshape = (ng,nb,nb)

        method = "CCSD" if self.singles else "CCD"
        conv_options = {
                "econv":self.econv,
                "tconv":self.tconv,
                "max_iter":self.max_iter,
                "damp":self.damp}
        if self.rt_iter[0] == 'a' or T2in is not None:
            if self.rt_iter[0] != 'a':
                print("WARNING: Converngece scheme ({}) is being ignored.".format(self.rt_iter))
            # get MP2 T-amplitudes
            if T1in is not None and T2in is not None:
                T1aold = T1in[0] if self.singles else numpy.zeros(T1ashape)
                T1bold = T1in[1] if self.singles else numpy.zeros(T1bshape)
                T2aaold = T2in[0]
                T2abold = T2in[1]
                T2bbold = T2in[2]
            else:
                if self.singles:
                    Id = numpy.ones((ng))
                    T1aold = -einsum('v,ai->vai',Id,Fa.vo)
                    T1bold = -einsum('v,ai->vai',Id,Fb.vo)
                else:
                    T1old = numpy.zeros((ng,n,n))
                Id = numpy.ones((ng))
                T2aaold = -einsum('v,abij->vabij',Id,Ia.vvoo)
                T2abold = -einsum('v,abij->vabij',Id,Iabab.vvoo)
                T2bbold = -einsum('v,abij->vabij',Id,Ib.vvoo)
                T1aold = quadrature.int_tbar1(ng,T1aold,ti,D1a,G)
                T1bold = quadrature.int_tbar1(ng,T1bold,ti,D1b,G)
                T2aaold = quadrature.int_tbar2(ng,T2aaold,ti,D2aa,G)
                T2abold = quadrature.int_tbar2(ng,T2abold,ti,D2ab,G)
                T2bbold = quadrature.int_tbar2(ng,T2bbold,ti,D2bb,G)

            # MP2 energy
            E2 = ft_cc_energy.ft_ucc_energy(T1aold,T1bold,T2aaold,T2abold,T2bbold,
                Fa.ov,Fb.ov,Ia.oovv,Ib.oovv,Iabab.oovv,g,beta,Qterm=False)
            if self.iprint > 0:
                print('MP2 Energy: {:.10f}'.format(E2))


            # run CC iterations
            Eccn,T1,T2 = cc_utils.ft_ucc_iter(method, T1aold, T1bold, T2aaold, T2abold, T2bbold,
                    Fa, Fb, Ia, Ib, Iabab, D1a, D1b, D2aa, D2ab, D2bb,
                    g, G, beta, ng, ti, self.iprint, conv_options)
        else:
            T1,T2 = cc_utils.ft_ucc_iter_extrap(method, Fa, Fb, Ia, Ib, Iabab, D1a, D1b, D2aa, D2ab, D2bb,
                    g, G, beta, ng, ti, self.iprint, conv_options)
            Eccn = ft_cc_energy.ft_ucc_energy(T1[0], T1[1], T2[0], T2[1], T2[2],
                Fa.ov,Fb.ov,Ia.oovv,Ib.oovv,Iabab.oovv,g,beta)

        # save T amplitudes
        self.T1 = T1
        self.T2 = T2
        self.G0 = E0
        self.G1 = E1
        self.Gcc = Eccn
        self.Gtot = E0 + E1 + Eccn

        return (Eccn+E01,Eccn)

    def _ft_ccsd_lambda(self, L1=None, L2=None):
        """Solve FT-CCSD Lambda equations."""
        #T = self.T
        beta = 1.0 / (self.T + 1e-12)
        mu = self.mu

        # get time-grid
        ng = self.ngrid
        ti = self.ti
        G = self.G
        g = self.g

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        n = en.shape[0]
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)

        En = self.sys.const_energy()
        g0 = ft_utils.GP0(beta, en, mu)
        E0 = ft_mp.mp0(g0) + En

        # get HF free energies  
        E1 = self.sys.get_mp1()
        E01 = E0 + E1
        
        if self.athresh > 0.0:
            athresh = self.athresh
            focc = [x for x in fo if x > athresh]
            fvir = [x for x in fv if x > athresh]
            iocc = [i for i,x in enumerate(fo) if x > athresh]
            ivir = [i for i,x in enumerate(fv) if x > athresh]
            nocc = len(focc)
            nvir = len(fvir)
            nact = nocc + nvir - n
            ncor = nocc - nact
            nvvv = nvir - nact
            if self.iprint > 0:
                print("FT-CCSD orbital info:")
                print('  nocc: {:d}'.format(nocc))
                print('  nvir: {:d}'.format(nvir))
                print('  nact: {:d}'.format(nact))

            # get scaled active space integrals
            F,I = cc_utils.ft_active_integrals(self.sys, en, focc, fvir, iocc, ivir)

            # get exponentials
            D1 = en[:,None] - en[None,:]
            D2 = en[:,None,None,None] + en[None,:,None,None] \
                    - en[None,None,:,None] - en[None,None,None,:]
            D1 = D1[numpy.ix_(ivir,iocc)]
            D2 = D2[numpy.ix_(ivir,ivir,iocc,iocc)]
        else:
            # get scaled integrals
            F,I = cc_utils.ft_integrals(self.sys, en, beta, mu)

            # get energy differences
            D1 = en[:,None] - en[None,:]
            D2 = en[:,None,None,None] + en[None,:,None,None] \
                    - en[None,None,:,None] - en[None,None,None,:]

        if L2 is None and L1 is None:
            if self.singles:
                L1old,L2old = ft_cc_equations.ccsd_lambda_guess(F,I,self.T1,beta,ng)
            else:
                L2old = ft_cc_equations.ccd_lambda_guess(I,beta,ng)
        elif L2 is not None and L1 is None:
            L2old = L2
            if self.singles:
                ng,nv,no = self.T1.shape
                L1old = numpy.zeros((ng,no,nv))
        elif L1 is not None and L2 is None:
            ng,nv,no = self.T1.shape
            L1old = L1
            L2old = numpy.zeros((ng,no,nv))
            if not self.singles:
                raise Exception("Singles guess provided to FT-CCD Lambda equations")
        else:
            assert(L1 is not None and L2 is not None)
            L1old = L1
            L2old = L2

        # run lambda iterations
        conv_options = {
                "econv":self.econv,
                "tconv":self.tconv,
                "max_iter":self.max_iter,
                "damp":self.damp}
        method = "CCSD" if self.singles else "CCD"
        L1,L2 = cc_utils.ft_lambda_iter(method, L1old, L2old, self.T1, self.T2, F, I,
                D1, D2, g, G, beta, ng, ti, self.iprint, conv_options)

        # save lambda amplitudes
        self.L1 = L1
        self.L2 = L2

    def _ft_uccsd_lambda(self, L1=None, L2=None):
        """Solve FT-CCSD Lambda equations."""
        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu

        # get time-grid
        ng = self.ngrid
        ti = self.ti
        G = self.G
        g = self.g

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        ea,eb = self.sys.u_energies_tot()
        na = ea.shape[0]
        nb = eb.shape[0]

        En = self.sys.const_energy()
        g0 = ft_utils.uGP0(beta, ea, eb, mu)
        E0 = ft_mp.ump0(g0[0],g0[1]) + En
        E1 = self.sys.get_mp1()
        E01 = E0 + E1

        if self.athresh > 0.0:
            athresh = self.athresh
            foa = ft_utils.ff(beta, ea, mu)
            fva = ft_utils.ffv(beta, ea, mu)
            fob = ft_utils.ff(beta, eb, mu)
            fvb = ft_utils.ffv(beta, eb, mu)
            focca = [x for x in foa if x > athresh]
            fvira = [x for x in fva if x > athresh]
            iocca = [i for i,x in enumerate(foa) if x > athresh]
            ivira = [i for i,x in enumerate(fva) if x > athresh]
            foccb = [x for x in fob if x > athresh]
            fvirb = [x for x in fvb if x > athresh]
            ioccb = [i for i,x in enumerate(fob) if x > athresh]
            ivirb = [i for i,x in enumerate(fvb) if x > athresh]
            nocca = len(focca)
            nvira = len(fvira)
            noccb = len(foccb)
            nvirb = len(fvirb)
            nacta = nocca + nvira - na
            nactb = noccb + nvirb - nb
            if self.iprint > 0:
                print("FT-UCCSD orbital info:")
                print('  nocca: {:d}'.format(nocca))
                print('  nvira: {:d}'.format(nvira))
                print('  nacta: {:d}'.format(nacta))
                print('  noccb: {:d}'.format(nocca))
                print('  nvirb: {:d}'.format(nvira))
                print('  nactb: {:d}'.format(nacta))

            # get energy differences
            D1a = ea[:,None] - ea[None,:]
            D1b = eb[:,None] - eb[None,:]
            D2aa = ea[:,None,None,None] + ea[None,:,None,None] \
                    - ea[None,None,:,None] - ea[None,None,None,:]
            D2ab = ea[:,None,None,None] + eb[None,:,None,None] \
                    - ea[None,None,:,None] - eb[None,None,None,:]
            D2bb = eb[:,None,None,None] + eb[None,:,None,None] \
                    - eb[None,None,:,None] - eb[None,None,None,:]
            D1a = D1a[numpy.ix_(ivira,iocca)]
            D1b = D1b[numpy.ix_(ivirb,ioccb)]
            D2aa = D2aa[numpy.ix_(ivira,ivira,iocca,iocca)]
            D2ab = D2ab[numpy.ix_(ivira,ivirb,iocca,ioccb)]
            D2bb = D2bb[numpy.ix_(ivirb,ivirb,ioccb,ioccb)]

            # get scaled integrals
            Fa,Fb,Ia,Ib,Iabab = cc_utils.uft_active_integrals(
                    self.sys, ea, eb, focca, fvira, foccb, fvirb, iocca, ivira, ioccb, ivirb)

            T1ashape = (ng,nvira,nocca)
            T2bshape = (ng,nvirb,noccb)

        else:
            # get scaled integrals
            Fa,Fb,Ia,Ib,Iabab = cc_utils.uft_integrals(self.sys, ea, eb, beta, mu)

            # get energy differences
            D1a = ea[:,None] - ea[None,:]
            D1b = eb[:,None] - eb[None,:]
            D2aa = ea[:,None,None,None] + ea[None,:,None,None] \
                    - ea[None,None,:,None] - ea[None,None,None,:]
            D2ab = ea[:,None,None,None] + eb[None,:,None,None] \
                    - ea[None,None,:,None] - eb[None,None,None,:]
            D2bb = eb[:,None,None,None] + eb[None,:,None,None] \
                    - eb[None,None,:,None] - eb[None,None,None,:]

        T1aold,T1bold = self.T1
        T2aaold,T2abold,T2bbold = self.T2
        if L2 is None and L1 is None:
            if self.singles:
                L1aold,L1bold,L2aaold,L2abold,L2bbold = ft_cc_equations.uccsd_lambda_guess(
                    Fa,Fb,Ia,Ib,Iabab,self.T1[0],self.T1[1],beta,ng)
            else:
                L2aaold,L2abold,L2bbold = ft_cc_equations.uccd_lambda_guess(Ia,Ib,Iabab,beta,ng)
        elif L2 is not None and L1 is None:
            L2aaold = L2aa
            L2abold = L2ab
            L2bbold = L2bb
            if self.singles:
                ng,nv,no = self.T1.shape
                L1aold = numpy.zeros((ng,no,nv))
                L1bold = numpy.zeros((ng,no,nv))
        elif L1 is not None and L2 is None:
            ng,nv,no = self.T1.shape
            L1aold = L1[0]
            L1bold = L1[1]
            L2aaold = numpy.zeros((ng,no,nv))
            L2abold = numpy.zeros((ng,no,nv))
            L2bbold = numpy.zeros((ng,no,nv))
            if not self.singles:
                raise Exception("Singles guess provided to FT-CCD Lambda equations")
        else:
            assert(L1 is not None and L2 is not None)
            L1aold = L1[0]
            L1bold = L1[1]
            L2aaold = L2[0]
            L2abold = L2[1]
            L2bbold = L2[2]

        # run lambda iterations
        conv_options = {
                "econv":self.econv,
                "tconv":self.tconv,
                "max_iter":self.max_iter,
                "damp":self.damp}
        method = "CCSD" if self.singles else "CCD"
        L1a,L1b,L2aa,L2ab,L2bb = cc_utils.ft_ulambda_iter(
                method, L1aold, L1bold, L2aaold, L2abold, L2bbold, T1aold, T1bold, 
                T2aaold, T2abold, T2bbold, Fa, Fb, Ia, Ib, Iabab, D1a, D1b, D2aa, D2ab, D2bb, 
                g, G, beta, ng, ti, self.iprint, conv_options)

        # save lambda amplitudes
        self.L1 = (L1a,L1b)
        self.L2 = (L2aa,L2ab,L2bb)

    def _g_nocc_deriv(self, dvec):
        """Evaluate the Lagrangian with scaled occupation number derivatives."""
        # temperature info
        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)
        #n = fo.shape[0]

        # first order contributions
        der1 = self.sys.g_d_mp1(dvec)

        # get time-grid
        ng = self.ngrid
        ti = self.ti
        G = self.G
        g = self.g

        if self.athresh > 0.0:
            athresh = self.athresh
            focc = [x for x in fo if x > athresh]
            fvir = [x for x in fv if x > athresh]
            iocc = [i for i,x in enumerate(fo) if x > athresh]
            ivir = [i for i,x in enumerate(fv) if x > athresh]
            F,I = cc_utils.ft_d_active_integrals(
                    self.sys, en, fo, fv, iocc, ivir, dvec)
        else:
            F,I = cc_utils.ft_d_integrals(self.sys, en, fo, fv, dvec)
        A1 = (1.0/beta)*einsum('ia,ai->',self.dia,F.vo)
        A1 += (1.0/beta)*einsum('ba,ab->',self.dba,F.vv)
        A1 += (1.0/beta)*einsum('ji,ij->',self.dji,F.oo)
        A1 += (1.0/beta)*einsum('ai,ia->',self.dai,F.ov)
        A2 = (0.25/beta)*einsum('cdab,abcd->',self.P2[0],I.vvvv)
        A2 += (0.5/beta)*einsum('ciab,abci->',self.P2[1],I.vvvo)
        A2 += (0.5/beta)*einsum('bcai,aibc->',self.P2[2],I.vovv)
        A2 += (0.25/beta)*einsum('ijab,abij->',self.P2[3],I.vvoo)
        A2 += (1.0/beta)*einsum('bjai,aibj->',self.P2[4],I.vovo)
        A2 += (0.25/beta)*einsum('abij,ijab->',self.P2[5],I.oovv)
        A2 += (0.5/beta)*einsum('jkai,aijk->',self.P2[6],I.vooo)
        A2 += (0.5/beta)*einsum('kaij,ijka->',self.P2[7],I.ooov)
        A2 += (0.25/beta)*einsum('klij,ijkl->',self.P2[8],I.oooo)
        der_cc = A1 + A2

        return der1,der_cc

    def _u_nocc_deriv(self, dveca, dvecb):
        """Evaluate the Lagrangian with scaled occupation number derivatives."""
        dvec = numpy.concatenate((dveca,dvecb))
        # temperature info
        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu

        # get energies and occupation numbers
        ea,eb = self.sys.u_energies_tot()
        na = ea.shape[0]
        nb = eb.shape[0]
        n = na + nb
        foa = ft_utils.ff(beta, ea, mu)
        fob = ft_utils.ff(beta, eb, mu)
        fva = ft_utils.ffv(beta, ea, mu)
        fvb = ft_utils.ffv(beta, eb, mu)

        # first order contributions
        der1 = self.sys.u_d_mp1(dveca,dvecb)

        # get time-grid
        ng = self.ngrid
        ti = self.ti
        G = self.G
        g = self.g

        # get exponentials
        D1a = ea[:,None] - ea[None,:]
        D1b = eb[:,None] - eb[None,:]
        D2aa = ea[:,None,None,None] + ea[None,:,None,None] \
            - ea[None,None,:,None] - ea[None,None,None,:]
        D2ab = ea[:,None,None,None] + eb[None,:,None,None] \
            - ea[None,None,:,None] - eb[None,None,None,:]
        D2bb = eb[:,None,None,None] + eb[None,:,None,None] \
            - eb[None,None,:,None] - eb[None,None,None,:]
        if self.athresh > 0.0:
            athresh = self.athresh
            foa = ft_utils.ff(beta, ea, mu)
            fva = ft_utils.ffv(beta, ea, mu)
            fob = ft_utils.ff(beta, eb, mu)
            fvb = ft_utils.ffv(beta, eb, mu)
            focca = [x for x in foa if x > athresh]
            fvira = [x for x in fva if x > athresh]
            iocca = [i for i,x in enumerate(foa) if x > athresh]
            ivira = [i for i,x in enumerate(fva) if x > athresh]
            foccb = [x for x in fob if x > athresh]
            fvirb = [x for x in fvb if x > athresh]
            ioccb = [i for i,x in enumerate(fob) if x > athresh]
            ivirb = [i for i,x in enumerate(fvb) if x > athresh]
            D1a = D1a[numpy.ix_(ivira,iocca)]
            D1b = D1b[numpy.ix_(ivirb,ioccb)]
            D2aa = D2aa[numpy.ix_(ivira,ivira,iocca,iocca)]
            D2ab = D2ab[numpy.ix_(ivira,ivirb,iocca,ioccb)]
            Fa,Fb,Ia,Ib,Iabab = cc_utils.uft_d_active_integrals(
                    self.sys, ea, eb, focca, fvira, foccb, fvirb, 
                    iocca, ivira, ioccb, ivirb, dveca, dvecb)
        else:
            Fa,Fb,Ia,Ib,Iabab = cc_utils.u_ft_d_integrals(self.sys, ea, eb, foa, fva, fob, fvb, dveca, dvecb)
        A1 = (1.0/beta)*einsum('ia,ai->', self.dia[0], Fa.vo)
        A1 += (1.0/beta)*einsum('ia,ai->', self.dia[1], Fb.vo)
        A1 += (1.0/beta)*einsum('ba,ab->', self.dba[0], Fa.vv)
        A1 += (1.0/beta)*einsum('ba,ab->', self.dba[1], Fb.vv)
        A1 += (1.0/beta)*einsum('ji,ij->', self.dji[0], Fa.oo)
        A1 += (1.0/beta)*einsum('ji,ij->', self.dji[1], Fb.oo)
        A1 += (1.0/beta)*einsum('ai,ia->', self.dai[0], Fa.ov)
        A1 += (1.0/beta)*einsum('ai,ia->', self.dai[1], Fb.ov)

        A2 = (0.25/beta)*einsum('ijab,abij->', self.P2[3][0], Ia.vvoo)
        A2 += (0.25/beta)*einsum('ijab,abij->', self.P2[3][1], Ib.vvoo)
        A2 += (1.0/beta)*einsum('ijab,abij->', self.P2[3][2], Iabab.vvoo)

        A2 += (0.5/beta)*einsum('ciab,abci->', self.P2[1][0], Ia.vvvo)
        A2 += (0.5/beta)*einsum('ciab,abci->', self.P2[1][1], Ib.vvvo)
        A2 += (1.0/beta)*einsum('ciab,abci->', self.P2[1][2], Iabab.vvvo)
        A2 += (1.0/beta)*einsum('ciab,baic->', self.P2[1][3], Iabab.vvov)

        A2 += (0.5/beta)*einsum('jkai,aijk->', self.P2[6][0], Ia.vooo)
        A2 += (0.5/beta)*einsum('jkai,aijk->', self.P2[6][1], Ib.vooo)
        A2 += (1.0/beta)*einsum('jKaI,aIjK->', self.P2[6][2], Iabab.vooo)
        A2 += (1.0/beta)*einsum('JkAi,iAkJ->', self.P2[6][3], Iabab.ovoo)

        A2 += (0.25/beta)*einsum('cdab,abcd->', self.P2[0][0], Ia.vvvv)
        A2 += (0.25/beta)*einsum('cdab,abcd->', self.P2[0][1], Ib.vvvv)
        A2 += (1.0/beta)*einsum('cdab,abcd->', self.P2[0][2], Iabab.vvvv)

        A2 += (1.0/beta)*einsum('bjai,aibj->', self.P2[4][0], Ia.vovo)
        A2 += (1.0/beta)*einsum('BJAI,AIBJ->', self.P2[4][1], Ib.vovo)
        A2 += (1.0/beta)*einsum('bJaI,aIbJ->', self.P2[4][2], Iabab.vovo)
        A2 -= (1.0/beta)*einsum('bJAi,iAbJ->', self.P2[4][3], Iabab.ovvo)
        A2 -= (1.0/beta)*einsum('BjaI,aIjB->', self.P2[4][4], Iabab.voov)
        A2 += (1.0/beta)*einsum('BjAi,iAjB->', self.P2[4][5], Iabab.ovov)

        A2 += (0.25/beta)*einsum('klij,ijkl->', self.P2[8][0], Ia.oooo)
        A2 += (0.25/beta)*einsum('klij,ijkl->', self.P2[8][1], Ib.oooo)
        A2 += (1.0/beta)*einsum('kLiJ,iJkL->', self.P2[8][2], Iabab.oooo)

        A2 += (0.5/beta)*einsum('bcai,aibc->', self.P2[2][0], Ia.vovv)
        A2 += (0.5/beta)*einsum('bcai,aibc->', self.P2[2][1], Ib.vovv)
        A2 += (1.0/beta)*einsum('bCaI,aIbC->', self.P2[2][2], Iabab.vovv)
        A2 += (1.0/beta)*einsum('BcAi,iAcB->', self.P2[2][3], Iabab.ovvv)

        A2 += (0.5/beta)*einsum('kaij,ijka->', self.P2[7][0], Ia.ooov)
        A2 += (0.5/beta)*einsum('kaij,ijka->', self.P2[7][1], Ib.ooov)
        A2 += (1.0/beta)*einsum('kaij,ijka->', self.P2[7][2], Iabab.ooov)
        A2 += (1.0/beta)*einsum('KaIj,jIaK->', self.P2[7][3], Iabab.oovo)

        A2 += (0.25/beta)*einsum('abij,ijab->', self.P2[5][0], Ia.oovv)
        A2 += (0.25/beta)*einsum('abij,ijab->', self.P2[5][1], Ib.oovv)
        A2 += (1.0/beta)*einsum('aBiJ,iJaB->', self.P2[5][2], Iabab.oovv)
        der_cc = A1 + A2

        return der1,der_cc

    def _g_nocc_gderiv(self):
        """Evaluate the derivatives of the weight matrices in 
        the time-dependent formulation.
        """
        # temperature info
        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)

        # get time-grid
        ng = self.ngrid
        ti = self.ti
        G = self.G
        g = self.g
        gd,Gd = quadrature.d_ft_quad(ng, beta, self.quad)

        # get exponentials
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
            - en[None,None,:,None] - en[None,None,None,:]
        if self.athresh > 0.0:
            athresh = self.athresh
            focc = [x for x in fo if x > athresh]
            fvir = [x for x in fv if x > athresh]
            iocc = [i for i,x in enumerate(fo) if x > athresh]
            ivir = [i for i,x in enumerate(fv) if x > athresh]
            D1 = D1[numpy.ix_(ivir,iocc)]
            D2 = D2[numpy.ix_(ivir,ivir,iocc,iocc)]
            F,I = cc_utils.ft_active_integrals(
                    self.sys, en, focc, fvir, iocc, ivir)
        else:
            F,I = cc_utils.ft_integrals(self.sys, en, beta, mu)

        A1 = (1.0/beta)*einsum('ia,ai->',self.dia,F.vo)
        # get derivative with respect to g
        Eterm = ft_cc_energy.ft_cc_energy(self.T1,self.T2,
            F.ov,I.oovv,gd,beta)

        dg = Eterm

        # get derivative with respect to G
        T1temp,T2temp = ft_cc_equations.ccsd_stanton(F,I,self.T1,self.T2,
                D1,D2,ti,ng,Gd)

        A1 = (1.0/beta)*einsum('via,vai->v',self.L1, T1temp)
        A2 = (1.0/beta)*0.25*einsum('vijab,vabij->v',self.L2, T2temp)

        A1g = einsum('v,v->',A1,g)
        A2g = einsum('v,v->',A2,g)
        dG = A1g + A2g

        # append derivative with respect to ti points
        Gnew = G.copy()
        m = G.shape[0]
        n = G.shape[0]
        for i in range(m):
            for j in range(n):
                Gnew[i,j] *= (self.ti[j] - self.ti[i])/beta
 
        T1temp,T2temp = ft_cc_equations.ccsd_stanton(F,I,self.T1,self.T2,
                D1,D2,ti,ng,Gnew)
        T1temp *= D1
        T2temp *= D2

        A1 = (1.0/beta)*einsum('via,vai->v',self.L1, T1temp)
        A2 = (1.0/beta)*0.25*einsum('vijab,vabij->v',self.L2, T2temp)

        A1g = einsum('v,v->',A1,g)
        A2g = einsum('v,v->',A2,g)
        dG += A1g + A2g

        return dg,dG

    def _u_nocc_gderiv(self):
        """Evaluate the derivatives of the weight matrices in
        the time-dependent formulation.
        """
        # temperature info
        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        ea,eb = self.sys.u_energies_tot()
        na = ea.shape[0]
        nb = eb.shape[0]

        # get time-grid
        ng = self.ngrid
        ti = self.ti
        G = self.G
        g = self.g
        gd,Gd = quadrature.d_ft_quad(ng, beta, self.quad)

        # get energy differences
        D1a = ea[:,None] - ea[None,:]
        D1b = eb[:,None] - eb[None,:]
        D2aa = ea[:,None,None,None] + ea[None,:,None,None] \
            - ea[None,None,:,None] - ea[None,None,None,:]
        D2ab = ea[:,None,None,None] + eb[None,:,None,None] \
            - ea[None,None,:,None] - eb[None,None,None,:]
        D2bb = eb[:,None,None,None] + eb[None,:,None,None] \
            - eb[None,None,:,None] - eb[None,None,None,:]
        if self.athresh > 0.0:
            athresh = self.athresh
            foa = ft_utils.ff(beta, ea, mu)
            fva = ft_utils.ffv(beta, ea, mu)
            fob = ft_utils.ff(beta, eb, mu)
            fvb = ft_utils.ffv(beta, eb, mu)
            focca = [x for x in foa if x > athresh]
            fvira = [x for x in fva if x > athresh]
            iocca = [i for i,x in enumerate(foa) if x > athresh]
            ivira = [i for i,x in enumerate(fva) if x > athresh]
            foccb = [x for x in fob if x > athresh]
            fvirb = [x for x in fvb if x > athresh]
            ioccb = [i for i,x in enumerate(fob) if x > athresh]
            ivirb = [i for i,x in enumerate(fvb) if x > athresh]
            D1a = D1a[numpy.ix_(ivira,iocca)]
            D1b = D1b[numpy.ix_(ivirb,ioccb)]
            D2aa = D2aa[numpy.ix_(ivira,ivira,iocca,iocca)]
            D2ab = D2ab[numpy.ix_(ivira,ivirb,iocca,ioccb)]
            Fa,Fb,Ia,Ib,Iabab = cc_utils.uft_active_integrals(
                    self.sys, ea, eb, focca, fvira, foccb, fvirb, iocca, ivira, ioccb, ivirb)
        else:
            Fa,Fb,Ia,Ib,Iabab = cc_utils.uft_integrals(self.sys, ea, eb, beta, mu)
        T1aold,T1bold = self.T1
        T2aaold,T2abold,T2bbold = self.T2
        L1aold,L1bold = self.L1
        L2aaold,L2abold,L2bbold = self.L2

        # get derivative with respect to g
        Eterm = ft_cc_energy.ft_ucc_energy(T1aold,T1bold,T2aaold,T2abold,T2bbold,
            Fa.ov,Fb.ov,Ia.oovv,Ib.oovv,Iabab.oovv,gd,beta)

        dg = Eterm

        # get derivative with respect to G
        T1t,T2t = ft_cc_equations.uccsd_stanton(Fa,Fb,Ia,Ib,Iabab,T1aold,T1bold,
                T2aaold,T2abold,T2bbold,D1a,D1b,D2aa,D2ab,D2bb,ti,ng,Gd)

        A1a = (1.0/beta)*einsum('via,vai->v',L1aold, T1t[0])
        A1b = (1.0/beta)*einsum('via,vai->v',L1bold, T1t[1])
        A2a = (1.0/beta)*0.25*einsum('vijab,vabij->v',L2aaold, T2t[0])
        A2b = (1.0/beta)*0.25*einsum('vijab,vabij->v',L2bbold, T2t[2])
        A2ab = (1.0/beta)*einsum('vijab,vabij->v',L2abold, T2t[1])

        A2g = einsum('v,v->',A2a,g)
        A2g += einsum('v,v->',A2ab,g)
        A2g += einsum('v,v->',A2b,g)
        A1g = einsum('v,v->',A1a,g)
        A1g += einsum('v,v->',A1b,g)
        dG = A1g + A2g

        # append derivative with respect to ti points
        Gnew = G.copy()
        m = G.shape[0]
        n = G.shape[0]
        for i in range(m):
            for j in range(n):
                Gnew[i,j] *= (self.ti[j] - self.ti[i])/beta

        T1t,T2t = ft_cc_equations.uccsd_stanton(Fa,Fb,Ia,Ib,Iabab,T1aold,T1bold,
                T2aaold,T2abold,T2bbold,D1a,D1b,D2aa,D2ab,D2bb,ti,ng,Gnew)
        T1a,T1b = T1t
        T2aa,T2ab,T2bb = T2t
        T1a *= D1a
        T1b *= D1b
        T2aa *= D2aa
        T2ab *= D2ab
        T2bb *= D2bb

        A1a = (1.0/beta)*einsum('via,vai->v',L1aold, T1a)
        A1b = (1.0/beta)*einsum('via,vai->v',L1bold, T1b)
        A2a = (1.0/beta)*0.25*einsum('vijab,vabij->v',L2aaold, T2aa)
        A2b = (1.0/beta)*0.25*einsum('vijab,vabij->v',L2bbold, T2bb)
        A2ab = (1.0/beta)*einsum('vijab,vabij->v',L2abold, T2ab)

        A2g = einsum('v,v->',A2a,g)
        A2g += einsum('v,v->',A2ab,g)
        A2g += einsum('v,v->',A2b,g)
        A1g = einsum('v,v->',A1a,g)
        A1g += einsum('v,v->',A1b,g)
        dG += A1g + A2g

        return dg,dG

    def _g_gderiv_approx(self):
        # temperature info
        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)

        ng = self.ngrid
        tf = ng - 1
        if self.athresh > 0.0:
            athresh = self.athresh
            focc = [x for x in fo if x > athresh]
            fvir = [x for x in fv if x > athresh]
            iocc = [i for i,x in enumerate(fo) if x > athresh]
            ivir = [i for i,x in enumerate(fv) if x > athresh]
            F,I = cc_utils.ft_active_integrals(
                    self.sys, en, focc, fvir, iocc, ivir)
        else:
            F,I = cc_utils.ft_integrals(self.sys, en, beta, mu)
        t2_temp = 0.25*self.T2[tf] + 0.5*einsum('ai,bj->abij',self.T1[tf],self.T1[tf])
        Es1 = einsum('ai,ia->',self.T1[tf],F.ov)
        Es2 = einsum('abij,ijab->',t2_temp,I.oovv)
        return (Es1 + Es2)/beta

    def _u_gderiv_approx(self):
        # temperature info
        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        ea,eb = self.sys.u_energies_tot()
        na = ea.shape[0]
        nb = eb.shape[0]

        ng = self.ngrid
        tf = ng - 1
        if self.athresh > 0.0:
            athresh = self.athresh
            foa = ft_utils.ff(beta, ea, mu)
            fva = ft_utils.ffv(beta, ea, mu)
            fob = ft_utils.ff(beta, eb, mu)
            fvb = ft_utils.ffv(beta, eb, mu)
            focca = [x for x in foa if x > athresh]
            fvira = [x for x in fva if x > athresh]
            iocca = [i for i,x in enumerate(foa) if x > athresh]
            ivira = [i for i,x in enumerate(fva) if x > athresh]
            foccb = [x for x in fob if x > athresh]
            fvirb = [x for x in fvb if x > athresh]
            ioccb = [i for i,x in enumerate(fob) if x > athresh]
            ivirb = [i for i,x in enumerate(fvb) if x > athresh]
            Fa,Fb,Ia,Ib,Iabab = cc_utils.uft_active_integrals(
                    self.sys, ea, eb, focca, fvira, foccb, fvirb, iocca, ivira, ioccb, ivirb)
        else:
            Fa,Fb,Ia,Ib,Iabab = cc_utils.uft_integrals(self.sys, ea, eb, beta, mu)
        T1a,T1b = self.T1
        T2aa,T2ab,T2bb = self.T2
        t2aa_temp = 0.25*T2aa[tf] + 0.5*einsum('ai,bj->abij',T1a[tf],T1a[tf])
        t2bb_temp = 0.25*T2bb[tf] + 0.5*einsum('ai,bj->abij',T1b[tf],T1b[tf])
        t2ab_temp = T2ab[tf] + einsum('ai,bj->abij',T1a[tf],T1b[tf])

        Es1 = einsum('ai,ia->',T1a[tf],Fa.ov)
        Es1 += einsum('ai,ia->',T1b[tf],Fb.ov)
        Es2 = einsum('abij,ijab->',t2aa_temp,Ia.oovv)
        Es2 += einsum('abij,ijab->',t2ab_temp,Iabab.oovv)
        Es2 += einsum('abij,ijab->',t2bb_temp,Ib.oovv)

        return (Es1 + Es2) / beta

    def _g_ft_1rdm(self):
        if self.L2 is None:
            self._ft_ccsd_lambda()
        # temperature info
        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu

        # get time-grid
        ng = self.ngrid
        ti = self.ti

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)

        # get energy differences
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
                - en[None,None,:,None] - en[None,None,None,:]
        if self.athresh > 0.0:
            athresh = self.athresh
            focc = [x for x in fo if x > athresh]
            fvir = [x for x in fv if x > athresh]
            iocc = [i for i,x in enumerate(fo) if x > athresh]
            ivir = [i for i,x in enumerate(fv) if x > athresh]
            D1 = D1[numpy.ix_(ivir,iocc)]
            D2 = D2[numpy.ix_(ivir,ivir,iocc,iocc)]

        pia,pba,pji,pai = ft_cc_equations.ccsd_1rdm(
                self.T1,self.T2,self.L1,self.L2,D1,D2,ti,ng,self.g,self.G)
        self.dia = pia
        self.dba = pba
        self.dji = pji
        self.dai = pai

    def _g_ft_2rdm(self):
        # temperature info
        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu

        # get time-grid
        ng = self.ngrid
        ti = self.ti

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)

        # get energy differences
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
                - en[None,None,:,None] - en[None,None,None,:]
        if self.athresh > 0.0:
            athresh = self.athresh
            focc = [x for x in fo if x > athresh]
            fvir = [x for x in fv if x > athresh]
            iocc = [i for i,x in enumerate(fo) if x > athresh]
            ivir = [i for i,x in enumerate(fv) if x > athresh]
            D1 = D1[numpy.ix_(ivir,iocc)]
            D2 = D2[numpy.ix_(ivir,ivir,iocc,iocc)]

        P2 = ft_cc_equations.ccsd_2rdm(
                self.T1,self.T2,self.L1,self.L2,D1,D2,ti,ng,self.g,self.G)
        self.P2 = P2

    def _grel_ft_1rdm(self):
        # build unrelaxed 1RDM and 2RDM if it doesn't exist
        if self.dia is None:
            self._g_ft_1rdm()
        if self.P2 is None:
            self._g_ft_2rdm()

        # temperature info
        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)

        # get integrals
        if self.athresh > 0.0:
            athresh = self.athresh
            focc = [x for x in fo if x > athresh]
            fvir = [x for x in fv if x > athresh]
            iocc = [i for i,x in enumerate(fo) if x > athresh]
            ivir = [i for i,x in enumerate(fv) if x > athresh]
            nocc = len(focc)
            nvir = len(fvir)
            F,I = cc_utils.ft_active_integrals(
                    self.sys, en, focc, fvir, iocc, ivir)
        else:
            focc = fo
            fvir = fv
            F,I = cc_utils.ft_integrals(self.sys, en, beta, mu)
        sfo = numpy.sqrt(focc)
        sfv = numpy.sqrt(fvir)
        if self.athresh > 0.0:
            dso = fv[numpy.ix_(iocc)]
            dsv = fo[numpy.ix_(ivir)]
        else:
            dso = fv
            dsv = fo
        n = fo.shape[0]

        # multiply unrelaxed RDMs by occupation numbers to form unrelaxed (normal-ordered) RDM
        self.ndia = einsum('ia,i,a->ia',self.dia,sfo,sfv)
        self.ndba = einsum('ba,b,a->ba',self.dba,sfv,sfv)
        self.ndji = einsum('ji,j,i->ji',self.dji,sfo,sfo)
        self.ndai = einsum('ai,a,i->ai',self.dai,sfv,sfo)
        if self.athresh > 0.0:
            self.n1rdm = numpy.zeros((n,n))
            self.n1rdm[numpy.ix_(iocc,ivir)] += self.ndia/beta
            self.n1rdm[numpy.ix_(ivir,ivir)] += self.ndba/beta
            self.n1rdm[numpy.ix_(iocc,iocc)] += self.ndji/beta
            self.n1rdm[numpy.ix_(ivir,iocc)] += self.ndai/beta
        else:
            self.n1rdm = (self.ndia + self.ndba + self.ndji + self.ndai)/beta

        # perturbed ON contribution to Fock matrix
        Fd = self.sys.g_fock_d_den()
        self.rdba = numpy.zeros((n,n))
        self.rdji = numpy.zeros((n,n))
        if self.athresh > 0.0:
            Fdai = Fd[numpy.ix_(ivir,iocc)]
            Fdab = Fd[numpy.ix_(ivir,ivir)]
            Fdij = Fd[numpy.ix_(iocc,iocc)]
            Fdia = Fd[numpy.ix_(iocc,ivir)]
            self.rdji -= numpy.diag(einsum('ia,aik->k',self.ndia,Fdai))
            self.rdji -= numpy.diag(einsum('ba,abk->k',self.ndba,Fdab))
            self.rdji -= numpy.diag(einsum('ji,ijk->k',self.ndji,Fdij))
            self.rdji -= numpy.diag(einsum('ai,iak->k',self.ndai,Fdia))
        else:
            self.rdji -= numpy.diag(einsum('ia,aik->k',self.ndia,Fd))
            self.rdji -= numpy.diag(einsum('ba,abk->k',self.ndba,Fd))
            self.rdji -= numpy.diag(einsum('ji,ijk->k',self.ndji,Fd))
            self.rdji -= numpy.diag(einsum('ai,iak->k',self.ndai,Fd))

        # append HF density matrix
        self.rdji += numpy.diag(fo)

        # append ON correction to HF density
        self.rdji += numpy.diag(self.sys.g_mp1_den())

        jitemp = numpy.zeros((nocc,nocc)) if self.athresh > 0.0 else numpy.zeros((n,n))
        batemp = numpy.zeros((nvir,nvir)) if self.athresh > 0.0 else numpy.zeros((n,n))

        # Add contributions to oo from occupation number relaxation
        jitemp -= numpy.diag((0.5*einsum('ia,ai->i',self.dia,F.vo)*dso))
        jitemp -= numpy.diag((0.5*einsum('ji,ij->i',self.dji,F.oo)*dso))
        jitemp -= numpy.diag((0.5*einsum('ji,ij->j',self.dji,F.oo)*dso))
        jitemp -= numpy.diag((0.5*einsum('ai,ia->i',self.dai,F.ov)*dso))

        jitemp -= numpy.diag((0.5*0.50*einsum('ciab,abci->i',self.P2[1],I.vvvo)*dso))
        jitemp -= numpy.diag((0.5*0.50*einsum('bcai,aibc->i',self.P2[2],I.vovv)*dso))
        jitemp -= numpy.diag((0.5*0.25*einsum('ijab,abij->i',self.P2[3],I.vvoo)*dso))
        jitemp -= numpy.diag((0.5*0.25*einsum('ijab,abij->j',self.P2[3],I.vvoo)*dso))
        jitemp -= numpy.diag((0.5*1.00*einsum('bjai,aibj->i',self.P2[4],I.vovo)*dso))
        jitemp -= numpy.diag((0.5*1.00*einsum('bjai,aibj->j',self.P2[4],I.vovo)*dso))
        jitemp -= numpy.diag((0.5*0.25*einsum('abij,ijab->i',self.P2[5],I.oovv)*dso))
        jitemp -= numpy.diag((0.5*0.25*einsum('abij,ijab->j',self.P2[5],I.oovv)*dso))
        jitemp -= numpy.diag((0.5*0.50*einsum('jkai,aijk->i',self.P2[6],I.vooo)*dso))
        jitemp -= numpy.diag((0.5*0.50*einsum('jkai,aijk->j',self.P2[6],I.vooo)*dso))
        jitemp -= numpy.diag((0.5*0.50*einsum('jkai,aijk->k',self.P2[6],I.vooo)*dso))
        jitemp -= numpy.diag((0.5*0.50*einsum('kaij,ijka->i',self.P2[7],I.ooov)*dso))
        jitemp -= numpy.diag((0.5*0.50*einsum('kaij,ijka->j',self.P2[7],I.ooov)*dso))
        jitemp -= numpy.diag((0.5*0.50*einsum('kaij,ijka->k',self.P2[7],I.ooov)*dso))
        jitemp -= numpy.diag((0.5*0.25*einsum('klij,ijkl->i',self.P2[8],I.oooo)*dso))
        jitemp -= numpy.diag((0.5*0.25*einsum('klij,ijkl->j',self.P2[8],I.oooo)*dso))
        jitemp -= numpy.diag((0.5*0.25*einsum('klij,ijkl->k',self.P2[8],I.oooo)*dso))
        jitemp -= numpy.diag((0.5*0.25*einsum('klij,ijkl->l',self.P2[8],I.oooo)*dso))

        # Add contributions to vv from occupation number relaxation
        batemp += numpy.diag((0.50*einsum('ia,ai->a',self.dia,F.vo)*dsv))
        batemp += numpy.diag((0.50*einsum('ba,ab->b',self.dba,F.vv)*dsv))
        batemp += numpy.diag((0.50*einsum('ba,ab->a',self.dba,F.vv)*dsv))
        batemp += numpy.diag((0.50*einsum('ai,ia->a',self.dai,F.ov)*dsv))

        batemp += numpy.diag((0.5*0.25*einsum('cdab,abcd->c',self.P2[0],I.vvvv)*dsv))
        batemp += numpy.diag((0.5*0.25*einsum('cdab,abcd->d',self.P2[0],I.vvvv)*dsv))
        batemp += numpy.diag((0.5*0.25*einsum('cdab,abcd->a',self.P2[0],I.vvvv)*dsv))
        batemp += numpy.diag((0.5*0.25*einsum('cdab,abcd->b',self.P2[0],I.vvvv)*dsv))
        batemp += numpy.diag((0.5*0.50*einsum('ciab,abci->c',self.P2[1],I.vvvo)*dsv))
        batemp += numpy.diag((0.5*0.50*einsum('ciab,abci->a',self.P2[1],I.vvvo)*dsv))
        batemp += numpy.diag((0.5*0.50*einsum('ciab,abci->b',self.P2[1],I.vvvo)*dsv))
        batemp += numpy.diag((0.5*0.50*einsum('bcai,aibc->b',self.P2[2],I.vovv)*dsv))
        batemp += numpy.diag((0.5*0.50*einsum('bcai,aibc->c',self.P2[2],I.vovv)*dsv))
        batemp += numpy.diag((0.5*0.50*einsum('bcai,aibc->a',self.P2[2],I.vovv)*dsv))
        batemp += numpy.diag((0.5*0.25*einsum('ijab,abij->a',self.P2[3],I.vvoo)*dsv))
        batemp += numpy.diag((0.5*0.25*einsum('ijab,abij->b',self.P2[3],I.vvoo)*dsv))
        batemp += numpy.diag((0.5*1.00*einsum('bjai,aibj->a',self.P2[4],I.vovo)*dsv))
        batemp += numpy.diag((0.5*1.00*einsum('bjai,aibj->b',self.P2[4],I.vovo)*dsv))
        batemp += numpy.diag((0.5*0.25*einsum('abij,ijab->a',self.P2[5],I.oovv)*dsv))
        batemp += numpy.diag((0.5*0.25*einsum('abij,ijab->b',self.P2[5],I.oovv)*dsv))
        batemp += numpy.diag((0.5*0.50*einsum('jkai,aijk->a',self.P2[6],I.vooo)*dsv))
        batemp += numpy.diag((0.5*0.50*einsum('kaij,ijka->a',self.P2[7],I.ooov)*dsv))

        if self.athresh > 0.0:
            self.rdji[numpy.ix_(iocc,iocc)] += jitemp
            self.rdba[numpy.ix_(ivir,ivir)] += batemp
        else:
            self.rdji += jitemp
            self.rdba += batemp

        # orbital energy derivatives
        Gnew = self.G.copy()
        m = Gnew.shape[0]
        n = Gnew.shape[0]
        for i in range(m):
            for j in range(n):
                Gnew[i,j] *= (self.ti[j] - self.ti[i])

        ng = self.ti.shape[0]
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
                - en[None,None,:,None] - en[None,None,None,:]
        if self.athresh > 0.0:
            D1 = D1[numpy.ix_(ivir,iocc)]
            D2 = D2[numpy.ix_(ivir,ivir,iocc,iocc)]
        T1temp,T2temp = ft_cc_equations.ccsd_stanton(F,I,self.T1,self.T2,
                D1,D2,self.ti,ng,Gnew)
        At1i = (1.0/beta)*einsum('via,vai->vi',self.L1, T1temp)
        At1a = (1.0/beta)*einsum('via,vai->va',self.L1, T1temp)
        At2i = (1.0/beta)*0.25*einsum('vijab,vabij->vi',self.L2, T2temp)
        At2j = (1.0/beta)*0.25*einsum('vijab,vabij->vj',self.L2, T2temp)
        At2a = (1.0/beta)*0.25*einsum('vijab,vabij->va',self.L2, T2temp)
        At2b = (1.0/beta)*0.25*einsum('vijab,vabij->vb',self.L2, T2temp)
        if self.athresh > 0.0:
            self.rdji[numpy.ix_(iocc,iocc)] -= numpy.diag(einsum('vi,v->i',At1i+At2i+At2j,self.g))
            self.rdba[numpy.ix_(ivir,ivir)] += numpy.diag(einsum('va,v->a',At1a+At2a+At2b,self.g))
        else:
            self.rdji -= numpy.diag(einsum('vi,v->i',At1i+At2i+At2j,self.g))
            self.rdba += numpy.diag(einsum('va,v->a',At1a+At2a+At2b,self.g))

        self.r1rdm = self.rdji + self.rdba

    def _u_ft_1rdm(self):
        if self.L2 is None:
            self._ft_uccsd_lambda()
        # temperature info
        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu

        # get time-grid
        ng = self.ngrid
        ti = self.ti

        # get energies and occupation numbers
        ea,eb = self.sys.u_energies_tot()
        na = ea.shape[0]
        nb = eb.shape[0]

        # get energy differences
        D1a = ea[:,None] - ea[None,:]
        D1b = eb[:,None] - eb[None,:]
        D2aa = ea[:,None,None,None] + ea[None,:,None,None] \
            - ea[None,None,:,None] - ea[None,None,None,:]
        D2ab = ea[:,None,None,None] + eb[None,:,None,None] \
            - ea[None,None,:,None] - eb[None,None,None,:]
        D2bb = eb[:,None,None,None] + eb[None,:,None,None] \
            - eb[None,None,:,None] - eb[None,None,None,:]
        if self.athresh > 0.0:
            athresh = self.athresh
            foa = ft_utils.ff(beta, ea, mu)
            fva = ft_utils.ffv(beta, ea, mu)
            fob = ft_utils.ff(beta, eb, mu)
            fvb = ft_utils.ffv(beta, eb, mu)
            focca = [x for x in foa if x > athresh]
            fvira = [x for x in fva if x > athresh]
            iocca = [i for i,x in enumerate(foa) if x > athresh]
            ivira = [i for i,x in enumerate(fva) if x > athresh]
            foccb = [x for x in fob if x > athresh]
            fvirb = [x for x in fvb if x > athresh]
            ioccb = [i for i,x in enumerate(fob) if x > athresh]
            ivirb = [i for i,x in enumerate(fvb) if x > athresh]
            D1a = D1a[numpy.ix_(ivira,iocca)]
            D1b = D1b[numpy.ix_(ivirb,ioccb)]
            D2aa = D2aa[numpy.ix_(ivira,ivira,iocca,iocca)]
            D2ab = D2ab[numpy.ix_(ivira,ivirb,iocca,ioccb)]
            D2bb = D2bb[numpy.ix_(ivirb,ivirb,ioccb,ioccb)]

        T1a,T1b = self.T1
        T2aa,T2ab,T2bb = self.T2
        L1a,L1b = self.L1
        L2aa,L2ab,L2bb = self.L2
        pia,pba,pji,pai = ft_cc_equations.uccsd_1rdm(
                T1a,T1b,T2aa,T2ab,T2bb,L1a,L1b,L2aa,L2ab,L2bb,
                D1a,D1b,D2aa,D2ab,D2bb,ti,ng,self.g,self.G)
        self.dia = pia
        self.dba = pba
        self.dji = pji
        self.dai = pai

    def _u_ft_2rdm(self):
        # temperature info
        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu

        # get time-grid
        ng = self.ngrid
        ti = self.ti

        # get energies and occupation numbers
        ea,eb = self.sys.u_energies_tot()
        na = ea.shape[0]
        nb = eb.shape[0]

        # get energy differences
        D1a = ea[:,None] - ea[None,:]
        D1b = eb[:,None] - eb[None,:]
        D2aa = ea[:,None,None,None] + ea[None,:,None,None] \
            - ea[None,None,:,None] - ea[None,None,None,:]
        D2ab = ea[:,None,None,None] + eb[None,:,None,None] \
            - ea[None,None,:,None] - eb[None,None,None,:]
        D2bb = eb[:,None,None,None] + eb[None,:,None,None] \
            - eb[None,None,:,None] - eb[None,None,None,:]
        if self.athresh > 0.0:
            athresh = self.athresh
            foa = ft_utils.ff(beta, ea, mu)
            fva = ft_utils.ffv(beta, ea, mu)
            fob = ft_utils.ff(beta, eb, mu)
            fvb = ft_utils.ffv(beta, eb, mu)
            focca = [x for x in foa if x > athresh]
            fvira = [x for x in fva if x > athresh]
            iocca = [i for i,x in enumerate(foa) if x > athresh]
            ivira = [i for i,x in enumerate(fva) if x > athresh]
            foccb = [x for x in fob if x > athresh]
            fvirb = [x for x in fvb if x > athresh]
            ioccb = [i for i,x in enumerate(fob) if x > athresh]
            ivirb = [i for i,x in enumerate(fvb) if x > athresh]
            D1a = D1a[numpy.ix_(ivira,iocca)]
            D1b = D1b[numpy.ix_(ivirb,ioccb)]
            D2aa = D2aa[numpy.ix_(ivira,ivira,iocca,iocca)]
            D2ab = D2ab[numpy.ix_(ivira,ivirb,iocca,ioccb)]

        T1a,T1b = self.T1
        T2aa,T2ab,T2bb = self.T2
        L1a,L1b = self.L1
        L2aa,L2ab,L2bb = self.L2
        P2 = ft_cc_equations.uccsd_2rdm(
                T1a,T1b,T2aa,T2ab,T2bb,L1a,L1b,L2aa,L2ab,L2bb,
                D1a,D1b,D2aa,D2ab,D2bb,ti,ng,self.g,self.G)

        self.P2 = P2
