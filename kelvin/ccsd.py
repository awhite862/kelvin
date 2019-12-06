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
        self.beta = None
        self.ti = None
        self.g = None
        self.G = None
        if self.realtime:
            if self.finite_T:
                self.beta = 1.0/T
                self.beta_max = self.beta
            else:
                self.beta_max = 80.0
            ng = self.ngrid
            self.ti,self.g,self.G = quadrature.ft_quad(self.ngrid, self.beta_max, self.quad)
        self.sys = sys
        # amplitudes
        self.T1 = None
        self.T2 = None
        self.L1 = None
        self.L2 = None
        # pieces of 1-rdm
        self.dia = None
        self.dba = None
        self.dji = None
        self.dai = None
        # pieces of 1-rdm with ONs
        self.ndia = None
        self.ndba = None
        self.ndji = None
        self.ndai = None
        # 2-rdm
        self.P2 = None
        # full 1-rdm with ONs
        self.n1rdm = None
        # full 2-rdm with ONs
        self.n2rdm = None
        # ON- and OE-relaxation contribution to 1-rdm
        self.r1rdm = None

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
    #            beta = self.beta
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
            assert(self.beta_max == self.beta)
            beta = self.beta
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
            assert(self.beta_max == self.beta)
            beta = self.beta
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
            self.Scc = self.S - self.S0 - self.S1

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
            beta = self.beta
            mu = self.mu if self.finite_T else None
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
            E2 = ft_cc_energy.ft_cc_energy(T1old, T2old,
                F.ov, I.oovv, g, self.beta_max, Qterm=False)
            if self.iprint > 0:
                print('MP2 Energy: {:.10f}'.format(E2))

            # run CC iterations
            Eccn,T1,T2 = cc_utils.ft_cc_iter(method, T1old, T2old, F, I, D1, D2, g, G,
                    self.beta_max, ng, ti, self.iprint, conv_options)
        else:
            T1,T2 = cc_utils.ft_cc_iter_extrap(method, F, I, D1, D2, g, G, self.beta_max, ng, ti,
                    self.iprint, conv_options)
            Eccn = ft_cc_energy.ft_cc_energy(T1,T2,
                F.ov,I.oovv,g,self.beta_max)

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
        # get time-grid
        ng = self.ngrid
        ti = self.ti
        G = self.G
        g = self.g

        if self.finite_T:
            beta = self.beta
            mu = self.mu if self.finite_T else None

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

            beta = self.beta
            mu = self.mu
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
                T1bshape = (ng,nvirb,noccb)

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
                T1bshape = (ng,nb,nb)
        else:
            # create energies in spin-orbital basis
            eoa,eva,eob,evb = self.sys.u_energies()
            no = eo.shape[0]
            nv = ev.shape[0]
            D1a = (eva[:,None] - eoa[None,:])
            D1b = (evb[:,None] - eob[None,:])
            D2aa = (eva[:,None,None,None] + eva[None,:,None,None]
                - eoa[None,None,:,None] - eoa[None,None,None,:])
            D2ab = (eva[:,None,None,None] + evb[None,:,None,None]
                - eoa[None,None,:,None] - eob[None,None,None,:])
            D2bb = (evb[:,None,None,None] + evb[None,:,None,None]
                - eob[None,None,:,None] - eob[None,None,None,:])

            # get HF energy
            En = self.sys.const_energy()
            E0 = zt_mp.ump0(eoa, eob) + En
            E1 = self.sys.get_mp1()
            E01 = E0 + E1

            # get Fock matrix
            Fa,Fb = self.sys.u_fock()
            Fa.oo = Fa.oo - numpy.diag(eoa) # subtract diagonal
            Fa.vv = Fa.vv - numpy.diag(eva) # subtract diagonal
            Fb.oo = Fb.oo - numpy.diag(eob) # subtract diagonal
            Fb.vv = Fb.vv - numpy.diag(evb) # subtract diagonal

            # get ERIs
            Ia, Ib, Iabab = self.sys.u_aint()

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
                Fa.ov,Fb.ov,Ia.oovv,Ib.oovv,Iabab.oovv,g,self.beta_max,Qterm=False)
            if self.iprint > 0:
                print('MP2 Energy: {:.10f}'.format(E2))


            # run CC iterations
            Eccn,T1,T2 = cc_utils.ft_ucc_iter(method, T1aold, T1bold, T2aaold, T2abold, T2bbold,
                    Fa, Fb, Ia, Ib, Iabab, D1a, D1b, D2aa, D2ab, D2bb,
                    g, G, self.beta_max, ng, ti, self.iprint, conv_options)
        else:
            T1,T2 = cc_utils.ft_ucc_iter_extrap(method, Fa, Fb, Ia, Ib, Iabab, D1a, D1b, D2aa, D2ab, D2bb,
                    g, G, self.beta_max, ng, ti, self.iprint, conv_options)
            Eccn = ft_cc_energy.ft_ucc_energy(T1[0], T1[1], T2[0], T2[1], T2[2],
                Fa.ov,Fb.ov,Ia.oovv,Ib.oovv,Iabab.oovv,g,self.beta_max)

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
        beta = self.beta
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
                L1old,L2old = ft_cc_equations.ccsd_lambda_guess(F,I,self.T1,self.beta_max,ng)
            else:
                L2old = ft_cc_equations.ccd_lambda_guess(I,self.beta_max,ng)
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
                D1, D2, g, G, self.beta_max, ng, ti, self.iprint, conv_options)

        # save lambda amplitudes
        self.L1 = L1
        self.L2 = L2

    def _ft_uccsd_lambda(self, L1=None, L2=None):
        """Solve FT-CCSD Lambda equations."""
        beta = self.beta
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
                    Fa,Fb,Ia,Ib,Iabab,self.T1[0],self.T1[1],self.beta_max,ng)
            else:
                L2aaold,L2abold,L2bbold = ft_cc_equations.uccd_lambda_guess(Ia,Ib,Iabab,self.beta_max,ng)
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
                g, G, self.beta_max, ng, ti, self.iprint, conv_options)

        # save lambda amplitudes
        self.L1 = (L1a,L1b)
        self.L2 = (L2aa,L2ab,L2bb)

    def _g_nocc_deriv(self, dvec):
        """Evaluate the Lagrangian with scaled occupation number derivatives."""
        # temperature info
        assert(self.beta == self.beta_max)
        beta = self.beta
        mu = self.mu

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)
        #n = fo.shape[0]

        # first order contributions
        der1 = self.sys.g_d_mp1(dvec)

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
        # temperature info
        assert(self.beta == self.beta_max)
        beta = self.beta
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
                    self.sys, ea, eb, foa, fva, fob, fvb,
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
        assert(self.beta == self.beta_max)
        beta = self.beta
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

        A1g = -einsum('v,v->',A1,g)
        A2g = -einsum('v,v->',A2,g)
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

        A1 = -(1.0/beta)*einsum('via,vai->v',self.L1, T1temp)
        A2 = -(1.0/beta)*0.25*einsum('vijab,vabij->v',self.L2, T2temp)

        A1g = einsum('v,v->',A1,g)
        A2g = einsum('v,v->',A2,g)
        dG += A1g + A2g

        return dg,dG

    def _u_nocc_gderiv(self):
        """Evaluate the derivatives of the weight matrices in
        the time-dependent formulation.
        """
        # temperature info
        assert(self.beta == self.beta_max)
        beta = self.beta
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
            D2bb = D2bb[numpy.ix_(ivira,ivira,iocca,iocca)]
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

        A1a = -(1.0/beta)*einsum('via,vai->v',L1aold, T1t[0])
        A1b = -(1.0/beta)*einsum('via,vai->v',L1bold, T1t[1])
        A2a = -(1.0/beta)*0.25*einsum('vijab,vabij->v',L2aaold, T2t[0])
        A2b = -(1.0/beta)*0.25*einsum('vijab,vabij->v',L2bbold, T2t[2])
        A2ab = -(1.0/beta)*einsum('vijab,vabij->v',L2abold, T2t[1])

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

        A1a = -(1.0/beta)*einsum('via,vai->v',L1aold, T1a)
        A1b = -(1.0/beta)*einsum('via,vai->v',L1bold, T1b)
        A2a = -(1.0/beta)*0.25*einsum('vijab,vabij->v',L2aaold, T2aa)
        A2b = -(1.0/beta)*0.25*einsum('vijab,vabij->v',L2bbold, T2bb)
        A2ab = -(1.0/beta)*einsum('vijab,vabij->v',L2abold, T2ab)

        A2g = einsum('v,v->',A2a,g)
        A2g += einsum('v,v->',A2ab,g)
        A2g += einsum('v,v->',A2b,g)
        A1g = einsum('v,v->',A1a,g)
        A1g += einsum('v,v->',A1b,g)
        dG += A1g + A2g

        return dg,dG

    def _g_gderiv_approx(self):
        # temperature info
        assert(self.beta == self.beta_max)
        beta = self.beta
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
        assert(self.beta == self.beta_max)
        beta = self.beta
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
        assert(self.beta == self.beta_max)
        if self.L2 is None:
            self._ft_ccsd_lambda()
        # temperature info
        beta = self.beta
        mu = self.mu

        # get time-grid
        ng = self.ngrid
        ti = self.ti

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)
        n = en.shape[0]

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
        else:
            focc = fo
            fvir = fv
        sfo = numpy.sqrt(focc)
        sfv = numpy.sqrt(fvir)

        pia,pba,pji,pai = ft_cc_equations.ccsd_1rdm(
                self.T1,self.T2,self.L1,self.L2,D1,D2,ti,ng,self.g,self.G)
        self.dia = pia
        self.dba = pba
        self.dji = pji
        self.dai = pai

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

    def _g_ft_2rdm(self):
        assert(self.beta == self.beta_max)
        # temperature info
        beta = self.beta
        mu = self.mu

        # get time-grid
        ng = self.ngrid
        ti = self.ti

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)
        n = en.shape[0]

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
        else:
            focc = fo
            fvir = fv

        sfo = numpy.sqrt(focc)
        sfv = numpy.sqrt(fvir)
        P2 = ft_cc_equations.ccsd_2rdm(
                self.T1,self.T2,self.L1,self.L2,D1,D2,ti,ng,self.g,self.G)
        self.P2 = P2

        # compute normal-ordered 2rdm
        if self.athresh > 0.0:
            self.n2rdm = numpy.zeros((n,n,n,n))
            self.n2rdm[numpy.ix_(ivir,ivir,ivir,ivir)] += \
                    (1.0/beta)*einsum('cdab,c,d,a,b->cdab',P2[0],sfv,sfv,sfv,sfv)
            self.n2rdm[numpy.ix_(ivir,iocc,ivir,ivir)] += \
                    (1.0/beta)*einsum('ciab,c,i,a,b->ciab',P2[1],sfv,sfo,sfv,sfv)
            self.n2rdm[numpy.ix_(iocc,ivir,ivir,ivir)] -= \
                    (1.0/beta)*einsum('ciab,c,i,a,b->icab',P2[1],sfv,sfo,sfv,sfv)
            self.n2rdm[numpy.ix_(ivir,ivir,ivir,iocc)] += \
                    (1.0/beta)*einsum('bcai,b,c,a,i->bcai',P2[2],sfv,sfv,sfv,sfo)
            self.n2rdm[numpy.ix_(ivir,ivir,iocc,ivir)] -= \
                    (1.0/beta)*einsum('bcai,b,c,a,i->bcia',P2[2],sfv,sfv,sfv,sfo)
            self.n2rdm[numpy.ix_(iocc,iocc,ivir,ivir)] += \
                    (1.0/beta)*einsum('ijab,i,j,a,b->ijab',P2[3],sfo,sfo,sfv,sfv)
            self.n2rdm[numpy.ix_(ivir,iocc,ivir,iocc)] += \
                    (1.0/beta)*einsum('bjai,b,j,a,i->bjai',P2[4],sfv,sfo,sfv,sfo)
            self.n2rdm[numpy.ix_(ivir,iocc,iocc,ivir)] -= \
                    (1.0/beta)*einsum('bjai,b,j,a,i->bjia',P2[4],sfv,sfo,sfv,sfo)
            self.n2rdm[numpy.ix_(iocc,ivir,ivir,iocc)] -= \
                    (1.0/beta)*einsum('bjai,b,j,a,i->jbai',P2[4],sfv,sfo,sfv,sfo)
            self.n2rdm[numpy.ix_(iocc,ivir,iocc,ivir)] += \
                    (1.0/beta)*einsum('bjai,b,j,a,i->jbia',P2[4],sfv,sfo,sfv,sfo)
            self.n2rdm[numpy.ix_(ivir,ivir,iocc,iocc)] += \
                    (1.0/beta)*einsum('abij,a,b,i,j->abij',P2[5],sfv,sfv,sfo,sfo)
            self.n2rdm[numpy.ix_(iocc,iocc,ivir,iocc)] += \
                    (1.0/beta)*einsum('jkai,j,k,a,i->jkai',P2[6],sfo,sfo,sfv,sfo)
            self.n2rdm[numpy.ix_(iocc,iocc,iocc,ivir)] -= \
                    (1.0/beta)*einsum('jkai,j,k,a,i->jkia',P2[6],sfo,sfo,sfv,sfo)
            self.n2rdm[numpy.ix_(iocc,ivir,iocc,iocc)] += \
                    (1.0/beta)*einsum('kaij,k,a,i,j->kaij',P2[7],sfo,sfv,sfo,sfo)
            self.n2rdm[numpy.ix_(ivir,iocc,iocc,iocc)] -= \
                    (1.0/beta)*einsum('kaij,k,a,i,j->akij',P2[7],sfo,sfv,sfo,sfo)
            self.n2rdm[numpy.ix_(iocc,iocc,iocc,iocc)] += \
                    (1.0/beta)*einsum('klij,k,l,i,j->klij',P2[8],sfo,sfo,sfo,sfo)
        else:
            self.n2rdm = (1.0/beta)*einsum('cdab,c,d,a,b->cdab',P2[0],sfv,sfv,sfv,sfv)
            self.n2rdm += (1.0/beta)*einsum('ciab,c,i,a,b->ciab',P2[1],sfv,sfo,sfv,sfv)
            self.n2rdm -= (1.0/beta)*einsum('ciab,c,i,a,b->icab',P2[1],sfv,sfo,sfv,sfv)
            self.n2rdm += (1.0/beta)*einsum('bcai,b,c,a,i->bcai',P2[2],sfv,sfv,sfv,sfo)
            self.n2rdm -= (1.0/beta)*einsum('bcai,b,c,a,i->bcia',P2[2],sfv,sfv,sfv,sfo)
            self.n2rdm += (1.0/beta)*einsum('ijab,i,j,a,b->ijab',P2[3],sfo,sfo,sfv,sfv)
            self.n2rdm += (1.0/beta)*einsum('bjai,b,j,a,i->bjai',P2[4],sfv,sfo,sfv,sfo)
            self.n2rdm -= (1.0/beta)*einsum('bjai,b,j,a,i->bjia',P2[4],sfv,sfo,sfv,sfo)
            self.n2rdm -= (1.0/beta)*einsum('bjai,b,j,a,i->jbai',P2[4],sfv,sfo,sfv,sfo)
            self.n2rdm += (1.0/beta)*einsum('bjai,b,j,a,i->jbia',P2[4],sfv,sfo,sfv,sfo)
            self.n2rdm += (1.0/beta)*einsum('abij,a,b,i,j->abij',P2[5],sfv,sfv,sfo,sfo)
            self.n2rdm += (1.0/beta)*einsum('jkai,j,k,a,i->jkai',P2[6],sfo,sfo,sfv,sfo)
            self.n2rdm -= (1.0/beta)*einsum('jkai,j,k,a,i->jkia',P2[6],sfo,sfo,sfv,sfo)
            self.n2rdm += (1.0/beta)*einsum('kaij,k,a,i,j->kaij',P2[7],sfo,sfv,sfo,sfo)
            self.n2rdm -= (1.0/beta)*einsum('kaij,k,a,i,j->akij',P2[7],sfo,sfv,sfo,sfo)
            self.n2rdm += (1.0/beta)*einsum('klij,k,l,i,j->klij',P2[8],sfo,sfo,sfo,sfo)

    def _grel_ft_1rdm(self):
        assert(self.beta == self.beta_max)
        # build unrelaxed 1RDM and 2RDM if it doesn't exist
        if self.dia is None:
            self._g_ft_1rdm()
        if self.P2 is None:
            self._g_ft_2rdm()

        # temperature info
        beta = self.beta
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
        if self.athresh > 0.0:
            dso = fv[numpy.ix_(iocc)]
            dsv = fo[numpy.ix_(ivir)]
        else:
            dso = fv
            dsv = fo
        n = fo.shape[0]

        # perturbed ON contribution to Fock matrix
        Fd = self.sys.g_fock_d_den()
        rdba = numpy.zeros((n,n))
        rdji = numpy.zeros((n,n))
        if self.athresh > 0.0:
            Fdai = Fd[numpy.ix_(ivir,iocc)]
            Fdab = Fd[numpy.ix_(ivir,ivir)]
            Fdij = Fd[numpy.ix_(iocc,iocc)]
            Fdia = Fd[numpy.ix_(iocc,ivir)]
            rdji -= numpy.diag(einsum('ia,aik->k',self.ndia,Fdai))
            rdji -= numpy.diag(einsum('ba,abk->k',self.ndba,Fdab))
            rdji -= numpy.diag(einsum('ji,ijk->k',self.ndji,Fdij))
            rdji -= numpy.diag(einsum('ai,iak->k',self.ndai,Fdia))
        else:
            rdji -= numpy.diag(einsum('ia,aik->k',self.ndia,Fd))
            rdji -= numpy.diag(einsum('ba,abk->k',self.ndba,Fd))
            rdji -= numpy.diag(einsum('ji,ijk->k',self.ndji,Fd))
            rdji -= numpy.diag(einsum('ai,iak->k',self.ndai,Fd))

        # append HF density matrix
        rdji += numpy.diag(fo)

        # append ON correction to HF density
        rdji += numpy.diag(self.sys.g_mp1_den())

        jitemp = numpy.zeros((nocc,nocc)) if self.athresh > 0.0 else numpy.zeros((n,n))
        batemp = numpy.zeros((nvir,nvir)) if self.athresh > 0.0 else numpy.zeros((n,n))

        # Add contributions to oo from occupation number relaxation
        jitemp -= numpy.diag((0.5*einsum('ia,ai->i',self.dia,F.vo)*dso))
        jitemp -= numpy.diag((0.5*einsum('ji,ij->i',self.dji,F.oo)*dso))
        jitemp -= numpy.diag((0.5*einsum('ji,ij->j',self.dji,F.oo)*dso))
        jitemp -= numpy.diag((0.5*einsum('ai,ia->i',self.dai,F.ov)*dso))

        jitemp -= numpy.diag((0.5*0.50*einsum('ciab,abci->i',self.P2[1],I.vvvo)*dso))
        jitemp -= numpy.diag((0.5*0.50*einsum('bcai,aibc->i',self.P2[2],I.vovv)*dso))
        jitemp -= numpy.diag((0.5*0.5*einsum('ijab,abij->i',self.P2[3],I.vvoo)*dso))
        jitemp -= numpy.diag((0.5*1.00*einsum('bjai,aibj->i',self.P2[4],I.vovo)*dso))
        jitemp -= numpy.diag((0.5*1.00*einsum('bjai,aibj->j',self.P2[4],I.vovo)*dso))
        jitemp -= numpy.diag((0.5*0.5*einsum('abij,ijab->i',self.P2[5],I.oovv)*dso))
        jitemp -= numpy.diag((0.5*0.50*einsum('jkai,aijk->i',self.P2[6],I.vooo)*dso))
        jitemp -= numpy.diag((0.5*1.0*einsum('jkai,aijk->j',self.P2[6],I.vooo)*dso))
        jitemp -= numpy.diag((0.5*1.0*einsum('kaij,ijka->i',self.P2[7],I.ooov)*dso))
        jitemp -= numpy.diag((0.5*0.50*einsum('kaij,ijka->k',self.P2[7],I.ooov)*dso))
        jitemp -= numpy.diag((0.5*0.5*einsum('klij,ijkl->i',self.P2[8],I.oooo)*dso))
        jitemp -= numpy.diag((0.5*0.5*einsum('klij,ijkl->k',self.P2[8],I.oooo)*dso))

        # Add contributions to vv from occupation number relaxation
        batemp += numpy.diag((0.50*einsum('ia,ai->a',self.dia,F.vo)*dsv))
        batemp += numpy.diag((0.50*einsum('ba,ab->b',self.dba,F.vv)*dsv))
        batemp += numpy.diag((0.50*einsum('ba,ab->a',self.dba,F.vv)*dsv))
        batemp += numpy.diag((0.50*einsum('ai,ia->a',self.dai,F.ov)*dsv))

        batemp += numpy.diag((0.5*0.5*einsum('cdab,abcd->c',self.P2[0],I.vvvv)*dsv))
        batemp += numpy.diag((0.5*0.5*einsum('cdab,abcd->a',self.P2[0],I.vvvv)*dsv))
        batemp += numpy.diag((0.5*0.50*einsum('ciab,abci->c',self.P2[1],I.vvvo)*dsv))
        batemp += numpy.diag((0.5*1.0*einsum('ciab,abci->a',self.P2[1],I.vvvo)*dsv))
        batemp += numpy.diag((0.5*1.0*einsum('bcai,aibc->b',self.P2[2],I.vovv)*dsv))
        batemp += numpy.diag((0.5*0.50*einsum('bcai,aibc->a',self.P2[2],I.vovv)*dsv))
        batemp += numpy.diag((0.5*0.50*einsum('ijab,abij->a',self.P2[3],I.vvoo)*dsv))
        batemp += numpy.diag((0.5*1.00*einsum('bjai,aibj->a',self.P2[4],I.vovo)*dsv))
        batemp += numpy.diag((0.5*1.00*einsum('bjai,aibj->b',self.P2[4],I.vovo)*dsv))
        batemp += numpy.diag((0.5*0.5*einsum('abij,ijab->a',self.P2[5],I.oovv)*dsv))
        batemp += numpy.diag((0.5*0.50*einsum('jkai,aijk->a',self.P2[6],I.vooo)*dsv))
        batemp += numpy.diag((0.5*0.50*einsum('kaij,ijka->a',self.P2[7],I.ooov)*dsv))

        if self.athresh > 0.0:
            rdji[numpy.ix_(iocc,iocc)] += jitemp
            rdba[numpy.ix_(ivir,ivir)] += batemp
        else:
            rdji += jitemp
            rdba += batemp

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
        At1i = -(1.0/beta)*einsum('via,vai->vi',self.L1, T1temp)
        At1a = -(1.0/beta)*einsum('via,vai->va',self.L1, T1temp)
        At2i = -(1.0/beta)*0.25*einsum('vijab,vabij->vi',self.L2, T2temp)
        At2j = -(1.0/beta)*0.25*einsum('vijab,vabij->vj',self.L2, T2temp)
        At2a = -(1.0/beta)*0.25*einsum('vijab,vabij->va',self.L2, T2temp)
        At2b = -(1.0/beta)*0.25*einsum('vijab,vabij->vb',self.L2, T2temp)
        if self.athresh > 0.0:
            rdji[numpy.ix_(iocc,iocc)] -= numpy.diag(einsum('vi,v->i',At1i+At2i+At2j,self.g))
            rdba[numpy.ix_(ivir,ivir)] += numpy.diag(einsum('va,v->a',At1a+At2a+At2b,self.g))
        else:
            rdji -= numpy.diag(einsum('vi,v->i',At1i+At2i+At2j,self.g))
            rdba += numpy.diag(einsum('va,v->a',At1a+At2a+At2b,self.g))

        self.r1rdm = rdji + rdba

    def _u_ft_1rdm(self):
        assert(self.beta == self.beta_max)
        if self.L2 is None:
            self._ft_uccsd_lambda()
        # temperature info
        beta = self.beta
        mu = self.mu

        # get time-grid
        ng = self.ngrid
        ti = self.ti

        # get energies and occupation numbers
        ea,eb = self.sys.u_energies_tot()
        na = ea.shape[0]
        nb = eb.shape[0]
        foa = ft_utils.ff(beta, ea, mu)
        fva = ft_utils.ffv(beta, ea, mu)
        fob = ft_utils.ff(beta, eb, mu)
        fvb = ft_utils.ffv(beta, eb, mu)

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
        else:
            focca = foa
            fvira = fva
            foccb = fob
            fvirb = fvb
        sfoa = numpy.sqrt(focca)
        sfva = numpy.sqrt(fvira)
        sfob = numpy.sqrt(foccb)
        sfvb = numpy.sqrt(fvirb)
        na = foa.shape[0]
        nb = fob.shape[0]

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

        # multiply unrelaxed RDMs by occupation numbers to form unrelaxed (normal-ordered) RDM
        self.ndia = (einsum('ia,i,a->ia',self.dia[0],sfoa,sfva),
                einsum('ia,i,a->ia',self.dia[1],sfob,sfvb))
        self.ndba = (einsum('ba,b,a->ba',self.dba[0],sfva,sfva),
                einsum('ba,b,a->ba',self.dba[1],sfvb,sfvb))
        self.ndji = (einsum('ji,j,i->ji',self.dji[0],sfoa,sfoa),
                einsum('ji,j,i->ji',self.dji[1],sfob,sfob))
        self.ndai = (einsum('ai,a,i->ai',self.dai[0],sfva,sfoa),
                einsum('ai,a,i->ai',self.dai[1],sfvb,sfob))
        if self.athresh > 0.0:
            self.n1rdm = [numpy.zeros((na,na)),numpy.zeros((nb,nb))]
            self.n1rdm[0][numpy.ix_(iocca,ivira)] += self.ndia[0]/beta
            self.n1rdm[0][numpy.ix_(ivira,ivira)] += self.ndba[0]/beta
            self.n1rdm[0][numpy.ix_(iocca,iocca)] += self.ndji[0]/beta
            self.n1rdm[0][numpy.ix_(ivira,iocca)] += self.ndai[0]/beta
            self.n1rdm[1][numpy.ix_(ioccb,ivirb)] += self.ndia[1]/beta
            self.n1rdm[1][numpy.ix_(ivirb,ivirb)] += self.ndba[1]/beta
            self.n1rdm[1][numpy.ix_(ioccb,ioccb)] += self.ndji[1]/beta
            self.n1rdm[1][numpy.ix_(ivirb,ioccb)] += self.ndai[1]/beta
        else:
            self.n1rdm = [(self.ndia[0] + self.ndba[0] + self.ndji[0] + self.ndai[0])/beta,
                    (self.ndia[1] + self.ndba[1] + self.ndji[1] + self.ndai[1])/beta]

    def _u_ft_2rdm(self):
        assert(self.beta == self.beta_max)
        # temperature info
        beta = self.beta
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
        foa = ft_utils.ff(beta, ea, mu)
        fva = ft_utils.ffv(beta, ea, mu)
        fob = ft_utils.ff(beta, eb, mu)
        fvb = ft_utils.ffv(beta, eb, mu)
        if self.athresh > 0.0:
            athresh = self.athresh
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
        else:
            focca = foa
            fvira = fva
            foccb = fob
            fvirb = fvb
        sfoa = numpy.sqrt(focca)
        sfva = numpy.sqrt(fvira)
        sfob = numpy.sqrt(foccb)
        sfvb = numpy.sqrt(fvirb)

        T1a,T1b = self.T1
        T2aa,T2ab,T2bb = self.T2
        L1a,L1b = self.L1
        L2aa,L2ab,L2bb = self.L2
        P2 = ft_cc_equations.uccsd_2rdm(
                T1a,T1b,T2aa,T2ab,T2bb,L1a,L1b,L2aa,L2ab,L2bb,
                D1a,D1b,D2aa,D2ab,D2bb,ti,ng,self.g,self.G)

        self.P2 = P2
        if self.athresh > 0.0:
            P2aa = numpy.zeros((na,na,na,na))
            P2aa[numpy.ix_(ivira,ivira,ivira,ivira)] += (1.0/beta)*einsum('cdab,c,d,a,b->cdab',self.P2[0][0],sfva,sfva,sfva,sfva)
            P2aa[numpy.ix_(ivira,iocca,ivira,ivira)] += (1.0/beta)*einsum('ciab,c,i,a,b->ciab',self.P2[1][0],sfva,sfoa,sfva,sfva)
            P2aa[numpy.ix_(iocca,ivira,ivira,ivira)] -= (1.0/beta)*einsum('ciab,c,i,a,b->icab',self.P2[1][0],sfva,sfoa,sfva,sfva)
            P2aa[numpy.ix_(ivira,ivira,ivira,iocca)] += (1.0/beta)*einsum('bcai,b,c,a,i->bcai',self.P2[2][0],sfva,sfva,sfva,sfoa)
            P2aa[numpy.ix_(ivira,ivira,iocca,ivira)] -= (1.0/beta)*einsum('bcai,b,c,a,i->bcia',self.P2[2][0],sfva,sfva,sfva,sfoa)
            P2aa[numpy.ix_(iocca,iocca,ivira,ivira)] += (1.0/beta)*einsum('ijab,i,j,a,b->ijab',self.P2[3][0],sfoa,sfoa,sfva,sfva)
            P2aa[numpy.ix_(ivira,iocca,ivira,iocca)] += (1.0/beta)*einsum('bjai,b,j,a,i->bjai',self.P2[4][0],sfva,sfoa,sfva,sfoa)
            P2aa[numpy.ix_(ivira,iocca,iocca,ivira)] -= (1.0/beta)*einsum('bjai,b,j,a,i->bjia',self.P2[4][0],sfva,sfoa,sfva,sfoa)
            P2aa[numpy.ix_(iocca,ivira,ivira,iocca)] -= (1.0/beta)*einsum('bjai,b,j,a,i->jbai',self.P2[4][0],sfva,sfoa,sfva,sfoa)
            P2aa[numpy.ix_(iocca,ivira,iocca,ivira)] += (1.0/beta)*einsum('bjai,b,j,a,i->jbia',self.P2[4][0],sfva,sfoa,sfva,sfoa)
            P2aa[numpy.ix_(ivira,ivira,iocca,iocca)] += (1.0/beta)*einsum('abij,a,b,i,j->abij',self.P2[5][0],sfva,sfva,sfoa,sfoa)
            P2aa[numpy.ix_(iocca,iocca,ivira,iocca)] += (1.0/beta)*einsum('jkai,j,k,a,i->jkai',self.P2[6][0],sfoa,sfoa,sfva,sfoa)
            P2aa[numpy.ix_(iocca,iocca,iocca,ivira)] -= (1.0/beta)*einsum('jkai,j,k,a,i->jkia',self.P2[6][0],sfoa,sfoa,sfva,sfoa)
            P2aa[numpy.ix_(iocca,ivira,iocca,iocca)] += (1.0/beta)*einsum('kaij,k,a,i,j->kaij',self.P2[7][0],sfoa,sfva,sfoa,sfoa)
            P2aa[numpy.ix_(ivira,iocca,iocca,iocca)] -= (1.0/beta)*einsum('kaij,k,a,i,j->akij',self.P2[7][0],sfoa,sfva,sfoa,sfoa)
            P2aa[numpy.ix_(iocca,iocca,iocca,iocca)] += (1.0/beta)*einsum('klij,k,l,i,j->klij',self.P2[8][0],sfoa,sfoa,sfoa,sfoa)

            P2bb = numpy.zeros((nb,nb,nb,nb))
            P2bb[numpy.ix_(ivirb,ivirb,ivirb,ivirb)] += (1.0/beta)*einsum('cdab,c,d,a,b->cdab',self.P2[0][1],sfvb,sfvb,sfvb,sfvb)
            P2bb[numpy.ix_(ivirb,ioccb,ivirb,ivirb)] += (1.0/beta)*einsum('ciab,c,i,a,b->ciab',self.P2[1][1],sfvb,sfob,sfvb,sfvb)
            P2bb[numpy.ix_(ioccb,ivirb,ivirb,ivirb)] -= (1.0/beta)*einsum('ciab,c,i,a,b->icab',self.P2[1][1],sfvb,sfob,sfvb,sfvb)
            P2bb[numpy.ix_(ivirb,ivirb,ivirb,ioccb)] += (1.0/beta)*einsum('bcai,b,c,a,i->bcai',self.P2[2][1],sfvb,sfvb,sfvb,sfob)
            P2bb[numpy.ix_(ivirb,ivirb,ioccb,ivirb)] -= (1.0/beta)*einsum('bcai,b,c,a,i->bcia',self.P2[2][1],sfvb,sfvb,sfvb,sfob)
            P2bb[numpy.ix_(ioccb,ioccb,ivirb,ivirb)] += (1.0/beta)*einsum('ijab,i,j,a,b->ijab',self.P2[3][1],sfob,sfob,sfvb,sfvb)
            P2bb[numpy.ix_(ivirb,ioccb,ivirb,ioccb)] += (1.0/beta)*einsum('bjai,b,j,a,i->bjai',self.P2[4][1],sfvb,sfob,sfvb,sfob)
            P2bb[numpy.ix_(ivirb,ioccb,ioccb,ivirb)] -= (1.0/beta)*einsum('bjai,b,j,a,i->bjia',self.P2[4][1],sfvb,sfob,sfvb,sfob)
            P2bb[numpy.ix_(ioccb,ivirb,ivirb,ioccb)] -= (1.0/beta)*einsum('bjai,b,j,a,i->jbai',self.P2[4][1],sfvb,sfob,sfvb,sfob)
            P2bb[numpy.ix_(ioccb,ivirb,ioccb,ivirb)] += (1.0/beta)*einsum('bjai,b,j,a,i->jbia',self.P2[4][1],sfvb,sfob,sfvb,sfob)
            P2bb[numpy.ix_(ivirb,ivirb,ioccb,ioccb)] += (1.0/beta)*einsum('abij,a,b,i,j->abij',self.P2[5][1],sfvb,sfvb,sfob,sfob)
            P2bb[numpy.ix_(ioccb,ioccb,ivirb,ioccb)] += (1.0/beta)*einsum('jkai,j,k,a,i->jkai',self.P2[6][1],sfob,sfob,sfvb,sfob)
            P2bb[numpy.ix_(ioccb,ioccb,ioccb,ivirb)] -= (1.0/beta)*einsum('jkai,j,k,a,i->jkia',self.P2[6][1],sfob,sfob,sfvb,sfob)
            P2bb[numpy.ix_(ioccb,ivirb,ioccb,ioccb)] += (1.0/beta)*einsum('kaij,k,a,i,j->kaij',self.P2[7][1],sfob,sfvb,sfob,sfob)
            P2bb[numpy.ix_(ivirb,ioccb,ioccb,ioccb)] -= (1.0/beta)*einsum('kaij,k,a,i,j->akij',self.P2[7][1],sfob,sfvb,sfob,sfob)
            P2bb[numpy.ix_(ioccb,ioccb,ioccb,ioccb)] += (1.0/beta)*einsum('klij,k,l,i,j->klij',self.P2[8][1],sfob,sfob,sfob,sfob)

            P2ab = numpy.zeros((na,nb,na,nb))
            P2ab[numpy.ix_(ivira,ivirb,ivira,ivirb)] += (1.0/beta)*einsum('cdab,c,d,a,b->cdab',self.P2[0][2],sfva,sfvb,sfva,sfvb)
            P2ab[numpy.ix_(ivira,ioccb,ivira,ivirb)] += (1.0/beta)*einsum('ciab,c,i,a,b->ciab',self.P2[1][2],sfva,sfob,sfva,sfvb)
            P2ab[numpy.ix_(ivira,ivirb,ivira,ioccb)] += (1.0/beta)*einsum('bcai,b,c,a,i->bcai',self.P2[2][2],sfva,sfvb,sfva,sfob)
            P2ab[numpy.ix_(iocca,ioccb,ivira,ivirb)] += (1.0/beta)*einsum('ijab,i,j,a,b->ijab',self.P2[3][2],sfoa,sfob,sfva,sfvb)
            P2ab[numpy.ix_(ivira,ioccb,ivira,ioccb)] += (1.0/beta)*einsum('bjai,b,j,a,i->bjai',self.P2[4][2],sfva,sfob,sfva,sfob)
            P2ab[numpy.ix_(ivira,ivirb,iocca,ioccb)] += (1.0/beta)*einsum('abij,a,b,i,j->abij',self.P2[5][2],sfva,sfvb,sfoa,sfob)
            P2ab[numpy.ix_(iocca,ioccb,ivira,ioccb)] += (1.0/beta)*einsum('jkai,j,k,a,i->jkai',self.P2[6][2],sfoa,sfob,sfva,sfob)
            P2ab[numpy.ix_(iocca,ivirb,iocca,ioccb)] += (1.0/beta)*einsum('kaij,k,a,i,j->kaij',self.P2[7][2],sfoa,sfvb,sfoa,sfob)
            P2ab[numpy.ix_(iocca,ioccb,iocca,ioccb)] += (1.0/beta)*einsum('klij,k,l,i,j->klij',self.P2[8][2],sfoa,sfob,sfoa,sfob)

            P2ab[numpy.ix_(iocca,ivirb,ivira,ivirb)] += (1.0/beta)*einsum('ciab,c,i,a,b->ciab',self.P2[1][3],sfvb,sfoa,sfvb,sfva).transpose((1,0,3,2))
            P2ab[numpy.ix_(ivira,ivirb,iocca,ivirb)] += (1.0/beta)*einsum('bcai,b,c,a,i->bcai',self.P2[2][3],sfvb,sfva,sfvb,sfoa).transpose((1,0,3,2))

            P2ab[numpy.ix_(iocca,ioccb,iocca,ivirb)] += (1.0/beta)*einsum('jkai,j,k,a,i->jkai',self.P2[6][3],sfob,sfoa,sfvb,sfoa).transpose((1,0,3,2))
            P2ab[numpy.ix_(ivira,ioccb,iocca,iocca)] += (1.0/beta)*einsum('kaij,k,a,i,j->kaij',self.P2[7][3],sfob,sfva,sfob,sfoa).transpose((1,0,3,2))

            P2ab[numpy.ix_(ivira,ioccb,iocca,ivirb)] -= (1.0/beta)*einsum('bjai,b,j,a,i->bjai',self.P2[4][3],sfva,sfob,sfvb,sfoa).transpose((0,1,3,2))
            P2ab[numpy.ix_(iocca,ivirb,ivira,ioccb)] -= (1.0/beta)*einsum('bjai,b,j,a,i->bjai',self.P2[4][4],sfvb,sfoa,sfva,sfob).transpose((1,0,2,3))
            P2ab[numpy.ix_(iocca,ivirb,iocca,ivirb)] += (1.0/beta)*einsum('bjai,b,j,a,i->bjai',self.P2[4][5],sfvb,sfoa,sfvb,sfoa).transpose((1,0,3,2))
        else:
            P2aa = numpy.zeros((na,na,na,na))
            P2aa += (1.0/beta)*einsum('cdab,c,d,a,b->cdab',self.P2[0][0],sfva,sfva,sfva,sfva)
            P2aa += (1.0/beta)*einsum('ciab,c,i,a,b->ciab',self.P2[1][0],sfva,sfoa,sfva,sfva)
            P2aa -= (1.0/beta)*einsum('ciab,c,i,a,b->icab',self.P2[1][0],sfva,sfoa,sfva,sfva)
            P2aa += (1.0/beta)*einsum('bcai,b,c,a,i->bcai',self.P2[2][0],sfva,sfva,sfva,sfoa)
            P2aa -= (1.0/beta)*einsum('bcai,b,c,a,i->bcia',self.P2[2][0],sfva,sfva,sfva,sfoa)
            P2aa += (1.0/beta)*einsum('ijab,i,j,a,b->ijab',self.P2[3][0],sfoa,sfoa,sfva,sfva)
            P2aa += (1.0/beta)*einsum('bjai,b,j,a,i->bjai',self.P2[4][0],sfva,sfoa,sfva,sfoa)
            P2aa -= (1.0/beta)*einsum('bjai,b,j,a,i->bjia',self.P2[4][0],sfva,sfoa,sfva,sfoa)
            P2aa -= (1.0/beta)*einsum('bjai,b,j,a,i->jbai',self.P2[4][0],sfva,sfoa,sfva,sfoa)
            P2aa += (1.0/beta)*einsum('bjai,b,j,a,i->jbia',self.P2[4][0],sfva,sfoa,sfva,sfoa)
            P2aa += (1.0/beta)*einsum('abij,a,b,i,j->abij',self.P2[5][0],sfva,sfva,sfoa,sfoa)
            P2aa += (1.0/beta)*einsum('jkai,j,k,a,i->jkai',self.P2[6][0],sfoa,sfoa,sfva,sfoa)
            P2aa -= (1.0/beta)*einsum('jkai,j,k,a,i->jkia',self.P2[6][0],sfoa,sfoa,sfva,sfoa)
            P2aa += (1.0/beta)*einsum('kaij,k,a,i,j->kaij',self.P2[7][0],sfoa,sfva,sfoa,sfoa)
            P2aa -= (1.0/beta)*einsum('kaij,k,a,i,j->akij',self.P2[7][0],sfoa,sfva,sfoa,sfoa)
            P2aa += (1.0/beta)*einsum('klij,k,l,i,j->klij',self.P2[8][0],sfoa,sfoa,sfoa,sfoa)

            P2bb = numpy.zeros((nb,nb,nb,nb))
            P2bb += (1.0/beta)*einsum('cdab,c,d,a,b->cdab',self.P2[0][1],sfvb,sfvb,sfvb,sfvb)
            P2bb += (1.0/beta)*einsum('ciab,c,i,a,b->ciab',self.P2[1][1],sfvb,sfob,sfvb,sfvb)
            P2bb -= (1.0/beta)*einsum('ciab,c,i,a,b->icab',self.P2[1][1],sfvb,sfob,sfvb,sfvb)
            P2bb += (1.0/beta)*einsum('bcai,b,c,a,i->bcai',self.P2[2][1],sfvb,sfvb,sfvb,sfob)
            P2bb -= (1.0/beta)*einsum('bcai,b,c,a,i->bcia',self.P2[2][1],sfvb,sfvb,sfvb,sfob)
            P2bb += (1.0/beta)*einsum('ijab,i,j,a,b->ijab',self.P2[3][1],sfob,sfob,sfvb,sfvb)
            P2bb += (1.0/beta)*einsum('bjai,b,j,a,i->bjai',self.P2[4][1],sfvb,sfob,sfvb,sfob)
            P2bb -= (1.0/beta)*einsum('bjai,b,j,a,i->bjia',self.P2[4][1],sfvb,sfob,sfvb,sfob)
            P2bb -= (1.0/beta)*einsum('bjai,b,j,a,i->jbai',self.P2[4][1],sfvb,sfob,sfvb,sfob)
            P2bb += (1.0/beta)*einsum('bjai,b,j,a,i->jbia',self.P2[4][1],sfvb,sfob,sfvb,sfob)
            P2bb += (1.0/beta)*einsum('abij,a,b,i,j->abij',self.P2[5][1],sfvb,sfvb,sfob,sfob)
            P2bb += (1.0/beta)*einsum('jkai,j,k,a,i->jkai',self.P2[6][1],sfob,sfob,sfvb,sfob)
            P2bb -= (1.0/beta)*einsum('jkai,j,k,a,i->jkia',self.P2[6][1],sfob,sfob,sfvb,sfob)
            P2bb += (1.0/beta)*einsum('kaij,k,a,i,j->kaij',self.P2[7][1],sfob,sfvb,sfob,sfob)
            P2bb -= (1.0/beta)*einsum('kaij,k,a,i,j->akij',self.P2[7][1],sfob,sfvb,sfob,sfob)
            P2bb += (1.0/beta)*einsum('klij,k,l,i,j->klij',self.P2[8][1],sfob,sfob,sfob,sfob)

            P2ab = numpy.zeros((na,nb,na,nb))
            P2ab += (1.0/beta)*einsum('cdab,c,d,a,b->cdab',self.P2[0][2],sfva,sfvb,sfva,sfvb)
            P2ab += (1.0/beta)*einsum('ciab,c,i,a,b->ciab',self.P2[1][2],sfva,sfob,sfva,sfvb)
            P2ab += (1.0/beta)*einsum('bcai,b,c,a,i->bcai',self.P2[2][2],sfva,sfvb,sfva,sfob)
            P2ab += (1.0/beta)*einsum('ijab,i,j,a,b->ijab',self.P2[3][2],sfoa,sfob,sfva,sfvb)
            P2ab += (1.0/beta)*einsum('bjai,b,j,a,i->bjai',self.P2[4][2],sfva,sfob,sfva,sfob)
            P2ab += (1.0/beta)*einsum('abij,a,b,i,j->abij',self.P2[5][2],sfva,sfvb,sfoa,sfob)
            P2ab += (1.0/beta)*einsum('jkai,j,k,a,i->jkai',self.P2[6][2],sfoa,sfob,sfva,sfob)
            P2ab += (1.0/beta)*einsum('kaij,k,a,i,j->kaij',self.P2[7][2],sfoa,sfvb,sfoa,sfob)
            P2ab += (1.0/beta)*einsum('klij,k,l,i,j->klij',self.P2[8][2],sfoa,sfob,sfoa,sfob)

            P2ab += (1.0/beta)*einsum('ciab,c,i,a,b->ciab',self.P2[1][3],sfvb,sfoa,sfvb,sfva).transpose((1,0,3,2))
            P2ab += (1.0/beta)*einsum('bcai,b,c,a,i->bcai',self.P2[2][3],sfvb,sfva,sfvb,sfoa).transpose((1,0,3,2))

            P2ab += (1.0/beta)*einsum('jkai,j,k,a,i->jkai',self.P2[6][3],sfob,sfoa,sfvb,sfoa).transpose((1,0,3,2))
            P2ab += (1.0/beta)*einsum('kaij,k,a,i,j->kaij',self.P2[7][3],sfob,sfva,sfob,sfoa).transpose((1,0,3,2))

            P2ab -= (1.0/beta)*einsum('bjai,b,j,a,i->bjai',self.P2[4][3],sfva,sfob,sfvb,sfoa).transpose((0,1,3,2))
            P2ab -= (1.0/beta)*einsum('bjai,b,j,a,i->bjai',self.P2[4][4],sfvb,sfoa,sfva,sfob).transpose((1,0,2,3))
            P2ab += (1.0/beta)*einsum('bjai,b,j,a,i->bjai',self.P2[4][5],sfvb,sfoa,sfvb,sfoa).transpose((1,0,3,2))

        self.n2rdm = (P2aa,P2bb,P2ab)

    def _urel_ft_1rdm(self):
        assert(self.beta == self.beta_max)
        # build unrelaxed 1RDM and 2RDM if it doesn't exist
        if self.dia is None:
            self._u_ft_1rdm()
        if self.P2 is None:
            self._u_ft_2rdm()

        # temperature info
        beta = self.beta
        mu = self.mu

        # get energies and occupation numbers
        ea,eb = self.sys.u_energies_tot()
        na = ea.shape[0]
        nb = eb.shape[0]
        foa = ft_utils.ff(beta, ea, mu)
        fva = ft_utils.ffv(beta, ea, mu)
        fob = ft_utils.ff(beta, eb, mu)
        fvb = ft_utils.ffv(beta, eb, mu)

        # get integrals
        if self.athresh > 0.0:
            athresh = self.athresh
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
            Fa,Fb,Ia,Ib,Iabab = cc_utils.uft_active_integrals(
                    self.sys, ea, eb, focca, fvira, foccb, fvirb, iocca, ivira, ioccb, ivirb)
        else:
            focca = foa
            fvira = fva
            foccb = fob
            fvirb = fvb
            Fa,Fb,Ia,Ib,Iabab = cc_utils.uft_integrals(self.sys, ea, eb, beta, mu)
        if self.athresh > 0.0:
            dsoa = fva[numpy.ix_(iocca)]
            dsva = foa[numpy.ix_(ivira)]
            dsob = fvb[numpy.ix_(ioccb)]
            dsvb = fob[numpy.ix_(ivirb)]
        else:
            dsoa = fva
            dsva = foa
            dsob = fvb
            dsvb = fob

        # perturbed ON contribution to Fock matrix
        Fdaa,Fdab,Fdbb,Fdba = self.sys.u_fock_d_den()
        rdba = [numpy.zeros((na,na)),numpy.zeros((nb,nb))]
        rdji = [numpy.zeros((na,na)),numpy.zeros((nb,nb))]
        if self.athresh > 0.0:
            Fdaik = Fdaa[numpy.ix_(ivira,iocca)]
            Fdabk = Fdaa[numpy.ix_(ivira,ivira)]
            Fdijk = Fdaa[numpy.ix_(iocca,iocca)]
            Fdiak = Fdaa[numpy.ix_(iocca,ivira)]
            FdaiK = Fdab[numpy.ix_(ivira,iocca)]
            FdabK = Fdab[numpy.ix_(ivira,ivira)]
            FdijK = Fdab[numpy.ix_(iocca,iocca)]
            FdiaK = Fdab[numpy.ix_(iocca,ivira)]
            FdAIK = Fdbb[numpy.ix_(ivirb,ioccb)]
            FdABK = Fdbb[numpy.ix_(ivirb,ivirb)]
            FdIJK = Fdbb[numpy.ix_(ioccb,ioccb)]
            FdIAK = Fdbb[numpy.ix_(ioccb,ivirb)]
            FdAIk = Fdba[numpy.ix_(ivirb,ioccb)]
            FdABk = Fdba[numpy.ix_(ivirb,ivirb)]
            FdIJk = Fdba[numpy.ix_(ioccb,ioccb)]
            FdIAk = Fdba[numpy.ix_(ioccb,ivirb)]
            rdji[0] -= numpy.diag(einsum('ia,aik->k',self.ndia[0],Fdaik))
            rdji[0] -= numpy.diag(einsum('ba,abk->k',self.ndba[0],Fdabk))
            rdji[0] -= numpy.diag(einsum('ji,ijk->k',self.ndji[0],Fdijk))
            rdji[0] -= numpy.diag(einsum('ai,iak->k',self.ndai[0],Fdiak))
            rdji[0] -= numpy.diag(einsum('ia,aik->k',self.ndia[1],FdAIk))
            rdji[0] -= numpy.diag(einsum('ba,abk->k',self.ndba[1],FdABk))
            rdji[0] -= numpy.diag(einsum('ji,ijk->k',self.ndji[1],FdIJk))
            rdji[0] -= numpy.diag(einsum('ai,iak->k',self.ndai[1],FdIAk))
            rdji[1] -= numpy.diag(einsum('ia,aik->k',self.ndia[1],FdAIK))
            rdji[1] -= numpy.diag(einsum('ba,abk->k',self.ndba[1],FdABK))
            rdji[1] -= numpy.diag(einsum('ji,ijk->k',self.ndji[1],FdIJK))
            rdji[1] -= numpy.diag(einsum('ai,iak->k',self.ndai[1],FdIAK))
            rdji[1] -= numpy.diag(einsum('ia,aik->k',self.ndia[0],FdaiK))
            rdji[1] -= numpy.diag(einsum('ba,abk->k',self.ndba[0],FdabK))
            rdji[1] -= numpy.diag(einsum('ji,ijk->k',self.ndji[0],FdijK))
            rdji[1] -= numpy.diag(einsum('ai,iak->k',self.ndai[0],FdiaK))
        else:
            rdji[0] -= numpy.diag(einsum('ia,aik->k',self.ndia[0],Fdaa))
            rdji[0] -= numpy.diag(einsum('ba,abk->k',self.ndba[0],Fdaa))
            rdji[0] -= numpy.diag(einsum('ji,ijk->k',self.ndji[0],Fdaa))
            rdji[0] -= numpy.diag(einsum('ai,iak->k',self.ndai[0],Fdaa))
            rdji[0] -= numpy.diag(einsum('ia,aik->k',self.ndia[1],Fdba))
            rdji[0] -= numpy.diag(einsum('ba,abk->k',self.ndba[1],Fdba))
            rdji[0] -= numpy.diag(einsum('ji,ijk->k',self.ndji[1],Fdba))
            rdji[0] -= numpy.diag(einsum('ai,iak->k',self.ndai[1],Fdba))
            rdji[1] -= numpy.diag(einsum('ia,aik->k',self.ndia[1],Fdbb))
            rdji[1] -= numpy.diag(einsum('ba,abk->k',self.ndba[1],Fdbb))
            rdji[1] -= numpy.diag(einsum('ji,ijk->k',self.ndji[1],Fdbb))
            rdji[1] -= numpy.diag(einsum('ai,iak->k',self.ndai[1],Fdbb))
            rdji[1] -= numpy.diag(einsum('ia,aik->k',self.ndia[0],Fdab))
            rdji[1] -= numpy.diag(einsum('ba,abk->k',self.ndba[0],Fdab))
            rdji[1] -= numpy.diag(einsum('ji,ijk->k',self.ndji[0],Fdab))
            rdji[1] -= numpy.diag(einsum('ai,iak->k',self.ndai[0],Fdab))

        # append HF density matrix
        rdji[0] += numpy.diag(foa)
        rdji[1] += numpy.diag(fob)

        # append ON correction to HF density
        mp1da,mp1db = self.sys.u_mp1_den()
        rdji[0] += numpy.diag(mp1da)
        rdji[1] += numpy.diag(mp1db)

        jitempa = numpy.zeros((nocca,nocca)) if self.athresh > 0.0 else numpy.zeros((na,na))
        batempa = numpy.zeros((nvira,nvira)) if self.athresh > 0.0 else numpy.zeros((na,na))
        jitempb = numpy.zeros((noccb,noccb)) if self.athresh > 0.0 else numpy.zeros((nb,nb))
        batempb = numpy.zeros((nvirb,nvirb)) if self.athresh > 0.0 else numpy.zeros((nb,nb))

        jitempa -= numpy.diag(0.5*einsum('ia,ai->i', self.dia[0], Fa.vo)*dsoa)
        jitempa -= numpy.diag(0.5*einsum('ji,ij->i', self.dji[0], Fa.oo)*dsoa)
        jitempa -= numpy.diag(0.5*einsum('ji,ij->j', self.dji[0], Fa.oo)*dsoa)
        jitempa -= numpy.diag(0.5*einsum('ai,ia->i', self.dai[0], Fa.ov)*dsoa)

        jitempb -= numpy.diag(0.5*einsum('ia,ai->i', self.dia[1], Fb.vo)*dsob)
        jitempb -= numpy.diag(0.5*einsum('ji,ij->i', self.dji[1], Fb.oo)*dsob)
        jitempb -= numpy.diag(0.5*einsum('ji,ij->j', self.dji[1], Fb.oo)*dsob)
        jitempb -= numpy.diag(0.5*einsum('ai,ia->i', self.dai[1], Fb.ov)*dsob)

        batempa += numpy.diag(0.5*einsum('ia,ai->a', self.dia[0], Fa.vo)*dsva)
        batempa += numpy.diag(0.5*einsum('ba,ab->a', self.dba[0], Fa.vv)*dsva)
        batempa += numpy.diag(0.5*einsum('ba,ab->b', self.dba[0], Fa.vv)*dsva)
        batempa += numpy.diag(0.5*einsum('ai,ia->a', self.dai[0], Fa.ov)*dsva)

        batempb += numpy.diag(0.5*einsum('ia,ai->a', self.dia[1], Fb.vo)*dsvb)
        batempb += numpy.diag(0.5*einsum('ba,ab->a', self.dba[1], Fb.vv)*dsvb)
        batempb += numpy.diag(0.5*einsum('ba,ab->b', self.dba[1], Fb.vv)*dsvb)
        batempb += numpy.diag(0.5*einsum('ai,ia->a', self.dai[1], Fb.ov)*dsvb)

        jitempa -= numpy.diag(0.5*0.5*einsum('ijab,abij->i', self.P2[3][0], Ia.vvoo)*dsoa)
        jitempb -= numpy.diag(0.5*0.5*einsum('ijab,abij->i', self.P2[3][1], Ib.vvoo)*dsob)
        jitempa -= numpy.diag(0.5*1.0*einsum('iJaB,aBiJ->i', self.P2[3][2], Iabab.vvoo)*dsoa)
        jitempb -= numpy.diag(0.5*1.0*einsum('iJaB,aBiJ->J', self.P2[3][2], Iabab.vvoo)*dsob)

        jitempa -= numpy.diag(0.5*0.5*einsum('ciab,abci->i', self.P2[1][0], Ia.vvvo)*dsoa)
        jitempb -= numpy.diag(0.5*0.5*einsum('ciab,abci->i', self.P2[1][1], Ib.vvvo)*dsob)
        jitempb -= numpy.diag(0.5*1.0*einsum('ciab,abci->i', self.P2[1][2], Iabab.vvvo)*dsob)
        jitempa -= numpy.diag(0.5*1.0*einsum('ciab,baic->i', self.P2[1][3], Iabab.vvov)*dsoa)

        jitempa -= numpy.diag(0.5*0.5*einsum('jkai,aijk->i', self.P2[6][0], Ia.vooo)*dsoa)
        jitempb -= numpy.diag(0.5*0.5*einsum('jkai,aijk->i', self.P2[6][1], Ib.vooo)*dsob)
        jitempb -= numpy.diag(0.5*1.0*einsum('jKaI,aIjK->I', self.P2[6][2], Iabab.vooo)*dsob)
        jitempa -= numpy.diag(0.5*1.0*einsum('JkAi,iAkJ->i', self.P2[6][3], Iabab.ovoo)*dsoa)
        jitempa -= numpy.diag(0.5*1.0*einsum('jkai,aijk->j', self.P2[6][0], Ia.vooo)*dsoa)
        jitempb -= numpy.diag(0.5*1.0*einsum('jkai,aijk->j', self.P2[6][1], Ib.vooo)*dsob)
        jitempa -= numpy.diag(0.5*1.0*einsum('jKaI,aIjK->j', self.P2[6][2], Iabab.vooo)*dsoa)
        jitempb -= numpy.diag(0.5*1.0*einsum('JkAi,iAkJ->J', self.P2[6][3], Iabab.ovoo)*dsob)
        jitempb -= numpy.diag(0.5*1.0*einsum('jKaI,aIjK->K', self.P2[6][2], Iabab.vooo)*dsob)
        jitempa -= numpy.diag(0.5*1.0*einsum('JkAi,iAkJ->k', self.P2[6][3], Iabab.ovoo)*dsoa)

        jitempa -= numpy.diag(0.5*1.0*einsum('bjai,aibj->i', self.P2[4][0], Ia.vovo)*dsoa)
        jitempa -= numpy.diag(0.5*1.0*einsum('bjai,aibj->j', self.P2[4][0], Ia.vovo)*dsoa)
        jitempb -= numpy.diag(0.5*1.0*einsum('BJAI,AIBJ->I', self.P2[4][1], Ib.vovo)*dsob)
        jitempb -= numpy.diag(0.5*1.0*einsum('BJAI,AIBJ->J', self.P2[4][1], Ib.vovo)*dsob)
        jitempb -= numpy.diag(0.5*1.0*einsum('bJaI,aIbJ->I', self.P2[4][2], Iabab.vovo)*dsob)
        jitempb -= numpy.diag(0.5*1.0*einsum('bJaI,aIbJ->J', self.P2[4][2], Iabab.vovo)*dsob)
        jitempa += numpy.diag(0.5*1.0*einsum('bJAi,iAbJ->i', self.P2[4][3], Iabab.ovvo)*dsoa)
        jitempb += numpy.diag(0.5*1.0*einsum('bJAi,iAbJ->J', self.P2[4][3], Iabab.ovvo)*dsob)
        jitempb += numpy.diag(0.5*1.0*einsum('BjaI,aIjB->I', self.P2[4][4], Iabab.voov)*dsob)
        jitempa += numpy.diag(0.5*1.0*einsum('BjaI,aIjB->j', self.P2[4][4], Iabab.voov)*dsoa)
        jitempa -= numpy.diag(0.5*1.0*einsum('BjAi,iAjB->i', self.P2[4][5], Iabab.ovov)*dsoa)
        jitempa -= numpy.diag(0.5*1.0*einsum('BjAi,iAjB->j', self.P2[4][5], Iabab.ovov)*dsoa)

        jitempa -= numpy.diag(0.5*0.5*einsum('klij,ijkl->i', self.P2[8][0], Ia.oooo)*dsoa)
        jitempa -= numpy.diag(0.5*0.5*einsum('klij,ijkl->k', self.P2[8][0], Ia.oooo)*dsoa)
        jitempb -= numpy.diag(0.5*0.5*einsum('klij,ijkl->i', self.P2[8][1], Ib.oooo)*dsob)
        jitempb -= numpy.diag(0.5*0.5*einsum('klij,ijkl->k', self.P2[8][1], Ib.oooo)*dsob)
        jitempa -= numpy.diag(0.5*1.0*einsum('kLiJ,iJkL->i', self.P2[8][2], Iabab.oooo)*dsoa)
        jitempb -= numpy.diag(0.5*1.0*einsum('kLiJ,iJkL->J', self.P2[8][2], Iabab.oooo)*dsob)
        jitempa -= numpy.diag(0.5*1.0*einsum('kLiJ,iJkL->k', self.P2[8][2], Iabab.oooo)*dsoa)
        jitempb -= numpy.diag(0.5*1.0*einsum('kLiJ,iJkL->L', self.P2[8][2], Iabab.oooo)*dsob)

        jitempa -= numpy.diag(0.5*0.5*einsum('bcai,aibc->i', self.P2[2][0], Ia.vovv)*dsoa)
        jitempb -= numpy.diag(0.5*0.5*einsum('bcai,aibc->i', self.P2[2][1], Ib.vovv)*dsob)
        jitempb -= numpy.diag(0.5*1.0*einsum('bCaI,aIbC->I', self.P2[2][2], Iabab.vovv)*dsob)
        jitempa -= numpy.diag(0.5*1.0*einsum('BcAi,iAcB->i', self.P2[2][3], Iabab.ovvv)*dsoa)

        jitempa -= numpy.diag(0.5*1.0*einsum('kaij,ijka->i', self.P2[7][0], Ia.ooov)*dsoa)
        jitempa -= numpy.diag(0.5*0.5*einsum('kaij,ijka->k', self.P2[7][0], Ia.ooov)*dsoa)
        jitempb -= numpy.diag(0.5*1.0*einsum('kaij,ijka->i', self.P2[7][1], Ib.ooov)*dsob)
        jitempb -= numpy.diag(0.5*0.5*einsum('kaij,ijka->k', self.P2[7][1], Ib.ooov)*dsob)
        jitempa -= numpy.diag(0.5*1.0*einsum('kAiJ,iJkA->i', self.P2[7][2], Iabab.ooov)*dsoa)
        jitempb -= numpy.diag(0.5*1.0*einsum('kAiJ,iJkA->J', self.P2[7][2], Iabab.ooov)*dsob)
        jitempa -= numpy.diag(0.5*1.0*einsum('kAiJ,iJkA->k', self.P2[7][2], Iabab.ooov)*dsoa)
        jitempb -= numpy.diag(0.5*1.0*einsum('KaIj,jIaK->I', self.P2[7][3], Iabab.oovo)*dsob)
        jitempa -= numpy.diag(0.5*1.0*einsum('KaIj,jIaK->j', self.P2[7][3], Iabab.oovo)*dsoa)
        jitempb -= numpy.diag(0.5*1.0*einsum('KaIj,jIaK->K', self.P2[7][3], Iabab.oovo)*dsob)

        jitempa -= numpy.diag(0.5*0.5*einsum('abij,ijab->i', self.P2[5][0], Ia.oovv)*dsoa)
        jitempb -= numpy.diag(0.5*0.5*einsum('abij,ijab->i', self.P2[5][1], Ib.oovv)*dsob)
        jitempa -= numpy.diag(0.5*1.0*einsum('aBiJ,iJaB->i', self.P2[5][2], Iabab.oovv)*dsoa)
        jitempb -= numpy.diag(0.5*1.0*einsum('aBiJ,iJaB->J', self.P2[5][2], Iabab.oovv)*dsob)

        batempa += numpy.diag(0.5*0.5*einsum('ijab,abij->a', self.P2[3][0], Ia.vvoo)*dsva)
        batempb += numpy.diag(0.5*0.5*einsum('ijab,abij->a', self.P2[3][1], Ib.vvoo)*dsvb)
        batempa += numpy.diag(0.5*1.0*einsum('iJaB,aBiJ->a', self.P2[3][2], Iabab.vvoo)*dsva)
        batempb += numpy.diag(0.5*1.0*einsum('iJaB,aBiJ->B', self.P2[3][2], Iabab.vvoo)*dsvb)

        batempa += numpy.diag(0.5*1.0*einsum('ciab,abci->a', self.P2[1][0], Ia.vvvo)*dsva)
        batempa += numpy.diag(0.5*0.5*einsum('ciab,abci->c', self.P2[1][0], Ia.vvvo)*dsva)
        batempb += numpy.diag(0.5*1.0*einsum('ciab,abci->a', self.P2[1][1], Ib.vvvo)*dsvb)
        batempb += numpy.diag(0.5*0.5*einsum('ciab,abci->c', self.P2[1][1], Ib.vvvo)*dsvb)
        batempa += numpy.diag(0.5*1.0*einsum('cIaB,aBcI->a', self.P2[1][2], Iabab.vvvo)*dsva)
        batempb += numpy.diag(0.5*1.0*einsum('cIaB,aBcI->B', self.P2[1][2], Iabab.vvvo)*dsvb)
        batempa += numpy.diag(0.5*1.0*einsum('cIaB,aBcI->c', self.P2[1][2], Iabab.vvvo)*dsva)
        batempb += numpy.diag(0.5*1.0*einsum('CiAb,bAiC->A', self.P2[1][3], Iabab.vvov)*dsvb)
        batempa += numpy.diag(0.5*1.0*einsum('CiAb,bAiC->b', self.P2[1][3], Iabab.vvov)*dsva)
        batempb += numpy.diag(0.5*1.0*einsum('CiAb,bAiC->C', self.P2[1][3], Iabab.vvov)*dsvb)

        batempa += numpy.diag(0.5*0.5*einsum('jkai,aijk->a', self.P2[6][0], Ia.vooo)*dsva)
        batempb += numpy.diag(0.5*0.5*einsum('jkai,aijk->a', self.P2[6][1], Ib.vooo)*dsvb)
        batempa += numpy.diag(0.5*1.0*einsum('jKaI,aIjK->a', self.P2[6][2], Iabab.vooo)*dsva)
        batempb += numpy.diag(0.5*1.0*einsum('JkAi,iAkJ->A', self.P2[6][3], Iabab.ovoo)*dsvb)

        batempa += numpy.diag(0.5*0.5*einsum('cdab,abcd->a', self.P2[0][0], Ia.vvvv)*dsva)
        batempa += numpy.diag(0.5*0.5*einsum('cdab,abcd->c', self.P2[0][0], Ia.vvvv)*dsva)
        batempb += numpy.diag(0.5*0.5*einsum('cdab,abcd->a', self.P2[0][1], Ib.vvvv)*dsvb)
        batempb += numpy.diag(0.5*0.5*einsum('cdab,abcd->c', self.P2[0][1], Ib.vvvv)*dsvb)
        batempa += numpy.diag(0.5*1.0*einsum('cDaB,aBcD->a', self.P2[0][2], Iabab.vvvv)*dsva)
        batempb += numpy.diag(0.5*1.0*einsum('cDaB,aBcD->B', self.P2[0][2], Iabab.vvvv)*dsvb)
        batempa += numpy.diag(0.5*1.0*einsum('cDaB,aBcD->c', self.P2[0][2], Iabab.vvvv)*dsva)
        batempb += numpy.diag(0.5*1.0*einsum('cDaB,aBcD->D', self.P2[0][2], Iabab.vvvv)*dsvb)

        batempa += numpy.diag(0.5*1.0*einsum('bjai,aibj->a', self.P2[4][0], Ia.vovo)*dsva)
        batempa += numpy.diag(0.5*1.0*einsum('bjai,aibj->b', self.P2[4][0], Ia.vovo)*dsva)
        batempb += numpy.diag(0.5*1.0*einsum('BJAI,AIBJ->A', self.P2[4][1], Ib.vovo)*dsvb)
        batempb += numpy.diag(0.5*1.0*einsum('BJAI,AIBJ->B', self.P2[4][1], Ib.vovo)*dsvb)
        batempa += numpy.diag(0.5*1.0*einsum('bJaI,aIbJ->a', self.P2[4][2], Iabab.vovo)*dsva)
        batempa += numpy.diag(0.5*1.0*einsum('bJaI,aIbJ->b', self.P2[4][2], Iabab.vovo)*dsva)
        batempb -= numpy.diag(0.5*1.0*einsum('bJAi,iAbJ->A', self.P2[4][3], Iabab.ovvo)*dsvb)
        batempa -= numpy.diag(0.5*1.0*einsum('bJAi,iAbJ->b', self.P2[4][3], Iabab.ovvo)*dsva)
        batempa -= numpy.diag(0.5*1.0*einsum('BjaI,aIjB->a', self.P2[4][4], Iabab.voov)*dsva)
        batempb -= numpy.diag(0.5*1.0*einsum('BjaI,aIjB->B', self.P2[4][4], Iabab.voov)*dsvb)
        batempb += numpy.diag(0.5*1.0*einsum('BjAi,iAjB->A', self.P2[4][5], Iabab.ovov)*dsvb)
        batempb += numpy.diag(0.5*1.0*einsum('BjAi,iAjB->B', self.P2[4][5], Iabab.ovov)*dsvb)

        batempa += numpy.diag(0.5*0.5*einsum('bcai,aibc->a', self.P2[2][0], Ia.vovv)*dsva)
        batempa += numpy.diag(0.5*1.0*einsum('bcai,aibc->b', self.P2[2][0], Ia.vovv)*dsva)
        batempb += numpy.diag(0.5*0.5*einsum('bcai,aibc->a', self.P2[2][1], Ib.vovv)*dsvb)
        batempb += numpy.diag(0.5*1.0*einsum('bcai,aibc->b', self.P2[2][1], Ib.vovv)*dsvb)
        batempa += numpy.diag(0.5*1.0*einsum('bCaI,aIbC->a', self.P2[2][2], Iabab.vovv)*dsva)
        batempa += numpy.diag(0.5*1.0*einsum('bCaI,aIbC->b', self.P2[2][2], Iabab.vovv)*dsva)
        batempb += numpy.diag(0.5*1.0*einsum('bCaI,aIbC->C', self.P2[2][2], Iabab.vovv)*dsvb)
        batempb += numpy.diag(0.5*1.0*einsum('BcAi,iAcB->A', self.P2[2][3], Iabab.ovvv)*dsvb)
        batempb += numpy.diag(0.5*1.0*einsum('BcAi,iAcB->B', self.P2[2][3], Iabab.ovvv)*dsvb)
        batempa += numpy.diag(0.5*1.0*einsum('BcAi,iAcB->c', self.P2[2][3], Iabab.ovvv)*dsva)

        batempa += numpy.diag(0.5*0.5*einsum('kaij,ijka->a', self.P2[7][0], Ia.ooov)*dsva)
        batempb += numpy.diag(0.5*0.5*einsum('kaij,ijka->a', self.P2[7][1], Ib.ooov)*dsvb)
        batempb += numpy.diag(0.5*1.0*einsum('kAiJ,iJkA->A', self.P2[7][2], Iabab.ooov)*dsvb)
        batempa += numpy.diag(0.5*1.0*einsum('KaIj,jIaK->a', self.P2[7][3], Iabab.oovo)*dsva)

        batempa += numpy.diag(0.5*0.5*einsum('abij,ijab->a', self.P2[5][0], Ia.oovv)*dsva)
        batempb += numpy.diag(0.5*0.5*einsum('abij,ijab->a', self.P2[5][1], Ib.oovv)*dsvb)
        batempa += numpy.diag(0.5*1.0*einsum('aBiJ,iJaB->a', self.P2[5][2], Iabab.oovv)*dsva)
        batempb += numpy.diag(0.5*1.0*einsum('aBiJ,iJaB->B', self.P2[5][2], Iabab.oovv)*dsvb)

        if self.athresh > 0.0:
            rdji[0][numpy.ix_(iocca,iocca)] += jitempa
            rdji[1][numpy.ix_(ioccb,ioccb)] += jitempb
            rdba[0][numpy.ix_(ivira,ivira)] += batempa
            rdba[1][numpy.ix_(ivirb,ivirb)] += batempb
        else:
            rdji[0] += jitempa
            rdji[1] += jitempb
            rdba[0] += batempa
            rdba[1] += batempb

        # orbital energy derivatives
        Gnew = self.G.copy()
        m = Gnew.shape[0]
        n = Gnew.shape[0]
        for i in range(m):
            for j in range(n):
                Gnew[i,j] *= (self.ti[j] - self.ti[i])

        ng = self.ti.shape[0]
        D1a = ea[:,None] - ea[None,:]
        D1b = eb[:,None] - eb[None,:]
        D2aa = ea[:,None,None,None] + ea[None,:,None,None] \
            - ea[None,None,:,None] - ea[None,None,None,:]
        D2ab = ea[:,None,None,None] + eb[None,:,None,None] \
            - ea[None,None,:,None] - eb[None,None,None,:]
        D2bb = eb[:,None,None,None] + eb[None,:,None,None] \
            - eb[None,None,:,None] - eb[None,None,None,:]
        if self.athresh > 0.0:
            D1a = D1a[numpy.ix_(ivira,iocca)]
            D1b = D1b[numpy.ix_(ivirb,ioccb)]
            D2aa = D2aa[numpy.ix_(ivira,ivira,iocca,iocca)]
            D2ab = D2ab[numpy.ix_(ivira,ivirb,iocca,ioccb)]
            D2bb = D2bb[numpy.ix_(ivirb,ivirb,ioccb,ioccb)]
        T1t,T2t = ft_cc_equations.uccsd_stanton(Fa,Fb,Ia,Ib,Iabab,self.T1[0],self.T1[1],
                self.T2[0],self.T2[1],self.T2[2],D1a,D1b,D2aa,D2ab,D2bb,self.ti,ng,Gnew)
        At1i = -(1.0/beta)*einsum('via,vai->vi',self.L1[0], T1t[0])
        At1I = -(1.0/beta)*einsum('via,vai->vi',self.L1[1], T1t[1])
        At1a = -(1.0/beta)*einsum('via,vai->va',self.L1[0], T1t[0])
        At1A = -(1.0/beta)*einsum('via,vai->va',self.L1[1], T1t[1])
        At2i = -(1.0/beta)*0.25*einsum('vijab,vabij->vi',self.L2[0], T2t[0])
        At2I = -(1.0/beta)*0.25*einsum('vijab,vabij->vi',self.L2[2], T2t[2])
        At2j = -(1.0/beta)*0.25*einsum('vijab,vabij->vj',self.L2[0], T2t[0])
        At2J = -(1.0/beta)*0.25*einsum('vijab,vabij->vj',self.L2[2], T2t[2])
        At2a = -(1.0/beta)*0.25*einsum('vijab,vabij->va',self.L2[0], T2t[0])
        At2A = -(1.0/beta)*0.25*einsum('vijab,vabij->va',self.L2[2], T2t[2])
        At2b = -(1.0/beta)*0.25*einsum('vijab,vabij->vb',self.L2[0], T2t[0])
        At2B = -(1.0/beta)*0.25*einsum('vijab,vabij->vb',self.L2[2], T2t[2])
        At2i -= (1.0/beta)*einsum('viJaB,vaBiJ->vi',self.L2[1], T2t[1])
        At2J -= (1.0/beta)*einsum('viJaB,vaBiJ->vJ',self.L2[1], T2t[1])
        At2a -= (1.0/beta)*einsum('viJaB,vaBiJ->va',self.L2[1], T2t[1])
        At2B -= (1.0/beta)*einsum('viJaB,vaBiJ->vB',self.L2[1], T2t[1])
        if self.athresh > 0.0:
            rdji[0][numpy.ix_(iocca,iocca)] -= numpy.diag(einsum('vi,v->i',At1i+At2i+At2j,self.g))
            rdji[1][numpy.ix_(ioccb,ioccb)] -= numpy.diag(einsum('vi,v->i',At1I+At2I+At2J,self.g))
            rdba[0][numpy.ix_(ivira,ivira)] += numpy.diag(einsum('va,v->a',At1a+At2a+At2b,self.g))
            rdba[1][numpy.ix_(ivirb,ivirb)] += numpy.diag(einsum('va,v->a',At1A+At2A+At2B,self.g))
        else:
            rdji[0] -= numpy.diag(einsum('vi,v->i',At1i+At2i+At2j,self.g))
            rdji[1] -= numpy.diag(einsum('vi,v->i',At1I+At2I+At2J,self.g))
            rdba[0] += numpy.diag(einsum('va,v->a',At1a+At2a+At2b,self.g))
            rdba[1] += numpy.diag(einsum('va,v->a',At1A+At2A+At2B,self.g))

        self.r1rdm = [rdji[0] + rdba[0],rdji[1] + rdba[1]]

    def full_1rdm(self, relax=False):
        beta = self.beta
        mu = self.mu
        if self.sys.orbtype == 'u':
            if relax:
                if self.r1rdm is None:
                    self._urel_ft_1rdm()
                return [self.r1rdm[0] + (self.n1rdm[0] - numpy.diag(self.n1rdm[0].diagonal())),
                        self.r1rdm[1] + (self.n1rdm[1] - numpy.diag(self.n1rdm[1].diagonal()))]
            if self.n1rdm is None:
                self._u_ft_1rdm()
            ea,eb = self.sys.u_energies_tot()
            foa = ft_utils.ff(beta, ea, mu)
            fva = ft_utils.ffv(beta, ea, mu)
            fob = ft_utils.ff(beta, eb, mu)
            fvb = ft_utils.ffv(beta, eb, mu)
            rdm1 = [self.n1rdm[0].copy(), self.n1rdm[1].copy()]
            if self.athresh > 0.0:
                athresh = self.athresh
                focca = [x for x in foa if x > athresh]
                fvira = [x for x in fva if x > athresh]
                iocca = [i for i,x in enumerate(foa) if x > athresh]
                ivira = [i for i,x in enumerate(fva) if x > athresh]
                foccb = [x for x in fob if x > athresh]
                fvirb = [x for x in fvb if x > athresh]
                ioccb = [i for i,x in enumerate(fob) if x > athresh]
                ivirb = [i for i,x in enumerate(fvb) if x > athresh]
                nocca = len(focca)
                rdm1[0][numpy.ix_(iocca,iocca)] += numpy.diag(focca)
                rdm1[1][numpy.ix_(ioccb,ioccb)] += numpy.diag(foccb)
            else:
                rdm1[0] += numpy.diag(foa)
                rdm1[1] += numpy.diag(fob)
            return rdm1
        elif self.sys.orbtype == 'g':
            if relax:
                if self.r1rdm is None:
                    self._grel_ft_1rdm()
                return self.r1rdm + (self.n1rdm - numpy.diag(self.n1rdm.diagonal()))
            if self.n1rdm is None:
                self._g_ft_1rdm()
            en = self.sys.g_energies_tot()
            fo = ft_utils.ff(beta, en, mu)
            fv = ft_utils.ffv(beta, en, mu)
            rdm1 = self.n1rdm.copy()
            if self.athresh > 0.0:
                athresh = self.athresh
                focc = [x for x in fo if x > athresh]
                fvir = [x for x in fv if x > athresh]
                iocc = [i for i,x in enumerate(fo) if x > athresh]
                ivir = [i for i,x in enumerate(fv) if x > athresh]
                rdm1[numpy.ix_(iocc,iocc)] += numpy.diag(fo)
            else:
                rem1 += numpy.diag(fo)
            return rdm1
        else:
            raise Exception("orbital type " + self.sys.orbtype + " is not implemented for 1rdm")

    def full_2rdm(self, relax=False):
        assert(self.beta == self.beta_max)
        if relax:
            raise Exception("Rexalex 2-RDM is not implemented")
        beta = self.beta
        mu = self.mu
        if self.sys.orbtype == 'u':
            if self.n1rdm is None:
                self._u_ft_1rdm()
            if self.n2rdm is None:
                self._u_ft_2rdm()
            ea,eb = self.sys.u_energies_tot()
            foa = ft_utils.ff(beta, ea, mu)
            fva = ft_utils.ffv(beta, ea, mu)
            fob = ft_utils.ff(beta, eb, mu)
            fvb = ft_utils.ffv(beta, eb, mu)
            rdm2 = [self.n2rdm[0].copy(), self.n2rdm[1].copy(), self.n2rdm[2].copy()]
            if self.athresh > 0.0:
                athresh = self.athresh
                focca = [x for x in foa if x > athresh]
                fvira = [x for x in fva if x > athresh]
                iocca = [i for i,x in enumerate(foa) if x > athresh]
                ivira = [i for i,x in enumerate(fva) if x > athresh]
                ialla = [i for i in range(len(foa))]
                foccb = [x for x in fob if x > athresh]
                fvirb = [x for x in fvb if x > athresh]
                ioccb = [i for i,x in enumerate(fob) if x > athresh]
                ivirb = [i for i,x in enumerate(fvb) if x > athresh]
                iallb = [i for i in range(len(fob))]
                nocca = len(focca)
                rdm2[0][numpy.ix_(iocca,iocca,iocca,iocca)] += numpy.einsum('pr,qs->pqrs',numpy.diag(foa),numpy.diag(foa))
                rdm2[0][numpy.ix_(iocca,iocca,iocca,iocca)] -= numpy.einsum('pr,qs->pqsr',numpy.diag(foa),numpy.diag(foa))
                rdm2[0][numpy.ix_(iocca,ialla,iocca,ialla)] += 0.5*numpy.einsum('pr,qs->pqrs',numpy.diag(foa),self.n1rdm[0])
                rdm2[0][numpy.ix_(iocca,ialla,ialla,iocca)] -= 0.5*numpy.einsum('pr,qs->pqsr',numpy.diag(foa),self.n1rdm[0])
                rdm2[0][numpy.ix_(ialla,iocca,ialla,iocca)] += 0.5*numpy.einsum('pr,qs->pqrs',self.n1rdm[0],numpy.diag(foa))
                rdm2[0][numpy.ix_(ialla,iocca,iocca,ialla)] -= 0.5*numpy.einsum('pr,qs->pqsr',self.n1rdm[0],numpy.diag(foa))

                rdm2[1][numpy.ix_(ioccb,ioccb,ioccb,ioccb)] += numpy.einsum('pr,qs->pqrs',numpy.diag(fob),numpy.diag(fob))
                rdm2[1][numpy.ix_(ioccb,ioccb,ioccb,ioccb)] -= numpy.einsum('pr,qs->pqsr',numpy.diag(fob),numpy.diag(fob))
                rdm2[1][numpy.ix_(ioccb,iallb,ioccb,iallb)] += 0.5*numpy.einsum('pr,qs->pqrs',numpy.diag(fob),self.n1rdm[1])
                rdm2[1][numpy.ix_(ioccb,iallb,iallb,ioccb)] -= 0.5*numpy.einsum('pr,qs->pqsr',numpy.diag(fob),self.n1rdm[1])
                rdm2[1][numpy.ix_(iallb,ioccb,iallb,ioccb)] += 0.5*numpy.einsum('pr,qs->pqrs',self.n1rdm[1],numpy.diag(fob))
                rdm2[1][numpy.ix_(iallb,ioccb,ioccb,iallb)] -= 0.5*numpy.einsum('pr,qs->pqsr',self.n1rdm[1],numpy.diag(fob))

                rdm2[2][numpy.ix_(iocca,ioccb,iocca,ioccb)] += numpy.einsum('pr,qs->pqrs',numpy.diag(foa),numpy.diag(fob))
                rdm2[2][numpy.ix_(iocca,iallb,iocca,iallb)] += 0.5*numpy.einsum('pr,qs->pqrs',numpy.diag(foa),self.n1rdm[1])
                rdm2[2][numpy.ix_(ialla,ioccb,ialla,ioccb)] += 0.5*numpy.einsum('pr,qs->pqrs',self.n1rdm[0],numpy.diag(fob))
            else:
                rdm2[0] += numpy.einsum('pr,qs->pqrs',numpy.diag(foa),numpy.diag(foa))
                rdm2[0] -= numpy.einsum('pr,qs->pqsr',numpy.diag(foa),numpy.diag(foa))
                rdm2[0] += 0.5*numpy.einsum('pr,qs->pqrs',numpy.diag(foa),self.n1rdm[0])
                rdm2[0] -= 0.5*numpy.einsum('pr,qs->pqsr',numpy.diag(foa),self.n1rdm[0])
                rdm2[0] += 0.5*numpy.einsum('pr,qs->pqrs',self.n1rdm[0],numpy.diag(foa))
                rdm2[0] -= 0.5*numpy.einsum('pr,qs->pqsr',self.n1rdm[0],numpy.diag(foa))

                rdm2[1] += numpy.einsum('pr,qs->pqrs',numpy.diag(fob),numpy.diag(fob))
                rdm2[1] -= numpy.einsum('pr,qs->pqsr',numpy.diag(fob),numpy.diag(foa))
                rdm2[1] += 0.5*numpy.einsum('pr,qs->pqrs',numpy.diag(fob),self.n1rdm[1])
                rdm2[1] -= 0.5*numpy.einsum('pr,qs->pqsr',numpy.diag(fob),self.n1rdm[1])
                rdm2[1] += 0.5*numpy.einsum('pr,qs->pqrs',self.n1rdm[1],numpy.diag(fob))
                rdm2[1] -= 0.5*numpy.einsum('pr,qs->pqsr',self.n1rdm[1],numpy.diag(fob))

                rdm2[2] += numpy.einsum('pr,qs->pqrs',numpy.diag(foa),numpy.diag(fob))
                rdm2[2] += 0.5*numpy.einsum('pr,qs->pqrs',numpy.diag(foa),self.n1rdm[1])
                rdm2[2] += 0.5*numpy.einsum('pr,qs->pqrs',self.n1rdm[0],numpy.diag(fob))
            return rdm2
        elif self.sys.orbtype == 'g':
            if self.n1rdm is None:
                self._g_ft_1rdm()
            if self.n2rdm is None:
                self._g_ft_2rdm()
            en = self.sys.g_energies_tot()
            fo = ft_utils.ff(beta, en, mu)
            fv = ft_utils.ffv(beta, en, mu)
            rdm2 = self.n2rdm.copy()
            if self.athresh > 0.0:
                athresh = self.athresh
                focc = [x for x in fo if x > athresh]
                fvir = [x for x in fv if x > athresh]
                iocc = [i for i,x in enumerate(fo) if x > athresh]
                ivir = [i for i,x in enumerate(fv) if x > athresh]
                iall = [i for i in range(len(fo))]
                rdm2[numpy.ix_(iocc,iocc,iocc,iocc)] += numpy.einsum('pr,qs->pqrs',numpy.diag(fo),numpy.diag(fo))
                rdm2[numpy.ix_(iocc,iocc,iocc,iocc)] -= numpy.einsum('pr,qs->pqsr',numpy.diag(fo),numpy.diag(fo))
                rdm2[numpy.ix_(iocc,iall,iocc,iall)] += 0.5*numpy.einsum('pr,qs->pqrs',numpy.diag(fo),self.n1rdm)
                rdm2[numpy.ix_(iocc,iall,iall,iocc)] -= 0.5*numpy.einsum('pr,qs->pqsr',numpy.diag(fo),self.n1rdm)
                rdm2[numpy.ix_(iall,iocc,iall,iocc)] += 0.5*numpy.einsum('pr,qs->pqrs',self.n1rdm,numpy.diag(fo))
                rdm2[numpy.ix_(iall,iocc,iocc,iall)] -= 0.5*numpy.einsum('pr,qs->pqsr',self.n1rdm,numpy.diag(fo))
            else:
                rdm2 += numpy.einsum('pr,qs->pqrs',numpy.diag(fo),numpy.diag(fo))
                rdm2 -= numpy.einsum('pr,qs->pqsr',numpy.diag(fo),numpy.diag(fo))
                rdm2 += 0.5*numpy.einsum('pr,qs->pqrs',numpy.diag(fo),self.n1rdm)
                rdm2 -= 0.5*numpy.einsum('pr,qs->pqsr',numpy.diag(fo),self.n1rdm)
                rdm2 += 0.5*numpy.einsum('pr,qs->pqrs',self.n1rdm,numpy.diag(fo))
                rdm2 -= 0.5*numpy.einsum('pr,qs->pqsr',self.n1rdm,numpy.diag(fo))
            return rdm2
        else:
            raise Exception("orbital type " + self.sys.orbtype + " is not implemented for 1rdm")
