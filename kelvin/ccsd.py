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
#einsum = einsum

class ccsd(object):
    """Coupled cluster singles and doubles (CCSD) driver.

    Attributes:
        sys: System object.
        T (float): Temperature.    
        mu (float): Chemical potential.
        iprint (int): Print level.
        singles (bool): Include singles (False -> CCD).
        econv (float): Energy difference convergence threshold.
        max_iter (int): Max number of iterations.
        damp (float): Mixing parameter to damp iterations.
        ngrid (int): Number of grid points.
        realtime (bool): Force time-dependent formulation for zero T
        athresh (float): Threshold for ignoring small occupations
        T1: Saved T1 amplitudes
        T2: Saved T2 amplitudes
        L1: Saved L1 amplitudes
        L2: Saved L2 amplitudes
    """
    def __init__(self, sys, T=0.0, mu=0.0, iprint=0,
        singles=True, econv=1e-8, max_iter=40, 
        damp=0.0, ngrid=10, realtime=False, athresh=0.0):

        self.T = T
        self.mu = mu
        self.finite_T = False if T == 0 else True
        self.iprint = iprint
        self.singles = singles
        self.econv = econv
        self.max_iter = max_iter
        self.damp = damp
        self.ngrid = ngrid
        self.realtime = realtime
        self.athresh = athresh
        if self.finite_T:
            self.realtime = True
        if not sys.verify(self.T,self.mu):
            raise Exception("Sytem temperature inconsistent with CC temp")
        self.sys = sys
        self.T1 = None
        self.T2 = None
        self.L1 = None
        self.L2 = None

    def run(self,T1=None,T2=None):
        """Run CCSD calculation."""
        if self.finite_T:
            if self.iprint > 0:
                print('Running CCSD at an electronic temperature of %f K'
                    % ft_utils.HtoK(self.T))
            if self.athresh > 0.0:
                return self._ft_ccsd_active(T1in=T1,T2in=T2)
            else:
                if self.sys.has_u():
                    return self._ft_uccsd(T1in=T1,T2in=T2)
                else:
                    return self._ft_ccsd(T1in=T1,T2in=T2)
        else:
            if self.iprint > 0:
                print('Running CCSD at zero Temperature')
            if self.realtime:
                return self._ccsd_rt()
            else:
                if self.sys.has_u():
                    return self._uccsd(T1in=T1,T2in=T2)
                else:
                    return self._ccsd()

    def compute_ESN(self,L1=None,L2=None):
        """Compute energy, entropy, particle number."""
        if not self.finite_T:
            N = self.sys.g_energies()[0].shape[0]
            print("T = 0: ")
            print('  E = {}'.format(self.Etot))
            print('  S = {}'.format(0.0))
            print('  N = {}'.format(N))
        else:
            if self.L1 is None:
                if self.athresh > 0.0:
                    raise Exception("Lambda equations for A-FT-CCSD aren't implemented")
                if self.sys.has_u():
                    self._ft_uccsd_lambda(L1=L1,L2=L2)
                else:
                    self._ft_ccsd_lambda(L1=L1,L2=L2)
                    self._ft_1rdm()
                    self._ft_2rdm()
            if self.sys.has_u():
                self._u_ft_ESN(L1,L2)
            else:
                self._g_ft_ESN(L1,L2)

    def _g_ft_ESN(self,L1=None,L2=None):
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
            dvec = mu - en
            B1,Bcc = self._g_nocc_deriv(dvec)

            # compute other contributions to CC derivative
            Bcc -= self.Gcc/(beta)
            dg,dG = self._g_nocc_gderiv()
            Bcc += dG + dg

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

    def _u_ft_ESN(self,L1=None,L2=None):
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
            dveca = mu - ea
            dvecb = mu - eb
            B1,Bcc = self._u_nocc_deriv(dveca,dvecb)

            # compute other contributions to CC derivative
            Bcc -= self.Gcc/(beta)
            dg,dG = self._u_nocc_gderiv()
            Bcc += dG + dg

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
        thresh = self.econv
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
            if numpy.abs(E - Eold) < thresh:
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
        thresh = self.econv
        max_iter = self.max_iter
        i = 0
        Eold = 1000000
        alpha = self.damp
        while i < max_iter and not converged:
            (T1a,T1b),(T2aa,T2ab,T2bb) = cc_equations.uccsd_stanton(Fa, Fb, Ia, Ib, Iabab, T1olds, T2olds)
            T1a = einsum('ai,ia->ai',T1a,Dova)
            T1b = einsum('ai,ia->ai',T1b,Dovb)
            T2aa = einsum('abij,ijab->abij',T2aa,Doovvaa)
            T2ab = einsum('abij,ijab->abij',T2ab,Doovvab)
            T2bb = einsum('abij,ijab->abij',T2bb,Doovvbb)
            E = cc_energy.ucc_energy((T1a,T1b),(T2aa,T2ab,T2bb),
                    Fa.ov,Fb.ov,Ia.oovv,Ib.oovv,Iabab.oovv)
            if self.iprint > 0:
                print(' %2d  %.10f' % (i+1,E))
            i = i + 1
            if numpy.abs(E - Eold) < thresh:
                converged = True
            Eold = E
            T1a = alpha*T1olds[0] + (1.0 - alpha)*T1a
            T1b = alpha*T1olds[1] + (1.0 - alpha)*T1b
            T2aa = alpha*T2olds[0] + (1.0 - alpha)*T2aa
            T2ab = alpha*T2olds[1] + (1.0 - alpha)*T2ab
            T2bb = alpha*T2olds[2] + (1.0 - alpha)*T2bb

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
        thresh = self.econv
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
            if res1 + res2 < thresh:
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
        thresh = self.econv
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
            if res1 + res2 < thresh:
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

    def _ccsd_rt(self,T1in=None,T2in=None):
        """Solve CCSD equations at zero temperature with time-dependent 
        formulation.
        """
        # create energies in spin-orbital basis
        eo,ev = self.sys.g_energies()
        no = eo.shape[0]
        nv = ev.shape[0]
        Dvo = (ev[:,None] - eo[None,:])
        Dvvoo = (ev[:,None,None,None] + ev[None,:,None,None]
            - eo[None,None,:,None] - eo[None,None,None,:])

        # get time-grid
        beta_max = 80.0
        ng = self.ngrid
        delta = beta_max/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng,delta)
        g = quadrature.get_gint(ng, delta)

        # get HF energy
        En = self.sys.const_energy()
        E0 = zt_mp.mp0(eo) + En
        E1 = self.sys.get_mp1()
        Ehf = E0 + E1

        # get Fock matrix
        F = self.sys.g_fock()
        F.oo = F.oo - numpy.diag(eo) # subtract diagonal
        F.vv = F.vv - numpy.diag(ev) # subtract diagonal

        # get ERIs
        I = self.sys.g_aint()

        # get MP2 T-amplitudes
        Id = numpy.ones(ng)
        if T1in is not None and T2in is not None:
            T1old = T1in if self.singles else numpy.zeros((ng,n,n))
            T2old = T2in
        else:
            G = quadrature.get_G(ng,delta)
            if self.singles:
                T1old = -einsum('v,ai->vai',Id,F.vo)
                T1old = quadrature.int_tbar1(ng,T1old,ti,Dvo,G)
            else:
                T1old = numpy.zeros((ng,nr,no))
            T2old = -einsum('v,abij->vabij',Id,I.vvoo)
            T2old = quadrature.int_tbar2(ng,T2old,ti,Dvvoo,G)
        E2 = ft_cc_energy.ft_cc_energy(T1old,T2old,
            F.ov,I.oovv,ti,g,beta_max,Qterm=False)
        if self.iprint > 0:
            print('MP2 energy: {:.10f}'.format(E2))

        # run CC iterations
        method = "CCSD" if self.singles else "CCD"
        conv_options = {"econv":self.econv, "max_iter":self.max_iter, "damp":self.damp}
        Ecc,T1,T2 = cc_utils.ft_cc_iter(method, T1old, T2old, F, I, Dvo, Dvvoo,
                g, G, beta_max, ng, ti, self.iprint, conv_options)

        # save T amplitudes
        self.T1 = T1
        self.T2 = T2

        return (Ecc+Ehf,Ecc)

    def _ft_ccsd(self,T1in=None,T2in=None):
        """Solve finite temperature coupled cluster equations."""
        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu

        # get time-grid
        ng = self.ngrid
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng,delta)
        g = quadrature.get_gint(ng, delta)

        # get orbital energies
        en = self.sys.g_energies_tot()

        # compute requisite memory
        n = en.shape[0]
        mem1e = 6*n*n + 3*ng*n*n
        mem2e = 3*n*n*n*n + 5*ng*n*n*n*n
        mem_mb = (mem1e + mem2e)*8.0/1024.0/1024.0
        if self.iprint > 0:
            print('  FT-CCSD will use %f mb' % mem_mb)

        # get 0th and 1st order contributions
        En = self.sys.const_energy()
        g0 = ft_utils.GP0(beta, en, mu)
        E0 = ft_mp.mp0(g0) + En
        E1 = self.sys.get_mp1()
        E01 = E0 + E1

        # get scaled integrals
        F,I = cc_utils.get_ft_integrals(self.sys, en, beta, mu)

        # get energy differences
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
                - en[None,None,:,None] - en[None,None,None,:]

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
            G = quadrature.get_G(ng,delta)
            T1old = quadrature.int_tbar1(ng,T1old,ti,D1,G)
            T2old = quadrature.int_tbar2(ng,T2old,ti,D2,G)
        E2 = ft_cc_energy.ft_cc_energy(T1old,T2old,
            F.ov,I.oovv,ti,g,beta,Qterm=False)
        if self.iprint > 0:
            print('MP2 Energy: {:.10f}'.format(E2))

        # run CC iterations
        method = "CCSD" if self.singles else "CCD"
        conv_options = {"econv":self.econv, "max_iter":self.max_iter, "damp":self.damp}
        Eccn,T1,T2 = cc_utils.ft_cc_iter(method, T1old, T2old, F, I, D1, D2, g, G,
                beta, ng, ti, self.iprint, conv_options)

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
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng,delta)
        g = quadrature.get_gint(ng, delta)

        # get orbital energies
        ea,eb = self.sys.u_energies_tot()
        na = ea.shape[0]
        nb = eb.shape[0]

        # compute requisite memory
        n = na + nb
        mem1e = 6*n*n + 3*ng*n*n
        mem2e = 3*n*n*n*n + 5*ng*n*n*n*n
        mem_mb = (mem1e + mem2e)*8.0/1024.0/1024.0
        if self.iprint > 0:
            print('  FT-CCSD will use %f mb' % mem_mb)

        # get 0th and 1st order contributions
        En = self.sys.const_energy()
        g0 = ft_utils.uGP0(beta, ea, eb, mu)
        E0 = ft_mp.ump0(g0[0],g0[1]) + En
        E1 = self.sys.get_mp1()
        E01 = E0 + E1

        # get scaled integrals
        Fa,Fb,Ia,Ib,Iabab = cc_utils.get_uft_integrals(self.sys, ea, eb, beta, mu)

        # get energy differences
        D1a = ea[:,None] - ea[None,:]
        D1b = eb[:,None] - eb[None,:]
        D2aa = ea[:,None,None,None] + ea[None,:,None,None] \
                - ea[None,None,:,None] - ea[None,None,None,:]
        D2ab = ea[:,None,None,None] + eb[None,:,None,None] \
                - ea[None,None,:,None] - eb[None,None,None,:]
        D2bb = eb[:,None,None,None] + eb[None,:,None,None] \
                - eb[None,None,:,None] - eb[None,None,None,:]

        # get MP2 T-amplitudes
        if T1in is not None and T2in is not None:
            T1aold = T1in[0] if self.singles else numpy.zeros((ng, na, na))
            T1bold = T1in[1] if self.singles else numpy.zeros((ng, na, na))
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
            G = quadrature.get_G(ng,delta)
            T1aold = quadrature.int_tbar1(ng,T1aold,ti,D1a,G)
            T1bold = quadrature.int_tbar1(ng,T1bold,ti,D1b,G)
            T2aaold = quadrature.int_tbar2(ng,T2aaold,ti,D2aa,G)
            T2abold = quadrature.int_tbar2(ng,T2abold,ti,D2ab,G)
            T2bbold = quadrature.int_tbar2(ng,T2bbold,ti,D2bb,G)

        # MP2 energy
        E2 = ft_cc_energy.ft_ucc_energy(T1aold,T1bold,T2aaold,T2abold,T2bbold,
            Fa.ov,Fb.ov,Ia.oovv,Ib.oovv,Iabab.oovv,ti,g,beta,Qterm=False)
        if self.iprint > 0:
            print('MP2 Energy: {:.10f}'.format(E2))


        # run CC iterations
        method = "CCSD" if self.singles else "CCD"
        conv_options = {"econv":self.econv, "max_iter":self.max_iter, "damp":self.damp}
        Eccn,T1,T2 = cc_utils.ft_ucc_iter(method, T1aold, T1bold, T2aaold, T2abold, T2bbold,
                Fa, Fb, Ia, Ib, Iabab, D1a, D1b, D2aa, D2ab, D2bb,
                g, G, beta, ng, ti, self.iprint, conv_options)

        # save T amplitudes
        self.T1 = T1
        self.T2 = T2
        self.G0 = E0
        self.G1 = E1
        self.Gcc = Eccn
        self.Gtot = E0 + E1 + Eccn

        return (Eccn+E01,Eccn)

    def _ft_ccsd_active(self,T1in=None,T2in=None):
        """Solve finite temperature coupled cluster equations with some 
        frozen occupations.
        """
        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu

        # get time-grid
        ng = self.ngrid
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng,delta)
        g = quadrature.get_gint(ng, delta)

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)

        # compute active space
        n = en.shape[0]
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

        # compute requisite memory
        mem1e = 6*n*n + 3*ng*n*n
        mem2e = 3*n*n*n*n + 5*ng*n*n*n*n
        mem_mb = (mem1e + mem2e)*8.0/1024.0/1024.0
        if self.iprint > 0:
            print('  FT-CCSD will use %f mb' % mem_mb)

        # get 0th and 1st order contributions
        En = self.sys.const_energy()
        g0 = ft_utils.GP0(beta, en, mu)
        E0 = ft_mp.mp0(g0) + En
        E1 = self.sys.get_mp1()
        E01 = E0 + E1

        # get scaled active space integrals
        F,I = cc_utils.get_ft_active_integrals(self.sys, en, focc, fvir, iocc, ivir)

        # get exponentials
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
                - en[None,None,:,None] - en[None,None,None,:]
        D1 = D1[numpy.ix_(ivir,iocc)]
        D2 = D2[numpy.ix_(ivir,ivir,iocc,iocc)]

        # get MP2 T-amplitudes
        if T1in is not None and T2in is not None:
            T1old = T1in if self.singles else numpy.zeros((ng,n,n))
            T2old = T2in
        else:
            if self.singles:
                Id = numpy.ones((ng))
                T1old = -einsum('v,ai->vai',Id,F.vo)
            else:
                T1old = numpy.zeros((ng,nvir,nocc))
            Id = numpy.ones((ng))
            T2old = -einsum('v,abij->vabij',Id,I.vvoo)
            G = quadrature.get_G(ng,delta)
            T1old = quadrature.int_tbar1(ng,T1old,ti,D1,G)
            T2old = quadrature.int_tbar2(ng,T2old,ti,D2,G)
        E2 = ft_cc_energy.ft_cc_energy(T1old,T2old,
            F.ov,I.oovv,ti,g,beta,Qterm=False)
        if self.iprint > 0:
            print('MP2 Energy: {:.10f}'.format(E2))

        # run CC iterations
        method = "CCSD" if self.singles else "CCD"
        conv_options = {"econv":self.econv, "max_iter":self.max_iter, "damp":self.damp}
        Eccn,T1,T2 = cc_utils.ft_cc_iter(method, T1old, T2old, F, I, D1, D2,
                g, G, beta, ng, ti, self.iprint, conv_options)

        # save T amplitudes
        self.T1 = T1
        self.T2 = T2

        return (Eccn+E01,Eccn)

    def _ft_ccsd_lambda(self, L1=None, L2=None):
        """Solve FT-CCSD Lambda equations."""
        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu

        # get time-grid
        ng = self.ngrid
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng,delta)
        g = quadrature.get_gint(ng, delta)

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()

        # compute requisite memory
        n = en.shape[0]
        mem1e = 6*n*n + 3*ng*n*n
        mem2e = 3*n*n*n*n + 5*ng*n*n*n*n
        mem_mb = (mem1e + mem2e)*8.0/1024.0/1024.0
        if self.iprint > 0:
            print('  FT-CCSD will use %f mb' % mem_mb)

        En = self.sys.const_energy()
        g0 = ft_utils.GP0(beta, en, mu)
        E0 = ft_mp.mp0(g0) + En

        # get FT Fock matrix
        E1 = self.sys.get_mp1()
        E01 = E0 + E1

        # get scaled integrals
        F,I = cc_utils.get_ft_integrals(self.sys, en, beta, mu)

        # get energy differences
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
                - en[None,None,:,None] - en[None,None,None,:]

        if L2 is None:
            # Use T^{\dagger} as a guess for Lambda
            if self.singles:
                L1old = numpy.transpose(self.T1,(0,2,1))
            else:
                L1old = numpy.zeros(self.T1.shape)
            L2old = numpy.transpose(self.T2,(0,3,4,1,2))
        else:
            L2old = L2
            if L1 is None:
                L1old = numpy.zeros(self.T1.shape)
            else:
                L1old = L1

        # run lambda iterations
        conv_options = {"econv":self.econv, "max_iter":self.max_iter, "damp":self.damp}
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
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng,delta)
        g = quadrature.get_gint(ng, delta)

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        ea,eb = self.sys.u_energies_tot()
        na = ea.shape[0]
        nb = eb.shape[0]

        # compute requisite memory
        n = en.shape[0]
        mem1e = 6*n*n + 3*ng*n*n
        mem2e = 3*n*n*n*n + 5*ng*n*n*n*n
        mem_mb = (mem1e + mem2e)*8.0/1024.0/1024.0
        if self.iprint > 0:
            print('  FT-CCSD will use %f mb' % mem_mb)

        En = self.sys.const_energy()
        g0 = ft_utils.uGP0(beta, ea, eb, mu)
        E0 = ft_mp.ump0(g0[0],g0[1]) + En
        E1 = self.sys.get_mp1()
        E01 = E0 + E1

        # get scaled integrals
        Fa,Fb,Ia,Ib,Iabab = cc_utils.get_uft_integrals(self.sys, ea, eb, beta, mu)

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

        if L2 is None:
            # Use T^{\dagger} as a guess for Lambda
            if self.singles:
                L1aold = numpy.transpose(T1aold,(0,2,1))
                L1bold = numpy.transpose(T1bold,(0,2,1))
            else:
                L1aold = numpy.zeros((ng,na,na))
                L1bold = numpy.zeros((ng,nb,nb))
            L2aaold = numpy.transpose(T2aaold,(0,3,4,1,2))
            L2abold = numpy.transpose(T2abold,(0,3,4,1,2))
            L2bbold = numpy.transpose(T2bbold,(0,3,4,1,2))
        else:
            L2aaold = L2[0]
            L2abold = L2[1]
            L2bbold = L2[2]
            if L1 is None:
                L1aold = numpy.zeros((ng,na,na))
                L1bold = numpy.zeros((ng,nb,nb))
            else:
                L1aold = L1[0]
                L1bold = L1[1]

        # run lambda iterations
        conv_options = {"econv":self.econv, "max_iter":self.max_iter, "damp":self.damp}
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
        n = fo.shape[0]

        # first order contributions
        der1 = self.sys.g_d_mp1(dvec)

        # get time-grid
        ng = self.ngrid
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng,delta)
        g = quadrature.get_gint(ng, delta)

        # get exponentials
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
            - en[None,None,:,None] - en[None,None,None,:]

        F,I = cc_utils.get_ft_d_integrals(self.sys, en, fo, fv, dvec)
        #Eterm = ft_cc_energy.ft_cc_energy(self.T1,self.T2,
        #    F.ov,I.oovv,ti,g,beta)

        #Fvo = F.vo.copy()
        #Fvv = F.vv.copy()
        #F.vo = numpy.zeros(Fvo.shape)
        #F.vv = numpy.zeros(Fvv.shape)
        #T1temp,T2temp = ft_cc_equations.ccsd_stanton(F,I,self.T1,self.T2,
        #        D1,D2,ti,ng,G)

        #print(numpy.linalg.norm(Fvv))
        #A1 = (1.0/beta)*einsum('via,vai->v',self.L1, T1temp)
        #A2 = (1.0/beta)*0.25*einsum('vijab,vabij->v',self.L2, T2temp)
        #A1 += (1.0/beta)*einsum('via,ai->v',self.dia,Fvo)
        #A1 += (1.0/beta)*einsum('vba,ab->v',self.dba,Fvv)

        #g = ft_utils.get_gint(ng, delta)
        #A1g = einsum('v,v->',A1,g)
        #A2g = einsum('v,v->',A2,g)
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
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng,delta)
        g = quadrature.get_gint(ng, delta)

        # get exponentials
        D1a = ea[:,None] - ea[None,:]
        D1b = eb[:,None] - eb[None,:]
        D2aa = ea[:,None,None,None] + ea[None,:,None,None] \
            - ea[None,None,:,None] - ea[None,None,None,:]
        D2ab = ea[:,None,None,None] + eb[None,:,None,None] \
            - ea[None,None,:,None] - eb[None,None,None,:]
        D2bb = eb[:,None,None,None] + eb[None,:,None,None] \
            - eb[None,None,:,None] - eb[None,None,None,:]

        Fa,Fb,Ia,Ib,Iabab = cc_utils.u_ft_d_integrals(self.sys, ea, eb, foa, fob, fva, fvb, dveca, dvecb)
        T1aold,T1bold = self.T1
        T2aaold,T2abold,T2bbold = self.T2
        L1aold,L1bold = self.L1
        L2aaold,L2abold,L2bbold = self.L2

        Eterm = ft_cc_energy.ft_ucc_energy(T1aold,T1bold,T2aaold,T2abold,T2bbold,
            Fa.ov,Fb.ov,Ia.oovv,Ib.oovv,Iabab.oovv,ti,g,beta)

        T1t,T2t = ft_cc_equations.uccsd_stanton(Fa,Fb,Ia,Ib,Iabab,T1aold,T1bold,
                T2aaold,T2abold,T2bbold,D1a,D1b,D2aa,D2ab,D2bb,ti,ng,G)

        A1a = (1.0/beta)*einsum('via,vai->v',L1aold, T1t[0])
        A1b = (1.0/beta)*einsum('via,vai->v',L1bold, T1t[1])
        A2a = (1.0/beta)*0.25*einsum('vijab,vabij->v',L2aaold, T2t[0])
        A2b = (1.0/beta)*0.25*einsum('vijab,vabij->v',L2bbold, T2t[2])
        A2ab = (1.0/beta)*einsum('vijab,vabij->v',L2abold, T2t[1])

        g = quadrature.get_gint(ng, delta)
        A2g = einsum('v,v->',A2a,g)
        A2g += einsum('v,v->',A2ab,g)
        A2g += einsum('v,v->',A2b,g)
        A1g = einsum('v,v->',A1a,g)
        A1g += einsum('v,v->',A1b,g)
        der_cc = Eterm + A1g + A2g

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
        n = fo.shape[0]

        # get time-grid
        ng = self.ngrid
        delta = beta/(ng - 1.0)
        ddelta = delta/beta
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng,delta)
        g = quadrature.get_gint(ng, delta)
        Gd = quadrature.get_G(ng,ddelta)
        gd = quadrature.get_gint(ng, ddelta)

        # get exponentials
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
            - en[None,None,:,None] - en[None,None,None,:]

        F,I = cc_utils.get_ft_integrals(self.sys, en, beta, mu)

        # get derivative with respect to g
        Eterm = ft_cc_energy.ft_cc_energy(self.T1,self.T2,
            F.ov,I.oovv,ti,gd,beta)

        dg = Eterm

        # get derivative with respect to G
        T1temp,T2temp = ft_cc_equations.ccsd_stanton(F,I,self.T1,self.T2,
                D1,D2,ti,ng,Gd)

        A1 = (1.0/beta)*einsum('via,vai->v',self.L1, T1temp)
        A2 = (1.0/beta)*0.25*einsum('vijab,vabij->v',self.L2, T2temp)

        g = quadrature.get_gint(ng, delta)
        A1g = einsum('v,v->',A1,g)
        A2g = einsum('v,v->',A2,g)
        dG = A1g + A2g

        # append derivative with respect to ti points
        Gnew = G.copy()
        m = G.shape[0]
        n = G.shape[0]
        for i in range(m):
            for j in range(n):
                Gnew[i,j] *= (j - i)/(ng - 1.0)
 
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
        delta = beta/(ng - 1.0)
        ddelta = delta/beta
        ti = numpy.asarray([float(i)*delta for i in range(ng)])
        G = quadrature.get_G(ng,delta)
        g = quadrature.get_gint(ng, delta)
        Gd = quadrature.get_G(ng,ddelta)
        gd = quadrature.get_gint(ng, ddelta)

        # get exponentials
        D1a = ea[:,None] - ea[None,:]
        D1b = eb[:,None] - eb[None,:]
        D2aa = ea[:,None,None,None] + ea[None,:,None,None] \
            - ea[None,None,:,None] - ea[None,None,None,:]
        D2ab = ea[:,None,None,None] + eb[None,:,None,None] \
            - ea[None,None,:,None] - eb[None,None,None,:]
        D2bb = eb[:,None,None,None] + eb[None,:,None,None] \
            - eb[None,None,:,None] - eb[None,None,None,:]

        F,I = cc_utils.get_ft_integrals(self.sys, en, beta, mu)
        Fa,Fb,Ia,Ib,Iabab = cc_utils.get_uft_integrals(self.sys, ea, eb, beta, mu)
        T1aold,T1bold = self.T1
        T2aaold,T2abold,T2bbold = self.T2
        L1aold,L1bold = self.L1
        L2aaold,L2abold,L2bbold = self.L2

        # get derivative with respect to g
        Eterm = ft_cc_energy.ft_ucc_energy(T1aold,T1bold,T2aaold,T2abold,T2bbold,
            Fa.ov,Fb.ov,Ia.oovv,Ib.oovv,Iabab.oovv,ti,gd,beta)

        dg = Eterm

        # get derivative with respect to G
        T1t,T2t = ft_cc_equations.uccsd_stanton(Fa,Fb,Ia,Ib,Iabab,T1aold,T1bold,
                T2aaold,T2abold,T2bbold,D1a,D1b,D2aa,D2ab,D2bb,ti,ng,Gd)

        A1a = (1.0/beta)*einsum('via,vai->v',L1aold, T1t[0])
        A1b = (1.0/beta)*einsum('via,vai->v',L1bold, T1t[1])
        A2a = (1.0/beta)*0.25*einsum('vijab,vabij->v',L2aaold, T2t[0])
        A2b = (1.0/beta)*0.25*einsum('vijab,vabij->v',L2bbold, T2t[2])
        A2ab = (1.0/beta)*einsum('vijab,vabij->v',L2abold, T2t[1])

        g = quadrature.get_gint(ng, delta)
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
                Gnew[i,j] *= (j - i)/(ng - 1.0)

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

    def _ft_1rdm(self):
        # temperature info
        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu

        # get time-grid
        ng = self.ngrid
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)

        # get energy differences
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
                - en[None,None,:,None] - en[None,None,None,:]

        pia,pba,pji,pai = ft_cc_equations.ccsd_1rdm(self.T1,self.T2,self.L1,self.L2,D1,D2,ti,ng,delta)
        self.dia = pia
        self.dba = pba
        self.dji = pji
        self.dai = pai

    def _ft_2rdm(self):
        # temperature info
        T = self.T
        beta = 1.0 / (T + 1e-12)
        mu = self.mu

        # get time-grid
        ng = self.ngrid
        delta = beta/(ng - 1.0)
        ti = numpy.asarray([float(i)*delta for i in range(ng)])

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)

        # get energy differences
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
                - en[None,None,:,None] - en[None,None,None,:]

        P2 = ft_cc_equations.ccsd_2rdm(self.T1,self.T2,self.L1,self.L2,D1,D2,ti,ng,delta)
        self.P2 = P2