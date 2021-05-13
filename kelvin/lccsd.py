import numpy
from cqcpy import cc_energy
from cqcpy import cc_equations
from cqcpy import ft_utils
from . import cc_utils
from . import ft_mp
from . import ft_cc_energy
from . import quadrature
from . import zt_mp
from .mp2 import mp2


class lccsd(object):
    """Linearized coupled cluster singles and doubles (LCCSD) driver.

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
    """
    def __init__(self, sys, T=0, mu=0, iprint=0,
            singles=True, econv=1e-8, tconv=None, max_iter=40,
            damp=0.0, ngrid=10, realtime=False, athresh=0.0):
        self.T = T
        self.mu = mu
        self.finite_T = False if T == 0 else True
        self.iprint = iprint
        self.singles = singles
        self.econv = econv
        self.tconv = tconv if tconv is not None else self.econv*1000.0
        self.max_iter = max_iter
        self.damp = damp
        self.ngrid = ngrid
        self.realtime = realtime
        self.athresh = athresh
        if self.finite_T:
            self.realtime = True
        if not sys.verify(self.T,self.mu):
            raise Exception("Sytem temperature inconsistent with LCCSD temp")
        self.beta = None
        self.ti = None
        self.g = None
        self.G = None
        if self.realtime:
            if self.finite_T:
                self.beta = 1.0/T
                self.beta_max = self.beta
            else:
                self.beta = None
                self.beta_max = 80
            self.ti,self.g,self.G = quadrature.ft_quad(self.ngrid, self.beta_max, self.quad)
        self.sys = sys

        self.sys = sys

    def run(self, T1=None, T2=None):
        if self.finite_T:
            if self.iprint > 0:
                print('Running LCCSD at an electronic temperature of %f K'
                    % ft_utils.HtoK(self.T))
            if self.athresh > 0.0:
                return self._ft_lccsd_active(T1in=T1,T2in=T2)
            else:
                return self._ft_lccsd(T1in=T1,T2in=T2)
        else:
            if self.iprint > 0:
                print('Running LCCSD at zero Temperature')
            if self.realtime:
                return self._lccsd_rt()
            else:
                return self._lccsd()

    def _lccsd(self):
        # create energies and denominators in spin-orbital basis
        eo,ev = self.sys.g_energies()
        Dov = 1.0/(eo[:,None] - ev[None,:])
        Doovv = 1.0/(eo[:,None,None,None] + eo[None,:,None,None]
            - ev[None,None,:,None] - ev[None,None,None,:])

        # get HF energy
        En = self.sys.const_energy()
        E0 = zt_mp.mp0(eo) + En
        E1 = self.sys.get_mp1()

        # get Fock matrix
        F = self.sys.g_fock()
        F.oo = F.oo - numpy.diag(eo) # subtract diagonal
        F.vv = F.vv - numpy.diag(ev) # subtract diagonal

        # get ERIs
        I = self.sys.g_aint()

        # get MP2 T-amplitudes
        mp20 = mp2(self.sys,saveT=True)
        mp20.run()
        T1old = (mp20.T1 if self.singles else numpy.zeros(F.vo.shape))
        T2old = mp20.T2

        # run CC iterations
        converged = False
        thresh = self.econv
        max_iter = self.max_iter
        i = 0
        Eold = 1000000
        while i < max_iter and not converged:
            if self.singles:
                T1,T2 = cc_equations.lccsd_simple(F, I, T1old, T2old)
            else:
                T2 = cc_equations.lccd_simple(F, I, T2old)
                T1 = numpy.zeros(F.vo.shape)
            T1 = numpy.einsum('ai,ia->ai', T1, Dov)
            T2 = numpy.einsum('abij,ijab->abij', T2, Doovv)
            E = cc_energy.cc_energy(T1,T2,F.ov,I.oovv)
            if self.iprint > 0:
                print(' %d  %.10f' % (i+1,E))
            i = i + 1
            if numpy.abs(E - Eold) < thresh:
                converged = True
            Eold = E
            T1old = T1
            T2old = T2
            T1 = None
            T2 = None

        # save and return
        self.T1 = T1old
        self.T2 = T2old
        return Eold

    def _lccsd_rt(self, T1in=None, T2in=None):
        # create energies in spin-orbital basis
        eo,ev = self.sys.g_energies()
        no = eo.shape[0]
        nv = ev.shape[0]
        Dvo = (ev[:,None] - eo[None,:])
        Dvvoo = (ev[:,None,None,None] + ev[None,:,None,None]
            - eo[None,None,:,None] - eo[None,None,None,:])

        # get time-grid
        ng = self.ngrid
        ti = self.ti
        G = self.G
        g = self.g

        # get HF energy
        En = self.sys.const_energy()
        E0 = zt_mp.mp0(eo) + En
        E1 = self.sys.get_mp1()

        # get Fock matrix
        F = self.sys.g_fock()
        F.oo = F.oo - numpy.diag(eo) # subtract diagonal
        F.vv = F.vv - numpy.diag(ev) # subtract diagonal

        # get ERIs
        I = self.sys.g_aint()

        # get MP2 T-amplitudes
        Id = numpy.ones(ng)
        if T1in is not None and T2in is not None:
            T1old = T1in if self.singles else numpy.zeros((ng,nv,no))
            T2old = T2in
        else:
            if self.singles:
                T1old = -numpy.einsum('v,ai->vai', Id, F.vo)
                T1old = quadrature.int_tbar1(ng,T1old,ti,Dvo,G)
            else:
                T1old = numpy.zeros((ng,nv,no))
            T2old = -numpy.einsum('v,abij->vabij', Id, I.vvoo)
            T2old = quadrature.int_tbar2(ng,T2old,ti,Dvvoo,G)
        E2 = ft_cc_energy.ft_cc_energy(T1old, T2old,
            F.ov, I.oovv, g, self.beta_max, Qterm=False)
        if self.iprint > 0:
            print('MP2 energy: {:.10f}'.format(E2))

        # run CC iterations
        conv_options = {"econv":self.econv, "max_iter":self.max_iter, "damp":self.damp}
        method = "LCCSD" if self.singles else "LCCD"
        Eccn,T1,T2 = cc_utils.ft_cc_iter(method, T1old, T2old, F, I,
                Dvo, Dvvoo, g, G, self.beta_max, ng, ti, self.iprint, conv_options)
        self.T1 = T1
        self.T2 = T2

        # save and return
        return Eccn

    def _ft_lccsd(self, T1in=None, T2in=None):
        # get T and mu variables
        beta = self.beta
        mu = self.mu
        assert(self.beta_max == self.beta)

        # get time-grid
        ng = self.ngrid
        ti = self.ti
        G = self.G
        g = self.g

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()

        # compute requisite memory
        n = en.shape[0]
        mem1e = 6*n*n + 3*ng*n*n
        mem2e = 3*n*n*n*n + 3*ng*n*n*n*n
        mem_mb = 2.0*(mem1e + mem2e)*8.0/1024.0/1024.0
        #assert(mem_mb < 4000)
        if self.iprint > 0:
            print('  FT-LCCSD will use %f mb' % mem_mb)

        # 0th and 1st order contributions
        En = self.sys.const_energy()
        g0 = ft_utils.GP0(beta, en, mu)
        E0 = ft_mp.mp0(g0) + En
        E1 = self.sys.get_mp1()
        E01 = E0 + E1

        # get scaled integrals
        F,I = cc_utils.ft_integrals(self.sys, en, beta, mu)

        # get energy differences
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
            - en[None,None,:,None] - en[None,None,None,:]

        # get MP2 T-amplitudes
        if T1in is not None and T2in is not None:
            T1old = T1in
            T2old = T2in
        else:
            Id = numpy.ones((ng))
            T1old = -numpy.einsum('v,ai->vai', Id, F.vo)
            T2old = -numpy.einsum('v,abij->vabij', Id, I.vvoo)
            T1old = quadrature.int_tbar1(ng,T1old,ti,D1,G)
            T2old = quadrature.int_tbar2(ng,T2old,ti,D2,G)
        if not self.singles:
            T1old = numpy.zeros(T1old.shape)
        #E2 = ft_cc_energy.ft_cc_energy(T1old,T2old,
        #    F.ov,I.oovv,g,beta,Qterm=False)

        # run CC iterations
        conv_options = {
            "econv":self.econv,
            "tconv":self.tconv,
            "max_iter":self.max_iter,
            "damp":self.damp}
        method = "LCCSD" if self.singles else "LCCD"
        Eccn,T1,T2 = cc_utils.ft_cc_iter(method, T1old, T2old, F, I, D1, D2,
                g, G, beta, ng, ti, self.iprint, conv_options)
        self.T1 = T1
        self.T2 = T2

        print('total energy: %f' % (Eccn+E01))
        return (Eccn+E01,Eccn)

    def _ft_lccsd_active(self, T1in=None, T2in=None):
        # get T and mu variables
        assert(self.beta_max == self.beta)
        beta = self.beta
        mu = self.mu

        # get time-grid
        ng = self.ngrid
        ti = self.ti
        G = self.G
        g = self.g

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
        if self.iprint > 0:
            print("FT-LCCSD orbital info:")
            print('  nocc: {:d}'.format(nocc))
            print('  nvir: {:d}'.format(nvir))
            print('  nact: {:d}'.format(nact))

        # compute requisite memory
        n = en.shape[0]
        mem1e = 6*n*n + 3*ng*n*n
        mem2e = 3*n*n*n*n + 3*ng*n*n*n*n
        mem_mb = 2.0*(mem1e + mem2e)*8.0/1024.0/1024.0
        if self.iprint > 0:
            print('  FT-LCCSD will use %f mb' % mem_mb)

        # get 0th and 1st order contributions
        En = self.sys.const_energy()
        g0 = ft_utils.GP0(beta, en, mu)
        E0 = ft_mp.mp0(g0) + En
        E1 = self.sys.get_mp1()
        E01 = E0 + E1

        # get scaled integrals
        F,I = cc_utils.ft_active_integrals(self.sys, en, focc, fvir, iocc, ivir)

        # get exponentials
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
            - en[None,None,:,None] - en[None,None,None,:]
        D1 = D1[numpy.ix_(ivir,iocc)]
        D2 = D2[numpy.ix_(ivir,ivir,iocc,iocc)]

        # get MP2 T-amplitudes
        if T1in is not None and T2in is not None:
            T1old = T1in
            T2old = T2in
        else:
            Id = numpy.ones((ng))
            T1old = -numpy.einsum('v,ai->vai', Id, F.vo)
            T2old = -numpy.einsum('v,abij->vabij', Id, I.vvoo)
            T1old = quadrature.int_tbar1(ng,T1old,ti,D1,G)
            T2old = quadrature.int_tbar2(ng,T2old,ti,D2,G)
        #E2 = ft_cc_energy.ft_cc_energy(T1old,T2old,
        #    F.ov,I.oovv,g,beta,Qterm=False)

        # run CC iterations
        method = "LCCSD" if self.singles else "LCCD"
        conv_options = {
            "econv":self.econv,
            "tconv":self.tconv,
            "max_iter":self.max_iter,
            "damp":self.damp}
        Eccn,T1,T2 = cc_utils.ft_cc_iter(method, T1old, T2old, F, I, D1, D2, g, G, beta,
                ng, ti, self.iprint, conv_options)
        self.T1 = T1
        self.T2 = T2

        return (Eccn+E01,Eccn)

    def _ft_lccsd_lambda(self):
        # get T and mu variables
        assert(self.beta_max == self.beta)
        beta = self.beta
        mu = self.mu

        # get time-grid
        ng = self.ngrid
        ti = self.ti
        G = self.G
        g = self.g

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()

        # compute requisite memory
        n = en.shape[0]
        mem1e = 6*n*n + 3*ng*n*n
        mem2e = 3*n*n*n*n + 3*ng*n*n*n*n
        mem_mb = 2.0*(mem1e + mem2e)*8.0/1024.0/1024.0
        if self.iprint > 0:
            print('  FT-LCCSD will use %f mb' % mem_mb)

        # get scaled integrals
        F,I = cc_utils.ft_integrals(self.sys, en, beta, mu)

        # get energy differences
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
            - en[None,None,:,None] - en[None,None,None,:]

        # run Lambda iterations
        if self.singles:
            L1old = numpy.transpose(self.T1,(0,2,1))
        else:
            L1old = numpy.zeros(self.T1.shape)
        L2old = numpy.transpose(self.T2,(0,3,4,1,2))
        method = "LCCSD" if self.singles else "LCCD"
        conv_options = {
            "econv":self.econv,
            "tconv":self.tconv,
            "max_iter":self.max_iter,
            "damp":self.damp}
        L1,L2 = cc_utils.ft_lambda_iter(method, L1old, L2old, self.T1, self.T2, F, I,
                D1, D2, g, G, beta, ng, ti, self.iprint, conv_options)

        # save lambda amplitudes
        self.L1 = L1
        self.L2 = L2
