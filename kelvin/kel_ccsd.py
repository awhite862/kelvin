import numpy
from cqcpy import cc_equations
from pyscf import lib
from . import cc_utils
from . import propagation
from .td_ccsd import _get_active

#einsum = numpy.einsum
einsum = lib.einsum

class KelCCSD(object):
    """Keldysh-contour CCSD class. This should be used in preference
    to 'neq_ccsd.'"""
    def __init__(self, sys, prop, T=0.0, mu=0.0, iprint=0, singles=True,
            athresh=0.0, fthresh=0.0):

        self.sys = sys
        self.T = T
        self.mu = mu
        self.finite_T = False if T == 0 else True
        self.iprint = iprint
        self.singles = singles
        self.athresh = athresh
        self.fthresh = fthresh
        self.prop = prop
        if not sys.verify(self.T,self.mu):
            raise Exception("Sytem temperature inconsistent with CC temp")
        self.beta = 1.0/T if T > 0.0 else None
        focc,fvir,iocc,ivir = _get_active(self.athresh, self.fthresh, self.beta, mu, self.sys, iprint)
        self.focc = focc
        self.fvir = fvir
        self.iocc = iocc
        self.ivir = ivir

        self.tcur = 0.0
        self.T1 = None
        self.T2 = None
        self.L1 = None
        self.L2 = None
        self.P = []
        self.ti = []

    def init(self, T1=None, T2=None, L1=None, L2=None):
        self.T1 = T1
        self.T2 = T2
        self.L1 = L1
        self.L2 = L2

    def init_from_ftccsd(self, tdccsd, contour=None, itime=None):
        """Initialize dynamics from an ftccsd onject

        Arguments:
            tdccsd (TDCCSD): FT-CCSD object
            contour (str): String indicating contour we use
            itime (int): Index of imaginary time indicating contour
        """
        if contour is None and itime is None:
            raise Exception("Please specify contour")

        if contour is not None and itime is not None:
            print("Warning: Both contour and itime are specified!")

        if contour is not None:
            if contour.lower() == "keldysh":
                itime = 0
            elif (contour.lower() == "rkeldysh"
                    or contour.lower() == "reverse_keldysh"):
                itime = tdccsd.ngrid - 1
            elif (contour.lower() == "skeldysh"
                    or contour.lower() == "symmetric_keldysh"):
                itime = tdccsd.ngrid//2
            else:
                raise Exception("Unrecognized contour!")
        else:
            assert(itime < tdccsd.ngrid)
        assert(self.singles == tdccsd.singles)
        self.T1 = tdccsd._read_T1(itime)
        self.T2 = tdccsd._read_T2(itime)
        self.L1 = tdccsd._read_L1(itime)
        self.L2 = tdccsd._read_L2(itime)

    def _step(self, prop, t0, var, h, func):
        if prop == "rk1":
            return propagation.rk1_gen(t0, var, h, func)
        elif prop == "rk2":
            return propagation.rk2_gen(t0, var, h, func)
        elif prop == "rk4":
            return propagation.rk4_gen(t0, var, h, func)
        else:
            raise Exception("Unrecognized propagation scheme: " + prop)

    def _ccsd(self, nstep, rdm2=False, step=0.1, save=1):
        mu = self.mu

        # get time-grid
        t0 = 0.0
        beta = self.beta

        # get orbital energies
        en = self.sys.g_energies_tot()
        n = en.shape[0]

        # get 0th and 1st order contributions
        #En = self.sys.const_energy()
        #g0 = ft_utils.GP0(beta, en, mu)
        #E0 = ft_mp.mp0(g0) + En
        #E1 = self.sys.get_mp1()
        #E01 = E0 + E1

        # get scaled integrals
        F,I = cc_utils.ft_active_integrals(self.sys, en, self.focc, self.fvir, self.iocc, self.ivir)

        # get exponentials
        D1 = en[:,None] - en[None,:]
        D2 = en[:,None,None,None] + en[None,:,None,None] \
                - en[None,None,:,None] - en[None,None,None,:]
        D1 = D1[numpy.ix_(self.ivir,self.iocc)]
        D2 = D2[numpy.ix_(self.ivir,self.ivir,self.iocc,self.iocc)]
        sfo = numpy.sqrt(self.focc)
        sfv = numpy.sqrt(self.fvir)

        # compute initial density matrix
        pia = self.L1.copy()
        pji = cc_equations.ccsd_1rdm_ji_opt(self.T1,self.T2,self.L1,self.L2)
        pba = cc_equations.ccsd_1rdm_ba_opt(self.T1,self.T2,self.L1,self.L2)
        pai = cc_equations.ccsd_1rdm_ai_opt(self.T1,self.T2,self.L1,self.L2)

        ndia = numpy.einsum('ia,i,a->ia',pia,sfo,sfv)
        ndba = numpy.einsum('ba,b,a->ba',pba,sfv,sfv)
        ndji = numpy.einsum('ji,j,i->ji',pji,sfo,sfo)
        ndai = numpy.einsum('ai,a,i->ai',pai,sfv,sfo)

        n1rdm = numpy.zeros((n,n), dtype=complex)
        n1rdm[numpy.ix_(self.iocc,self.iocc)] += ndji
        n1rdm[numpy.ix_(self.iocc,self.ivir)] += ndia
        n1rdm[numpy.ix_(self.ivir,self.iocc)] += ndai
        n1rdm[numpy.ix_(self.ivir,self.ivir)] += ndba
        n1rdm[numpy.ix_(self.iocc,self.iocc)] += numpy.diag(self.focc)
        self.P.append(n1rdm)
        self.ti.append(t0)

        if rdm2:
            raise Exception("2-rdm is not implemented!")

        for i in range(1,nstep):
            t = t0 + step
            h = step
            def fRHSt(t, ts):
                t1, t2 = ts
                F = cc_utils.ft_integrals_neq_1e(self.sys, en, beta, mu, t)
                k1s = -1.j*D1*t1 - 1.j*F.vo.copy()
                k1d = -1.j*D2*t2 - 1.j*I.vvoo.copy()
                cc_equations._Stanton(k1s,k1d,F,I,t1,t2,fac=-1.j)
                if not self.singles:
                    k1s = numpy.zeros(k1s.shape, k1s.dtype)
                return [k1s,k1d]

            dT = self._step(self.prop["tprop"], t0, [self.T1,self.T2], h, fRHSt)
            T1 = self.T1 + dT[0]
            T2 = self.T2 + dT[1]

            def fLRHSt(t, ls):
                l1,l2 = ls
                M1 = (T1 - self.T1)/h
                M2 = (T2 - self.T2)/h
                t1 = self.T1 + (t - t0)*M1
                t2 = self.T2 + (t - t0)*M2
                F = cc_utils.ft_integrals_neq_1e(self.sys, en, beta, mu, t)
                l1s = 1.j*D1.transpose((1,0))*l1 + 1.j*F.ov.copy()
                l1d = 1.j*D2.transpose((2,3,0,1))*l2 + 1.j*I.oovv.copy()
                cc_equations._Lambda_opt(l1s, l1d, F, I,
                        l1, l2, t1, t2, fac=1.j)
                cc_equations._LS_TS(l1s,I,t1,fac=1.j)
                if not self.singles:
                    l1s = numpy.zeros(l1s.shape, l1s.dtype)
                return [l1s,l1d]

            dL = self._step(self.prop["lprop"], t0, [self.L1,self.L2], h, fLRHSt)
            L1 = self.L1 + dL[0]
            L2 = self.L2 + dL[1]

            # compute density matrix
            pia = L1.copy()
            pji = cc_equations.ccsd_1rdm_ji_opt(T1,T2,L1,L2)
            pba = cc_equations.ccsd_1rdm_ba_opt(T1,T2,L1,L2)
            pai = cc_equations.ccsd_1rdm_ai_opt(T1,T2,L1,L2)

            ndia = numpy.einsum('ia,i,a->ia',pia,sfo,sfv)
            ndba = numpy.einsum('ba,b,a->ba',pba,sfv,sfv)
            ndji = numpy.einsum('ji,j,i->ji',pji,sfo,sfo)
            ndai = numpy.einsum('ai,a,i->ai',pai,sfv,sfo)

            # save current amplitudes
            self.T1 = T1
            self.T2 = T2
            self.L1 = L1
            self.L2 = L2

            # save density matrix
            n1rdm = numpy.zeros((n,n), dtype=complex)
            n1rdm[numpy.ix_(self.iocc,self.iocc)] += ndji
            n1rdm[numpy.ix_(self.iocc,self.ivir)] += ndia
            n1rdm[numpy.ix_(self.ivir,self.iocc)] += ndai
            n1rdm[numpy.ix_(self.ivir,self.ivir)] += ndba
            n1rdm[numpy.ix_(self.iocc,self.iocc)] += numpy.diag(self.focc)
            if i % save == 0:
                self.P.append(n1rdm)
                self.ti.append(t)
            t0 = t
