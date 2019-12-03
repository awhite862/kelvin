import numpy
import time
from pyscf import lib
from cqcpy import cc_equations
from cqcpy import ft_utils
from cqcpy.cc_energy import cc_energy
from . import cc_utils
from . import ft_cc_energy
from . import ft_cc_equations
from . import ft_mp
from . import quadrature
from . import propagation

class RTCCSD(object):
    """Real-time coupled cluster singles and doubles (CCSD) driver.
    """
    def __init__(self, sys, T=0.0, mu=0.0, iprint=0,
        singles=True, econv=1e-8, tconv=None, max_iter=40,
        damp=0.0, ngrid=10, athresh=0.0, quad='lin', prop="rk4"):

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
        self.athresh = athresh
        self.quad = quad
        self.prop = prop
        if not sys.verify(self.T,self.mu):
            raise Exception("Sytem temperature inconsistent with CC temp")
        if self.finite_T:
            beta_max = 1.0/(T + 1e-12)
        else:
            beta_max = 80
        self._beta_max = 80
        ng = self.ngrid
        self.ti,self.g,G = quadrature.ft_quad(self.ngrid,beta_max,self.quad)
        self.sys = sys
        self.dia = None
        self.dba = None
        self.dji = None
        self.dai = None
        self.P2 = None

    def run(self,response=None):
        """Run CCSD calculation."""
        if self.finite_T:
            if self.iprint > 0:
                print('Running CCSD at an electronic temperature of %f K'
                    % ft_utils.HtoK(self.T))
        else:
            if self.iprint > 0:
                print('Running CCSD at zero Temperature')
        return self._ccsd()

    def _ccsd(self):
        beta = 1.0 / self.T if self.finite_T else self._beta_max
        mu = self.mu if self.finite_T else None

        # get time-grid
        ng = self.ngrid
        ti = self.ti
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
            t1shape = (nv,no)
            t1shape = (nv,nv,no,no)

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
                t1shape = (nvir,nocc)
                t2shape = (nvir,nvir,nocc,nocc)

            else:
                # get scaled integrals
                F,I = cc_utils.ft_integrals(self.sys, en, beta, mu)

                # get energy differences
                D1 = en[:,None] - en[None,:]
                D2 = en[:,None,None,None] + en[None,:,None,None] \
                        - en[None,None,:,None] - en[None,None,None,:]
                t1shape = (n,n)
                t2shape = (n,n,n,n)

        t1 = numpy.zeros(t1shape)
        t2 = numpy.zeros(t2shape)
        Eccn = 0.0

        def fRHS(var):
            t1,t2 = var
            k1s = -D1*t1 - F.vo.copy()
            k1d = -D2*t2 - I.vvoo.copy()
            cc_equations._Stanton(k1s,k1d,F,I,t1,t2,fac=-1.0)
            return [k1s,k1d]

        for i in range(1,ng):
            # propagate
            h = self.ti[i] - self.ti[i - 1]
            if self.prop == "rk1" or self.prop == "ab1":
                d1,d2 = propagation.rk1(h, [t1,t2], fRHS)
                t1 += d1
                t2 += d2
            elif self.prop == "rk2":
                d1,d2 = propagation.rk2(h, [t1, t2], (fRHS,fRHS))
                t1 += d1
                t2 += d2
            elif self.prop == "rk4":
                d1,d2 = propagation.rk4(h, [t1, t2], (fRHS,fRHS,fRHS,fRHS))
                t1 += d1
                t2 += d2
            elif self.prop == "ab2":
                if i == 1:
                    k2s = None
                    k2d = None
                    d,k2 = propagation.ab2(h, [t1, t2], [k2s, k2d], fRHS)
                    d1,d2 = d
                    k2s, k2d = k2
                    t1 += d1
                    t2 += d2
                else:
                    d,k2, = propagation.ab2(h, [t1, t2], [k2s, k2d], fRHS)
                    d1,d2 = d
                    k2s, k2d = k2
                    t1 += d1
                    t2 += d2
            elif self.prop == "be":
                mi = 100
                alpha = 0.4
                thresh = 1e-5
                d1,d2 = propagation.be(h, [t1, t2], mi, alpha, thresh, fRHS, self.iprint)
                t1 += d1
                t2 += d2
            elif self.prop == "cn":
                mi = 100
                alpha = 0.6
                thresh = 1e-5
                d1,d2 = propagation.cn(h, [t1, t2], mi, alpha, thresh, (fRHS,fRHS), self.iprint)
                t1 += d1
                t2 += d2
            elif self.prop == "am2":
                mi = 100
                alpha = 0.6
                thresh = 1e-5
                d1,d2 = propagation.am2(h, [t1, t2], mi, alpha, thresh, fRHS, self.iprint)
                t1 += d1
                t2 += d2
            else:
                raise Exception("Unrecognized propagation scheme: " + self.prop)

            # compute free energy contribution
            Eccn += g[i]*cc_energy(t1, t2, F.ov, I.oovv)/beta

        self.G0 = E0
        self.G1 = E1
        self.Gcc = Eccn
        self.Gtot = E0 + E1 + Eccn
        self.T1 = t1
        self.T2 = t2

        return (Eccn+E01,Eccn)

    def _ccsd_lambda(self,rdm2=False):
        if self.T1 is None or self.T2 is None:
            raise Exception("No saved T-amplitudes")

        beta = 1.0 / self.T if self.finite_T else self._beta_max
        mu = self.mu if self.finite_T else None

        # get time-grid
        ng = self.ngrid
        ti = self.ti
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
            t1shape = (nv,no)
            t1shape = (nv,nv,no,no)

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
                t1shape = (nvir,nocc)
                t1shape = (nvir,nvir,nocc,nocc)

            else:
                # get scaled integrals
                F,I = cc_utils.ft_integrals(self.sys, en, beta, mu)

                # get energy differences
                D1 = en[:,None] - en[None,:]
                D2 = en[:,None,None,None] + en[None,:,None,None] \
                        - en[None,None,:,None] - en[None,None,None,:]
                t1shape = (n,n)
                t2shape = (n,n,n,n)

        def fRHS(var):
            t1,t2 = var
            k1s = -D1*t1 - F.vo.copy()
            k1d = -D2*t2 - I.vvoo.copy()
            cc_equations._Stanton(k1s,k1d,F,I,t1,t2,fac=-1.0)
            return [(-1.0)*k1s,(-1.0)*k1d]

        def fLRHS(t1,t2,var):
            l1,l2 = var
            l1s = -D1.transpose((1,0))*l1 - F.ov.copy()
            l1d = -D2.transpose((2,3,0,1))*l2 - I.oovv.copy()
            cc_equations._Lambda_opt(l1s, l1d, F, I,
                    l1, l2, t1, t2, fac=-1.0)
            cc_equations._LS_TS(l1s,I,t1,fac=-1.0)
            return [l1s,l1d]

        t1 = self.T1.copy()
        t2 = self.T2.copy()
        nv, no = t1.shape
        l1 = numpy.zeros((no,nv))
        l2 = numpy.zeros((no,no,nv,nv))
        pia = numpy.zeros((no,nv))
        pji = g[ng - 1]*cc_equations.ccsd_1rdm_ji_opt(t1,t2,l1,l2)
        pba = g[ng - 1]*cc_equations.ccsd_1rdm_ba_opt(t1,t2,l1,l2)
        pai = g[ng - 1]*cc_equations.ccsd_1rdm_ai_opt(t1,t2,l1,l2)

        Eccn = g[ng - 1]*cc_energy(t1, t2, F.ov, I.oovv)/beta
        for i in range(1,ng):
            h = self.ti[ng - i] - self.ti[ng - i - 1]
            # propagate T and lambda
            if self.prop == "rk1":
                d1,d2 = propagation.rk1(h, [t1, t2], fRHS)
                LRHS = lambda var: fLRHS(t1, t2, var)
                dl1,dl2 = propagation.rk1(h, [l1, l2], LRHS)
                t1 += d1
                t2 += d2
                l1 += dl1
                l2 += dl2
            elif self.prop == "rk2":
                d1,d2 = propagation.rk2(h, [t1, t2], (fRHS,fRHS))
                LRHS1 = lambda var: fLRHS(t1, t2, var)
                LRHS2 = lambda var: fLRHS(t1 + d1, t2 + d2, var)
                dl1,dl2 = propagation.rk2(h, [l1, l2], (LRHS1, LRHS2))
                t1 += d1
                t2 += d2
                l1 += dl1
                l2 += dl2
            elif self.prop == "rk4":
                d1,d2 = propagation.rk4(h, [t1, t2], (fRHS,fRHS,fRHS,fRHS))

                LRHS1 = lambda var: fLRHS(t1, t2, var)
                LRHS23 = lambda var: fLRHS(t1 + 0.5*d1, t2 + 0.5*d2, var)
                LRHS4 = lambda var: fLRHS(t1 + d1, t2 + d2, var)
                ld1, ld2 = propagation.rk4(h, [l1, l2], (LRHS1, LRHS23, LRHS23, LRHS4))

                t1 += d1
                t2 += d2
                l1 += ld1
                l2 += ld2
            elif self.prop == "cn":
                mi = 200
                alpha = 0.8
                thresh = 1e-5
                d1, d2 = propagation.cn(h, [t1, t2], mi, alpha, thresh, (fRHS,fRHS), self.iprint)
                LRHS1 = lambda var: fLRHS(t1, t2, var)
                LRHS2 = lambda var: fLRHS(t1 + d1, t2 + d2, var)
                ld1, ld2 = propagation.cn(h, [l1, l2], mi, alpha, thresh, (LRHS1,LRHS2), self.iprint)
                l1 += ld1
                l2 += ld2
                t1 += d1
                t2 += d2
            else:
                raise Exception("Unrecognized propagation scheme: " + self.prop)
            Eccn += g[ng - i - 1]*cc_energy(t1, t2, F.ov, I.oovv)/beta

            # increment the RDMs
            pia += g[ng - i - 1]*l1
            pji += g[ng - i - 1]*cc_equations.ccsd_1rdm_ji_opt(t1,t2,l1,l2)
            pba += g[ng - i - 1]*cc_equations.ccsd_1rdm_ba_opt(t1,t2,l1,l2)
            pai += g[ng - i - 1]*cc_equations.ccsd_1rdm_ai_opt(t1,t2,l1,l2)

        G0 = E0
        G1 = E1
        Gcc = Eccn
        Gtot = E0 + E1 + Eccn
        self.L1 = l1
        self.L2 = l2
        self.dia = pia
        self.dji = pji
        self.dba = pba
        self.dai = pai

        return (Eccn+E01,Eccn)
