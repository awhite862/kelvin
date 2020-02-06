import numpy
import time
from pyscf import lib
from cqcpy import cc_equations
from cqcpy import ft_utils
from cqcpy.cc_energy import cc_energy, ucc_energy, rcc_energy
from . import cc_utils
#from . import ft_cc_energy
#from . import ft_cc_equations
from . import ft_mp
from . import quadrature
from . import propagation

einsum = numpy.einsum

class TDCCSD(object):
    """Real-time coupled cluster singles and doubles (CCSD) driver.
    """
    def __init__(self, sys, prop, T=0.0, mu=0.0, iprint=0,
        singles=True, econv=1e-8, tconv=None, max_iter=40,
        damp=0.0, ngrid=10, athresh=0.0, quad='lin', saveT=False,
        tmem="mem", scratch=""):

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
        self.saveT = saveT
        self.tmem = tmem
        self.scratch = scratch
        if not sys.verify(self.T,self.mu):
            raise Exception("Sytem temperature inconsistent with CC temp")
        if self.finite_T:
            beta = 1.0/T
        else:
            beta = 80
        self.beta = beta
        ng = self.ngrid
        self.ti,self.g,G = quadrature.ft_quad(self.ngrid,beta,self.quad)
        self.sys = sys
        self.T1 = [None]*ng
        self.T2 = [None]*ng
        self.L1 = None
        self.L2 = None
        # pieces of normal-ordered 1-rdm
        self.dia = None
        self.dba = None
        self.dji = None
        self.dai = None
        # pieces of unrelaxed 1-rdm
        self.ndia = None
        self.ndba = None
        self.ndji = None
        self.ndai = None
        # pieces of normal-ordered 2-rdm
        self.P2 = None
        # occupation number response
        self.rono = None
        self.ronv = None
        self.ron1 = None
        # orbital energy response
        self.rorbo = None
        self.rorbv = None

    def __del__(self):
        self._rmfile()

    def run(self,response=None):
        """Run CCSD calculation."""
        if self.finite_T:
            if self.iprint > 0:
                print('Running CCSD at an electronic temperature of %f K'
                    % ft_utils.HtoK(self.T))
        else:
            if self.iprint > 0:
                print('Running CCSD at zero Temperature')
        if self.sys.has_r():
            return self._rccsd()
        elif self.sys.has_u():
            return self._uccsd()
        else:
            return self._ccsd()

    def compute_ESN(self):
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
                    self._uccsd_lambda(rdm2=True)
                else:
                    self._ccsd_lambda(rdm2=True)
            if self.sys.has_u():
                self._u_ft_ESN()
            else:
                self._g_ft_ESN()

    def _g_ft_ESN(self):
        # temperature info
        beta = self.beta
        mu = self.mu

        self._g_ft_ron()

        # zero order contributions
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        B0 = ft_utils.dGP0(beta, en, mu)
        N0 = fo.sum()
        E0 = beta*B0.sum() + mu*N0 + self.G0

        # higher order contributions
        dvec = -numpy.ones(en.shape) # mu derivative
        N1 = numpy.einsum('i,i->',dvec, self.ron1)
        Ncc = numpy.einsum('i,i->',dvec, self.rono + self.ronv)
        N1 *= -1.0 # N = - dG/dmu
        Ncc *= -1.0
        dvec = (en - mu)/beta # beta derivative
        B1 = numpy.einsum('i,i->',dvec, self.ron1)
        Bcc = numpy.einsum('i,i->',dvec, self.rono + self.ronv)

        # compute other contributions to CC derivative
        Bcc -= self.Gcc/(beta) # derivative from factors of 1/beta
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

    def _u_ft_ESN(self):
        # temperature info
        beta = self.beta
        mu = self.mu

        self._u_ft_ron()

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
        dveca = -numpy.ones(ea.shape) # mu derivative
        dvecb = -numpy.ones(eb.shape) # mu derivative
        N1 = numpy.einsum('i,i->',dveca, self.ron1[0])
        N1 += numpy.einsum('i,i->',dvecb, self.ron1[1])
        Ncc = numpy.einsum('i,i->',dveca, self.rono[0] + self.ronv[1])
        Ncc += numpy.einsum('i,i->',dvecb, self.rono[0] + self.ronv[1])
        N1 *= -1.0 # N = - dG/dmu
        Ncc *= -1.0
        dveca = (ea - mu)/beta
        dvecb = (eb - mu)/beta
        B1 = numpy.einsum('i,i->',dveca, self.ron1[0])
        B1 += numpy.einsum('i,i->',dvecb, self.ron1[1])
        Bcc = numpy.einsum('i,i->',dveca, self.rono[0] + self.ronv[1])
        Bcc += numpy.einsum('i,i->',dvecb, self.rono[0] + self.ronv[1])

        # compute other contributions to CC derivative
        Bcc -= self.Gcc/(beta)
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

    def _get_t_step(self, h, ttot, fRHS):
        if self.prop["tprop"] == "rk1":
            dtot = propagation.rk1(h, ttot, fRHS)
        elif self.prop["tprop"] == "rk2":
            dtot = propagation.rk2(h, ttot, (fRHS,fRHS))
        elif self.prop["tprop"] == "rk4":
            dtot = propagation.rk4(h, ttot, (fRHS,fRHS,fRHS,fRHS))
        elif self.prop["tprop"] == "cn":
            mi = self.prop["max_iter"]
            alpha = self.prop["damp"]
            thresh = self.prop["thresh"]
            dtot = propagation.cn(h, ttot, mi, alpha, thresh, (fRHS,fRHS), self.iprint)
        elif self.prop["tprop"] == "be":
            mi = self.prop["max_iter"]
            alpha = self.prop["damp"]
            thresh = self.prop["thresh"]
            dtot = propagation.be(h, ttot, mi, alpha, thresh, fRHS, self.iprint)
        elif self.prop["tprop"] == "am2":
            mi = self.prop["max_iter"]
            alpha = self.prop["damp"]
            thresh = self.prop["thresh"]
            dtot = propagation.am2(h, ttot, mi, alpha, thresh, fRHS, self.iprint)
        else:
            raise Exception("Unrecognized propagation scheme: " + self.prop)
        return dtot

    def _get_l_step(self, h, ltot, tb, te, fLRHS):
        if self.prop["lprop"] == "rk1":
            LRHS = lambda var: fLRHS(tb, var)
            ldtot = propagation.rk1(h, ltot, LRHS)
        elif self.prop["lprop"] == "rk2":
            LRHS1 = lambda var: fLRHS(tb, var)
            LRHS2 = lambda var: fLRHS(te, var)
            ldtot = propagation.rk2(h, ltot, (LRHS1, LRHS2))
        elif self.prop["lprop"] == "rk4":
            tx = [0.5*(b + e) for b,e in zip(tb,te)]
            LRHS1 = lambda var: fLRHS(tb, var)
            LRHS23 = lambda var: fLRHS(tx, var)
            LRHS4 = lambda var: fLRHS(te, var)
            ldtot = propagation.rk4(h, ltot, (LRHS1, LRHS23, LRHS23, LRHS4))
        elif self.prop["lprop"] == "cn":
            mi = self.prop["max_iter"]
            alpha = self.prop["damp"]
            thresh = self.prop["thresh"]
            LRHS1 = lambda var: fLRHS(tb, var)
            LRHS2 = lambda var: fLRHS(te, var)
            ldtot = propagation.cn(h, ltot, mi, alpha, thresh, (LRHS1,LRHS2), self.iprint)
        else:
            raise Exception("Unrecognized propagation scheme: " + self.prop)
        return ldtot

    def _save_T1(self, i, T1):
        if self.tmem == "mem":
            self.T1[i] = T1
        elif self.tmem == "hdf5":
            import h5py
            filename = self.scratch + "T1_" + "{:05d}".format(i)
            self.T1[i] = filename
            h5f = h5py.File(filename, 'w')
            if type(T1) is tuple:
                t1a,t1b = T1
                h5f.create_dataset("t1a", data=t1a)
                h5f.create_dataset("t1b", data=t1b)
            else:
                h5f.create_dataset("t1", data=T1)
            h5f.close()
        else:
            raise Exception("Unrecognized memory option for amplitudes!")

    def _save_T2(self, i, T2):
        if self.tmem == "mem":
            self.T2[i] = T2
        elif self.tmem == "hdf5":
            import h5py
            filename = self.scratch + "T2_" + "{:05d}".format(i)
            self.T2[i] = filename
            h5f = h5py.File(filename, 'w')
            if type(T2) is tuple:
                t2aa,t2ab,t2bb = T2
                h5f.create_dataset("t2aa", data=t2aa)
                h5f.create_dataset("t2ab", data=t2ab)
                h5f.create_dataset("t2bb", data=t2bb)
            else:
                h5f.create_dataset("t2", data=T2)
            h5f.close()
        else:
            raise Exception("Unrecognized memory option for amplitudes!")

    def _read_T1(self, i):
        if self.tmem == "mem":
            return self.T1[i]
        elif self.tmem == "hdf5":
            import h5py
            filename = self.T1[i]
            h5f = h5py.File(filename, 'r')
            n = len(h5f.keys())
            if n == 1:
                t1 = h5f["t1"][:]
                h5f.close()
                return t1
            elif n == 2:
                t1a = h5f["t1a"][:]
                t1b = h5f["t1b"][:]
                h5f.close()
                return (t1a,t1b)
            else:
                h5f.close()
                raise Exception("Wrong number of T1 amplitudes in " + filename)
        else:
            raise Exception("Unrecognized memory option for amplitudes!")

    def _read_T2(self, i):
        if self.tmem == "mem":
            return self.T2[i]
        elif self.tmem == "hdf5":
            import h5py
            filename = self.T2[i]
            h5f = h5py.File(filename, 'r')
            n = len(h5f.keys())
            if n == 1:
                t1 = h5f["t2"][:]
                h5f.close()
                return t2
            elif n == 3:
                t2aa = h5f["t2aa"][:]
                t2ab = h5f["t2ab"][:]
                t2bb = h5f["t2bb"][:]
                h5f.close()
                return (t2aa,t2ab,t2bb)
            else:
                h5f.close()
                raise Exception("Wrong number of T2 amplitudes in " + filename)
        else:
            raise Exception("Unrecognized memory option for amplitudes!")

    def _rmfile(self):
        if self.tmem == "hdf5":
            import os
            for f in self.T1:
                os.remove(f)
            for f in self.T2:
                os.remove(f)
        else:
            return

    def _ccsd(self):
        beta = self.beta
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

        t1 = numpy.zeros(t1shape, dtype=F.vo.dtype)
        t2 = numpy.zeros(t2shape, dtype=I.vvoo.dtype)
        Eccn = 0.0

        def fRHS(var):
            t1,t2 = var
            k1s = -D1*t1 - F.vo.copy()
            k1d = -D2*t2 - I.vvoo.copy()
            cc_equations._Stanton(k1s,k1d,F,I,t1,t2,fac=-1.0)
            return [k1s,k1d]

        if self.saveT:
            self._save_T1(0, t1.copy())
            self._save_T2(0, t2.copy())

        for i in range(1,ng):
            # propagate
            h = self.ti[i] - self.ti[i - 1]
            d1,d2 = self._get_t_step(h, [t1,t2], fRHS)
            t1 += d1
            t2 += d2

            # compute free energy contribution
            Eccn += g[i]*cc_energy(t1, t2, F.ov, I.oovv)/beta
            if self.saveT:
                self._save_T1(i, t1.copy())
                self._save_T2(i, t2.copy())

        self.G0 = E0
        self.G1 = E1
        self.Gcc = Eccn
        self.Gtot = E0 + E1 + Eccn
        if not self.saveT:
            self._save_T1(ng - 1,t1)
            self._save_T2(ng - 1,t2)

        return (Eccn+E01,Eccn)

    def _uccsd(self):
        beta = self.beta
        mu = self.mu if self.finite_T else None

        # get time-grid
        ng = self.ngrid
        ti = self.ti
        g = self.g

        if not self.finite_T:
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
        else:
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
                T1ashape = (na,na)
                T1bshape = (nb,nb)
                T2ashape = (na,na,na,na)
                T2bshape = (nb,nb,nb,nb)
                nocca = na
                nvira = na
                noccb = nb
                nvirb = nb

        t1a = numpy.zeros((nvira,nocca), dtype=Fa.vo.dtype)
        t1b = numpy.zeros((nvirb,noccb), dtype=Fa.vo.dtype)
        t2aa = numpy.zeros((nvira,nvira,nocca,nocca), dtype=Ia.vvoo.dtype)
        t2ab = numpy.zeros((nvira,nvirb,nocca,noccb), dtype=Iabab.vvoo.dtype)
        t2bb = numpy.zeros((nvirb,nvirb,noccb,noccb), dtype=Ib.vvoo.dtype)
        Eccn = 0.0

        def fRHS(var):
            t1a,t1b,t2aa,t2ab,t2bb = var
            k1sa = -D1a*t1a - Fa.vo.copy()
            k1sb = -D1b*t1b - Fb.vo.copy()
            k1daa = -D2aa*t2aa - Ia.vvoo.copy()
            k1dab = -D2ab*t2ab - Iabab.vvoo.copy()
            k1dbb = -D2bb*t2bb - Ib.vvoo.copy()
            cc_equations._u_Stanton(k1sa, k1sb, k1daa, k1dab, k1dbb,
                    Fa, Fb, Ia, Ib, Iabab, (t1a,t1b), (t2aa,t2ab,t2bb), fac=-1.0)
            return [k1sa,k1sb,k1daa,k1dab,k1dbb]

        if self.saveT:
            self._save_T1(0, (t1a.copy(),t1b.copy()))
            self._save_T2(0, (t2aa.copy(),t2ab.copy(),t2bb.copy()))

        for i in range(1,ng):
            # propagate
            h = self.ti[i] - self.ti[i - 1]
            d1a,d1b,d2aa,d2ab,d2bb = self._get_t_step(h, [t1a,t1b,t2aa,t2ab,t2bb], fRHS)
            t1a += d1a
            t1b += d1b
            t2aa += d2aa
            t2ab += d2ab
            t2bb += d2bb

            # compute free energy contribution
            Eccn += g[i]*ucc_energy((t1a,t1b), (t2aa,t2ab,t2bb),
                    Fa.ov, Fb.ov, Ia.oovv, Ib.oovv, Iabab.oovv)/beta
            if self.saveT:
                self._save_T1(i, (t1a.copy(),t1b.copy()))
                self._save_T2(i, (t2aa.copy(),t2ab.copy(),t2bb.copy()))

        self.G0 = E0
        self.G1 = E1
        self.Gcc = Eccn
        self.Gtot = E0 + E1 + Eccn
        if not self.saveT:
            self._save_T1(ng - 1, (t1a,t1b))
            self._save_T2(ng - 1, (t2aa,t2ab,t2bb))

        return (Eccn+E01,Eccn)

    def _rccsd(self):
        beta = self.beta
        mu = self.mu if self.finite_T else None

        # get time-grid
        ng = self.ngrid
        ti = self.ti
        g = self.g

        if not self.finite_T:
            # create energies in spin-orbital basis
            eo,ev = self.sys.r_energies()
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
            F = self.sys.r_fock()
            F.oo = F.oo - numpy.diag(eo) # subtract diagonal
            F.vv = F.vv - numpy.diag(ev) # subtract diagonal
            t1shape = (nv,no)
            t1shape = (nv,nv,no,no)

            # get ERIs
            I = self.sys.r_int()
        else:
            # get orbital energies
            en = self.sys.r_energies_tot()
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
                F,I = cc_utils.rft_active_integrals(self.sys, en, focc, fvir, iocc, ivir)

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
                F,I = cc_utils.rft_integrals(self.sys, en, beta, mu)

                # get energy differences
                D1 = en[:,None] - en[None,:]
                D2 = en[:,None,None,None] + en[None,:,None,None] \
                        - en[None,None,:,None] - en[None,None,None,:]
                t1shape = (n,n)
                t2shape = (n,n,n,n)
        t1 = numpy.zeros(t1shape, dtype=F.vo.dtype)
        t2 = numpy.zeros(t2shape, dtype=I.vvoo.dtype)
        Eccn = 0.0

        def fRHS(var):
            t1,t2 = var
            k1s = -D1*t1 - F.vo.copy()
            k1d = -D2*t2 - I.vvoo.copy()
            cc_equations._r_Stanton(k1s,k1d,F,I,t1,t2,fac=-1.0)
            return [k1s,k1d]

        if self.saveT:
            self._save_T1(0, t1.copy())
            self._save_T2(0, t2.copy())

        for i in range(1,ng):
            # propagate
            h = self.ti[i] - self.ti[i - 1]
            d1,d2 = self._get_t_step(h, [t1,t2], fRHS)
            t1 += d1
            t2 += d2

            # compute free energy contribution
            Eccn += g[i]*rcc_energy(t1, t2, F.ov, I.oovv)/beta
            if self.saveT:
                self._save_T1(i, t1.copy())
                self._save_T2(i, t2.copy())

        self.G0 = E0
        self.G1 = E1
        self.Gcc = Eccn
        self.Gtot = E0 + E1 + Eccn
        if not self.saveT:
            self._save_T1(ng - 1,t1)
            self._save_T2(ng - 1,t2)

        return (Eccn+E01,Eccn)

    def _ccsd_lambda(self, rdm2=False, erel=False):
        beta = self.beta
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
            sfo = numpy.ones(no)
            sfv = numpy.ones(nv)
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
                sfo = numpy.sqrt(focc)
                sfv = numpy.sqrt(fvir)

            else:
                # get scaled integrals
                F,I = cc_utils.ft_integrals(self.sys, en, beta, mu)

                # get energy differences
                D1 = en[:,None] - en[None,:]
                D2 = en[:,None,None,None] + en[None,:,None,None] \
                        - en[None,None,:,None] - en[None,None,None,:]
                t1shape = (n,n)
                t2shape = (n,n,n,n)
                sfo = numpy.sqrt(fo)
                sfv = numpy.sqrt(fv)

        def fRHS(var):
            t1,t2 = var
            k1s = -D1*t1 - F.vo.copy()
            k1d = -D2*t2 - I.vvoo.copy()
            cc_equations._Stanton(k1s,k1d,F,I,t1,t2,fac=-1.0)
            return [(-1.0)*k1s,(-1.0)*k1d]

        def fLRHS(ttot,var):
            l1,l2 = var
            t1,t2 = ttot
            l1s = -D1.transpose((1,0))*l1 - F.ov.copy()
            l1d = -D2.transpose((2,3,0,1))*l2 - I.oovv.copy()
            cc_equations._Lambda_opt(l1s, l1d, F, I,
                    l1, l2, t1, t2, fac=-1.0)
            cc_equations._LS_TS(l1s,I,t1,fac=-1.0)
            return [l1s,l1d]

        t1b = self._read_T1(ng - 1)
        t2b = self._read_T2(ng - 1)
        nv, no = t1b.shape
        l1 = numpy.zeros((no,nv), dtype=t1b.dtype)
        l2 = numpy.zeros((no,no,nv,nv), dtype=t2b.dtype)
        pia = numpy.zeros((no,nv), dtype=l1.dtype)
        pji = g[ng - 1]*cc_equations.ccsd_1rdm_ji_opt(t1b,t2b,l1,l2)
        pba = g[ng - 1]*cc_equations.ccsd_1rdm_ba_opt(t1b,t2b,l1,l2)
        pai = g[ng - 1]*cc_equations.ccsd_1rdm_ai_opt(t1b,t2b,l1,l2)

        if rdm2:
            Pcdab = g[ng - 1]*cc_equations.ccsd_2rdm_cdab_opt(t1b, t2b, l1, l2)
            Pciab = g[ng - 1]*cc_equations.ccsd_2rdm_ciab_opt(t1b, t2b, l1, l2)
            Pbcai = g[ng - 1]*cc_equations.ccsd_2rdm_bcai_opt(t1b, t2b, l1, l2)
            Pijab = g[ng - 1]*l2
            Pbjai = g[ng - 1]*cc_equations.ccsd_2rdm_bjai_opt(t1b, t2b, l1, l2)
            Pabij = g[ng - 1]*cc_equations.ccsd_2rdm_abij_opt(t1b, t2b, l1, l2)
            Pjkai = g[ng - 1]*cc_equations.ccsd_2rdm_jkai_opt(t1b, t2b, l1, l2)
            Pkaij = g[ng - 1]*cc_equations.ccsd_2rdm_kaij_opt(t1b, t2b, l1, l2)
            Pklij = g[ng - 1]*cc_equations.ccsd_2rdm_klij_opt(t1b, t2b, l1, l2)

        if erel:
            self.rorbo = numpy.zeros(n, dtype=l1.dtype)
            self.rorbv = numpy.zeros(n, dtype=l1.dtype)
            x1 = numpy.zeros(l1.shape, dtype=l1.dtype)
            x2 = numpy.zeros(l2.shape, dtype=l2.dtype)
            def fXRHS(ltot,var):
                l1,l2 = ltot
                x1,x2 = var
                l1s = -D1.transpose((1,0))*x1 - l1
                l1d = -D2.transpose((2,3,0,1))*x2 - l2
                return [l1s,l1d]

        Eccn = g[ng - 1]*cc_energy(t1b, t2b, F.ov, I.oovv)/beta
        for i in range(1,ng):
            h = self.ti[ng - i] - self.ti[ng - i - 1]
            if not self.saveT:
                d1,d2 = self._get_t_step(h, [t1b,t2b], fRHS)
                t1e = t1b + d1
                t2e = t2b + d2
            else:
                t1e = self._read_T1(ng - i - 1)
                t2e = self._read_T2(ng - i - 1)
            ld1, ld2 = self._get_l_step(h, (l1,l2), (t1b,t2b), (t1e,t2e), fLRHS)
            if erel:
                dx1, dx2 = self._get_l_step(h, (x1,x2), (l1,l2), (l1 + ld1, l2 + ld2), fXRHS)
                x1 += dx1
                x2 += dx2
                d1test = -F.vo.copy()
                d2test = -I.vvoo.copy()
                cc_equations._Stanton(d1test,d2test,F,I,t1e,t2e,fac=-1.0)
            l1 += ld1
            l2 += ld2
            Eccn += g[ng - i - 1]*cc_energy(t1e, t2e, F.ov, I.oovv)/beta

            # increment the RDMs
            pia += g[ng - i - 1]*l1
            pji += g[ng - i - 1]*cc_equations.ccsd_1rdm_ji_opt(t1e,t2e,l1,l2)
            pba += g[ng - i - 1]*cc_equations.ccsd_1rdm_ba_opt(t1e,t2e,l1,l2)
            pai += g[ng - i - 1]*cc_equations.ccsd_1rdm_ai_opt(t1e,t2e,l1,l2)
            if rdm2:
                Pcdab += g[ng - 1 - i]*cc_equations.ccsd_2rdm_cdab_opt(t1e, t2e, l1, l2)
                Pciab += g[ng - 1 - i]*cc_equations.ccsd_2rdm_ciab_opt(t1e, t2e, l1, l2)
                Pbcai += g[ng - 1 - i]*cc_equations.ccsd_2rdm_bcai_opt(t1e, t2e, l1, l2)
                Pijab += g[ng - 1 - i]*l2
                Pbjai += g[ng - 1 - i]*cc_equations.ccsd_2rdm_bjai_opt(t1e, t2e, l1, l2)
                Pabij += g[ng - 1 - i]*cc_equations.ccsd_2rdm_abij_opt(t1e, t2e, l1, l2)
                Pjkai += g[ng - 1 - i]*cc_equations.ccsd_2rdm_jkai_opt(t1e, t2e, l1, l2)
                Pkaij += g[ng - 1 - i]*cc_equations.ccsd_2rdm_kaij_opt(t1e, t2e, l1, l2)
                Pklij += g[ng - 1 - i]*cc_equations.ccsd_2rdm_klij_opt(t1e, t2e, l1, l2)
            if erel:
                At1i = -(1.0/beta)*einsum('ia,ai->i',x1, d1test)
                At1a = -(1.0/beta)*einsum('ia,ai->a',x1, d1test)
                At2i = -(1.0/beta)*0.5*einsum('ijab,abij->i',x2, d2test)
                At2a = -(1.0/beta)*0.5*einsum('ijab,abij->a',x2, d2test)
                if self.athresh > 0.0:
                    self.rorbo[numpy.ix_(iocc)] -= g[ng - 1 - i]*(At1i + At2i)
                    self.rorbv[numpy.ix_(ivir)] += g[ng - 1 - i]*(At1a + At2a)
                else:
                    self.rorbo -= g[ng - 1 - i]*(At1i + At2i)
                    self.rorbv += g[ng - 1 - i]*(At1a + At2a)

            t1b = t1e
            t2b = t2e

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
        self.ndia = numpy.einsum('ia,i,a->ia',self.dia,sfo,sfv)
        self.ndba = numpy.einsum('ba,b,a->ba',self.dba,sfv,sfv)
        self.ndji = numpy.einsum('ji,j,i->ji',self.dji,sfo,sfo)
        self.ndai = numpy.einsum('ai,a,i->ai',self.dai,sfv,sfo)
        if rdm2:
            self.P2 = (Pcdab, Pciab, Pbcai, Pijab, Pbjai, Pabij, Pjkai, Pkaij, Pklij)

        return (Eccn+E01,Eccn)

    def _uccsd_lambda(self, rdm2=False, erel=False):
        beta = self.beta
        mu = self.mu if self.finite_T else None

        # get time-grid
        ng = self.ngrid
        ti = self.ti
        g = self.g

        if not self.finite_T:
            # create energies in spin-orbital basis
            eoa,eva,eob,evb = self.sys.u_energies()
            noa = eoa.shape[0]
            nob = eob.shape[0]
            nva = eva.shape[0]
            nvb = evb.shape[0]
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
            sfoa = numpy.ones(noa)
            sfob = numpy.ones(nob)
            sfva = numpy.ones(nva)
            sfvb = numpy.ones(nvb)
        else:
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
            foa = ft_utils.ff(beta, ea, mu)
            fva = ft_utils.ffv(beta, ea, mu)
            fob = ft_utils.ff(beta, eb, mu)
            fvb = ft_utils.ffv(beta, eb, mu)

            beta = self.beta
            mu = self.mu
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
                noa = len(focca)
                nva = len(fvira)
                nob = len(foccb)
                nvb = len(fvirb)
                nacta = noa + nva - na
                nactb = nob + nvb - nb
                if self.iprint > 0:
                    print("FT-UCCSD orbital info:")
                    print('  nocca: {:d}'.format(noa))
                    print('  nvira: {:d}'.format(nva))
                    print('  nacta: {:d}'.format(nacta))
                    print('  noccb: {:d}'.format(noa))
                    print('  nvirb: {:d}'.format(nva))
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
                sfoa = numpy.sqrt(focca)
                sfob = numpy.sqrt(foccb)
                sfva = numpy.sqrt(fvira)
                sfvb = numpy.sqrt(fvirb)

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
                T1ashape = (na,na)
                T1bshape = (nb,nb)
                T2ashape = (na,na,na,na)
                T2bshape = (nb,nb,nb,nb)
                noa = na
                nva = na
                nob = nb
                nvb = nb
                sfoa = numpy.sqrt(foa)
                sfob = numpy.sqrt(fob)
                sfva = numpy.sqrt(fva)
                sfvb = numpy.sqrt(fvb)

        def fRHS(var):
            t1a,t1b,t2aa,t2ab,t2bb = var
            k1sa = -D1a*t1a - Fa.vo.copy()
            k1sb = -D1b*t1b - Fb.vo.copy()
            k1daa = -D2aa*t2aa - Ia.vvoo.copy()
            k1dab = -D2ab*t2ab - Iabab.vvoo.copy()
            k1dbb = -D2bb*t2bb - Ib.vvoo.copy()
            cc_equations._u_Stanton(k1sa, k1sb, k1daa, k1dab, k1dbb,
                    Fa, Fb, Ia, Ib, Iabab, (t1a,t1b), (t2aa,t2ab,t2bb), fac=-1.0)
            return [(-1)*k1sa,(-1)*k1sb,(-1)*k1daa,(-1)*k1dab,(-1)*k1dbb]

        def fLRHS(ttot,var):
            l1a,l1b,l2aa,l2ab,l2bb = var
            t1a,t1b,t2aa,t2ab,t2bb = ttot
            l1sa = -D1a.transpose((1,0))*l1a - Fa.ov.copy()
            l1sb = -D1b.transpose((1,0))*l1b - Fb.ov.copy()
            l1daa = -D2aa.transpose((2,3,0,1))*l2aa - Ia.oovv.copy()
            l1dab = -D2ab.transpose((2,3,0,1))*l2ab - Iabab.oovv.copy()
            l1dbb = -D2bb.transpose((2,3,0,1))*l2bb - Ib.oovv.copy()
            cc_equations._uccsd_Lambda_opt(l1sa, l1sb, l1daa, l1dab, l1dbb,
                    Fa, Fb, Ia, Ib, Iabab, (l1a,l1b), (l2aa,l2ab,l2bb),
                    (t1a,t1b), (t2aa,t2ab,t2bb), fac=-1.0)
            cc_equations._u_LS_TS(l1sa, l1sb, Ia, Ib, Iabab, t1a, t1b, fac=-1.0)
            return [l1sa,l1sb,l1daa,l1dab,l1dbb]

        t1da,t1db = self._read_T1(ng - 1)
        t2daa,t2dab,t2dbb = self._read_T2(ng - 1)
        l1a = numpy.zeros((noa,nva), dtype=t1da.dtype)
        l1b = numpy.zeros((nob,nvb), dtype=t1db.dtype)
        l2aa = numpy.zeros((noa,noa,nva,nva), dtype=t2daa.dtype)
        l2ab = numpy.zeros((noa,nob,nva,nvb), dtype=t2dab.dtype)
        l2bb = numpy.zeros((nob,nob,nvb,nvb), dtype=t2dbb.dtype)
        pia = numpy.zeros((noa,nva), dtype=l1a.dtype)
        pIA = numpy.zeros((nob,nvb), dtype=l1b.dtype)
        pji,pJI = cc_equations.uccsd_1rdm_ji(
                t1da, t1db, t2daa, t2dab, t2dbb, l1a, l1b, l2aa, l2ab, l2bb)
        pba,pBA = cc_equations.uccsd_1rdm_ba(
                t1da, t1db, t2daa, t2dab, t2dbb, l1a, l1b, l2aa, l2ab, l2bb)
        pai,pAI = cc_equations.uccsd_1rdm_ai(
                t1da, t1db, t2daa, t2dab, t2dbb, l1a, l1b, l2aa, l2ab, l2bb)
        pji *= g[ng - 1]
        pJI *= g[ng - 1]
        pba *= g[ng - 1]
        pBA *= g[ng - 1]
        pai *= g[ng - 1]
        pAI *= g[ng - 1]

        if rdm2:
            Pijab = numpy.zeros((noa,noa,nva,nva), dtype=l2aa.dtype)
            PiJaB = numpy.zeros((noa,nob,nva,nvb), dtype=l2ab.dtype)
            PIJAB = numpy.zeros((nob,nob,nvb,nvb), dtype=l2bb.dtype)

            Pcdab_tot = cc_equations.uccsd_2rdm_cdab(t1da, t1db, t2daa, t2dab, t2dbb,
                    l1a, l1b, l2aa, l2ab, l2bb)
            Pcdab = g[ng - 1]*Pcdab_tot[0]
            PCDAB = g[ng - 1]*Pcdab_tot[1]
            PcDaB = g[ng - 1]*Pcdab_tot[2]

            Pciab_tot = cc_equations.uccsd_2rdm_ciab(t1da, t1db, t2daa, t2dab, t2dbb,
                    l1a, l1b, l2aa, l2ab, l2bb)
            Pciab = g[ng - 1]*Pciab_tot[0]
            PCIAB = g[ng - 1]*Pciab_tot[1]
            PcIaB = g[ng - 1]*Pciab_tot[2]
            PCiAb = g[ng - 1]*Pciab_tot[3]

            Pbcai_tot = cc_equations.uccsd_2rdm_bcai(t1da, t1db, t2daa, t2dab, t2dbb,
                    l1a, l1b, l2aa, l2ab, l2bb)
            Pbcai = g[ng - 1]*Pbcai_tot[0]
            PBCAI = g[ng - 1]*Pbcai_tot[1]
            PbCaI = g[ng - 1]*Pbcai_tot[2]
            PBcAi = g[ng - 1]*Pbcai_tot[3]

            Pbjai_tot = cc_equations.uccsd_2rdm_bjai(t1da, t1db, t2daa, t2dab, t2dbb,
                    l1a, l1b, l2aa, l2ab, l2bb)
            Pbjai = g[ng - 1]*Pbjai_tot[0]
            PBJAI = g[ng - 1]*Pbjai_tot[1]
            PbJaI = g[ng - 1]*Pbjai_tot[2]
            PbJAi = g[ng - 1]*Pbjai_tot[3]
            PBjaI = g[ng - 1]*Pbjai_tot[4]
            PBjAi = g[ng - 1]*Pbjai_tot[5]

            Pabij_tot = cc_equations.uccsd_2rdm_abij(t1da, t1db, t2daa, t2dab, t2dbb,
                    l1a, l1b, l2aa, l2ab, l2bb)
            Pabij = g[ng - 1]*Pabij_tot[0]
            PABIJ = g[ng - 1]*Pabij_tot[1]
            PaBiJ = g[ng - 1]*Pabij_tot[2]

            Pjkai_tot = cc_equations.uccsd_2rdm_jkai(t1da, t1db, t2daa, t2dab, t2dbb,
                    l1a, l1b, l2aa, l2ab, l2bb)
            Pjkai = g[ng - 1]*Pjkai_tot[0]
            PJKAI = g[ng - 1]*Pjkai_tot[1]
            PjKaI = g[ng - 1]*Pjkai_tot[2]
            PJkAi = g[ng - 1]*Pjkai_tot[3]

            Pkaij_tot = cc_equations.uccsd_2rdm_kaij(t1da, t1db, t2daa, t2dab, t2dbb,
                    l1a, l1b, l2aa, l2ab, l2bb)
            Pkaij = g[ng - 1]*Pkaij_tot[0]
            PKAIJ = g[ng - 1]*Pkaij_tot[1]
            PkAiJ = g[ng - 1]*Pkaij_tot[2]
            PKaIj = g[ng - 1]*Pkaij_tot[3]

            Pklij_tot = cc_equations.uccsd_2rdm_klij(t1da, t1db, t2daa, t2dab, t2dbb,
                    l1a, l1b, l2aa, l2ab, l2bb)
            Pklij = g[ng - 1]*Pklij_tot[0]
            PKLIJ = g[ng - 1]*Pklij_tot[1]
            PkLiJ = g[ng - 1]*Pklij_tot[2]

        if erel:
            rorbo_a = numpy.zeros(na, dtype=l1a.dtype)
            rorbo_b = numpy.zeros(nb, dtype=l1b.dtype)
            rorbv_a = numpy.zeros(na, dtype=l1a.dtype)
            rorbv_b = numpy.zeros(nb, dtype=l1b.dtype)
            x1a = numpy.zeros(l1a.shape, dtype=l1a.dtype)
            x1b = numpy.zeros(l1b.shape, dtype=l1b.dtype)
            x2aa = numpy.zeros(l2aa.shape, dtype=l2aa.dtype)
            x2ab = numpy.zeros(l2ab.shape, dtype=l2ab.dtype)
            x2bb = numpy.zeros(l2bb.shape, dtype=l2bb.dtype)
            def fXRHS(ltot,var):
                l1a,l1b,l2aa,l2ab,l2bb = ltot
                x1a,x1b,x2aa,x2ab,x2bb = var
                l1sa = -D1a.transpose((1,0))*x1a - l1a
                l1sb = -D1b.transpose((1,0))*x1b - l1b
                l1daa = -D2aa.transpose((2,3,0,1))*x2aa - l2aa
                l1dab = -D2ab.transpose((2,3,0,1))*x2ab - l2ab
                l1dbb = -D2bb.transpose((2,3,0,1))*x2bb - l2bb
                return [l1sa,l1sb,l1daa,l1dab,l1dbb]

        Eccn = g[ng - 1]*ucc_energy((t1da,t1db), (t2daa,t2dab,t2dbb),
                Fa.ov, Fb.ov, Ia.oovv, Ib.oovv, Iabab.oovv)/beta
        for i in range(1,ng):
            h = self.ti[ng - i] - self.ti[ng - i - 1]
            if not self.saveT:
                d1a,d1b,d2aa,d2ab,d2bb = self._get_t_step(h, [t1da,t1db,t2daa,t2dab,t2dbb], fRHS)
                t1ea = t1da + d1a
                t1eb = t1db + d1b
                t2eaa = t2daa + d2aa
                t2eab = t2dab + d2ab
                t2ebb = t2dbb + d2bb
            else:
                t1ea,t1eb = self._read_T1(ng - i - 1)
                t2eaa,t2eab,t2ebb = self._read_T2(ng - i - 1)
            ld1a, ld1b, ld2aa, ld2ab, ld2bb = self._get_l_step(
                    h, (l1a,l1b,l2aa,l2ab,l2bb), (t1da,t1db,t2daa,t2dab,t2dbb), (t1ea,t1eb,t2eaa,t2eab,t2ebb), fLRHS)
            if erel:
                dx1a, dx1b, dx2aa, dx2ab, dx2bb = self._get_l_step(h, (x1a,x1b,x2aa,x2ab,x2bb),
                        (l1a,l1b,l2aa,l2ab,l2bb), (l1a + ld1a, l1b + ld1b, l2aa + ld2aa, l2ab + ld2ab, l2bb + ld2bb), fXRHS)
                x1a += dx1a
                x1b += dx1b
                x2aa += dx2aa
                x2ab += dx2ab
                x2bb += dx2bb
                d1atest = -Fa.vo.copy()
                d1btest = -Fb.vo.copy()
                d2aatest = -Ia.vvoo.copy()
                d2abtest = -Iabab.vvoo.copy()
                d2bbtest = -Ib.vvoo.copy()
                cc_equations._u_Stanton(d1atest,d1btest,d2aatest,d2abtest,d2bbtest,
                        Fa,Fb,Ia,Ib,Iabab,(t1ea,t1eb),(t2eaa,t2eab,t2ebb),fac=-1.0)
            l1a += ld1a
            l1b += ld1b
            l2aa += ld2aa
            l2ab += ld2ab
            l2bb += ld2bb
            Eccn += g[ng - i - 1]*ucc_energy((t1ea,t1eb), (t2eaa,t2eab,t2ebb),
                    Fa.ov, Fb.ov, Ia.oovv, Ib.oovv, Iabab.oovv)/beta

            # increment the RDMs
            pia += g[ng - i - 1]*l1a
            pIA += g[ng - i - 1]*l1b
            ji_inc = cc_equations.uccsd_1rdm_ji(
                    t1ea, t1eb, t2eaa, t2eab, t2ebb, l1a, l1b, l2aa, l2ab, l2bb)
            ba_inc = cc_equations.uccsd_1rdm_ba(
                    t1ea, t1eb, t2eaa, t2eab, t2ebb, l1a, l1b, l2aa, l2ab, l2bb)
            ai_inc = cc_equations.uccsd_1rdm_ai(
                    t1ea, t1eb, t2eaa, t2eab, t2ebb, l1a, l1b, l2aa, l2ab, l2bb)
            pji += g[ng - i - 1]*ji_inc[0]
            pJI += g[ng - i - 1]*ji_inc[1]
            pba += g[ng - i - 1]*ba_inc[0]
            pBA += g[ng - i - 1]*ba_inc[1]
            pai += g[ng - i - 1]*ai_inc[0]
            pAI += g[ng - i - 1]*ai_inc[1]

            if rdm2:
                cdab_tot = cc_equations.uccsd_2rdm_cdab(
                        t1ea, t1eb, t2eaa, t2eab, t2ebb, l1a, l1b, l2aa, l2ab, l2bb)
                ciab_tot = cc_equations.uccsd_2rdm_ciab(
                        t1ea, t1eb, t2eaa, t2eab, t2ebb, l1a, l1b, l2aa, l2ab, l2bb)
                bcai_tot = cc_equations.uccsd_2rdm_bcai(
                        t1ea, t1eb, t2eaa, t2eab, t2ebb, l1a, l1b, l2aa, l2ab, l2bb)
                bjai_tot = cc_equations.uccsd_2rdm_bjai(
                        t1ea, t1eb, t2eaa, t2eab, t2ebb, l1a, l1b, l2aa, l2ab, l2bb)
                abij_tot = cc_equations.uccsd_2rdm_abij(
                        t1ea, t1eb, t2eaa, t2eab, t2ebb, l1a, l1b, l2aa, l2ab, l2bb)
                jkai_tot = cc_equations.uccsd_2rdm_jkai(
                        t1ea, t1eb, t2eaa, t2eab, t2ebb, l1a, l1b, l2aa, l2ab, l2bb)
                kaij_tot = cc_equations.uccsd_2rdm_kaij(
                        t1ea, t1eb, t2eaa, t2eab, t2ebb, l1a, l1b, l2aa, l2ab, l2bb)
                klij_tot = cc_equations.uccsd_2rdm_klij(
                        t1ea, t1eb, t2eaa, t2eab, t2ebb, l1a, l1b, l2aa, l2ab, l2bb)

                Pijab += g[ng - i - 1]*l2aa
                PiJaB += g[ng - i - 1]*l2ab
                PIJAB += g[ng - i - 1]*l2bb

                Pcdab += g[ng - i - 1]*cdab_tot[0]
                PCDAB += g[ng - i - 1]*cdab_tot[1]
                PcDaB += g[ng - i - 1]*cdab_tot[2]

                Pciab += g[ng - i - 1]*ciab_tot[0]
                PCIAB += g[ng - i - 1]*ciab_tot[1]
                PcIaB += g[ng - i - 1]*ciab_tot[2]
                PCiAb += g[ng - i - 1]*ciab_tot[3]

                Pbcai += g[ng - i - 1]*bcai_tot[0]
                PBCAI += g[ng - i - 1]*bcai_tot[1]
                PbCaI += g[ng - i - 1]*bcai_tot[2]
                PBcAi += g[ng - i - 1]*bcai_tot[3]

                Pbjai += g[ng - i - 1]*bjai_tot[0]
                PBJAI += g[ng - i - 1]*bjai_tot[1]
                PbJaI += g[ng - i - 1]*bjai_tot[2]
                PbJAi += g[ng - i - 1]*bjai_tot[3]
                PBjaI += g[ng - i - 1]*bjai_tot[4]
                PBjAi += g[ng - i - 1]*bjai_tot[5]

                Pabij += g[ng - i - 1]*abij_tot[0]
                PABIJ += g[ng - i - 1]*abij_tot[1]
                PaBiJ += g[ng - i - 1]*abij_tot[2]

                Pjkai += g[ng - i - 1]*jkai_tot[0]
                PJKAI += g[ng - i - 1]*jkai_tot[1]
                PjKaI += g[ng - i - 1]*jkai_tot[2]
                PJkAi += g[ng - i - 1]*jkai_tot[3]

                Pkaij += g[ng - i - 1]*kaij_tot[0]
                PKAIJ += g[ng - i - 1]*kaij_tot[1]
                PkAiJ += g[ng - i - 1]*kaij_tot[2]
                PKaIj += g[ng - i - 1]*kaij_tot[3]

                Pklij += g[ng - i - 1]*klij_tot[0]
                PKLIJ += g[ng - i - 1]*klij_tot[1]
                PkLiJ += g[ng - i - 1]*klij_tot[2]

            if erel:
                At1i_a = -(1.0/beta)*einsum('ia,ai->i',x1a, d1atest)
                At1i_b = -(1.0/beta)*einsum('ia,ai->i',x1b, d1btest)
                At1a_a = -(1.0/beta)*einsum('ia,ai->a',x1a, d1atest)
                At1a_b = -(1.0/beta)*einsum('ia,ai->a',x1b, d1btest)
                At2i_a = -(1.0/beta)*0.5*einsum('ijab,abij->i',x2aa, d2aatest)
                At2i_b = -(1.0/beta)*0.5*einsum('ijab,abij->i',x2bb, d2bbtest)
                At2i_a -= (1.0/beta)*einsum('ijab,abij->i',x2ab, d2abtest)
                At2i_b -= (1.0/beta)*einsum('ijab,abij->j',x2ab, d2abtest)
                At2a_a = -(1.0/beta)*0.5*einsum('ijab,abij->a',x2aa, d2aatest)
                At2a_b = -(1.0/beta)*0.5*einsum('ijab,abij->a',x2bb, d2bbtest)
                At2a_a -= (1.0/beta)*einsum('ijab,abij->a',x2ab, d2abtest)
                At2a_b -= (1.0/beta)*einsum('ijab,abij->b',x2ab, d2abtest)
                if self.athresh > 0.0:
                    rorbo_a[numpy.ix_(iocca)] -= g[ng - 1 - i]*(At1i_a + At2i_a)
                    rorbo_b[numpy.ix_(ioccb)] -= g[ng - 1 - i]*(At1i_b + At2i_b)
                    rorbv_a[numpy.ix_(ivira)] += g[ng - 1 - i]*(At1a_a + At2a_a)
                    rorbv_b[numpy.ix_(ivirb)] += g[ng - 1 - i]*(At1a_b + At2a_b)
                else:
                    rorbo_a -= g[ng - 1 - i]*(At1i_a + At2i_a)
                    rorbo_b -= g[ng - 1 - i]*(At1i_b + At2i_b)
                    rorbv_a += g[ng - 1 - i]*(At1a_a + At2a_a)
                    rorbv_b += g[ng - 1 - i]*(At1a_b + At2a_b)

            t1da = t1ea
            t1db = t1eb
            t2daa = t2eaa
            t2dab = t2eab
            t2dbb = t2ebb

        G0 = E0
        G1 = E1
        Gcc = Eccn
        Gtot = E0 + E1 + Eccn
        self.L1 = (l1a,l1b)
        self.L2 = (l2aa,l2ab,l2bb)
        self.dia = (pia,pIA)
        self.dji = (pji,pJI)
        self.dba = (pba,pBA)
        self.dai = (pai,pAI)
        self.ndia = (numpy.einsum('ia,i,a->ia',self.dia[0],sfoa,sfva),
                numpy.einsum('ia,i,a->ia',self.dia[1],sfob,sfvb))
        self.ndba = (numpy.einsum('ba,b,a->ba',self.dba[0],sfva,sfva),
                numpy.einsum('ba,b,a->ba',self.dba[1],sfvb,sfvb))
        self.ndji = (numpy.einsum('ji,j,i->ji',self.dji[0],sfoa,sfoa),
                numpy.einsum('ji,j,i->ji',self.dji[1],sfob,sfob))
        self.ndai = (numpy.einsum('ai,a,i->ai',self.dai[0],sfva,sfoa),
                numpy.einsum('ai,a,i->ai',self.dai[1],sfvb,sfob))
        if rdm2:
            self.P2 = ((Pcdab,PCDAB,PcDaB),
                    (Pciab,PCIAB,PcIaB,PCiAb),
                    (Pbcai,PBCAI,PbCaI,PBcAi),
                    (Pijab,PIJAB,PiJaB),
                    (Pbjai,PBJAI,PbJaI,PbJAi,PBjaI,PBjAi),
                    (Pabij,PABIJ,PaBiJ),
                    (Pjkai,PJKAI,PjKaI,PJkAi),
                    (Pkaij,PKAIJ,PkAiJ,PKaIj),
                    (Pklij,PKLIJ,PkLiJ))
        if erel:
            self.rorbo = [rorbo_a, rorbo_b]
            self.rorbv = [rorbv_a, rorbv_b]

        return (Eccn+E01,Eccn)

    def _g_gderiv_approx(self):
        # temperature info
        beta = self.beta
        mu = self.mu

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)

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
        ng = self.ngrid
        t1 = self._read_T1(ng - 1)
        t2 = self._read_T2(ng - 1)
        t2_temp = 0.25*t2 + 0.5*numpy.einsum('ai,bj->abij',t1,t1)
        Es1 = numpy.einsum('ai,ia->',t1,F.ov)
        Es2 = numpy.einsum('abij,ijab->',t2_temp,I.oovv)
        return (Es1 + Es2)/beta

    def _u_gderiv_approx(self):
        # temperature info
        beta = self.beta
        mu = self.mu

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        ea,eb = self.sys.u_energies_tot()
        na = ea.shape[0]
        nb = eb.shape[0]

        ng = self.ngrid
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

        T1a,T1b = self._read_T1(ng - 1)
        T2aa,T2ab,T2bb = self._read_T2(ng - 1)
        t2aa_temp = 0.25*T2aa + 0.5*einsum('ai,bj->abij',T1a,T1a)
        t2bb_temp = 0.25*T2bb + 0.5*einsum('ai,bj->abij',T1b,T1b)
        t2ab_temp = T2ab + einsum('ai,bj->abij',T1a,T1b)

        Es1 = einsum('ai,ia->',T1a,Fa.ov)
        Es1 += einsum('ai,ia->',T1b,Fb.ov)
        Es2 = einsum('abij,ijab->',t2aa_temp,Ia.oovv)
        Es2 += einsum('abij,ijab->',t2ab_temp,Iabab.oovv)
        Es2 += einsum('abij,ijab->',t2bb_temp,Ib.oovv)

        return (Es1 + Es2) / beta

    def _g_ft_ron(self):
        # temperature info
        beta = self.beta
        mu = self.mu

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)

        # get ON correction to HF free energy
        self.ron1 = self.sys.g_mp1_den()

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
        rono = numpy.zeros(n, dtype=self.dji.dtype)
        ronv = numpy.zeros(n, dtype=self.dba.dtype)

        # perturbed ON contribution to Fock matrix
        Fd = self.sys.g_fock_d_den()
        if self.athresh > 0.0:
            rono += cc_utils.g_Fd_on_active(
                    Fd, iocc, ivir, self.ndia, self.ndba, self.ndji, self.ndai)
        else:
            rono += cc_utils.g_Fd_on(Fd, self.ndia, self.ndba, self.ndji, self.ndai)

        # Add contributions from occupation number relaxation
        jitemp = numpy.zeros(nocc, dtype=self.dji.dtype) if self.athresh > 0.0 else numpy.zeros(n, dtype=self.dji.dtype)
        batemp = numpy.zeros(nvir, dtype=self.dba.dtype) if self.athresh > 0.0 else numpy.zeros(n, dtype=self.dba.dtype)
        cc_utils.g_d_on_oo(dso, F, I, self.dia, self.dji, self.dai, self.P2, jitemp)
        cc_utils.g_d_on_vv(dsv, F, I, self.dia, self.dba, self.dai, self.P2, batemp)

        if self.athresh > 0.0:
            rono[numpy.ix_(iocc)] += jitemp
            ronv[numpy.ix_(ivir)] += batemp
        else:
            rono += jitemp
            ronv += batemp
        self.rono = rono
        self.ronv = ronv

    def _u_ft_ron(self):
        # temperature info
        beta = self.beta
        mu = self.mu

        # ON correction to HF free energy
        mp1da,mp1db = self.sys.u_mp1_den()
        self.ron1 = [mp1da, mp1db]

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
        dta = Fdaa.dtype
        dtb = Fdbb.dtype
        ronoa = numpy.zeros(na, dtype=dta)
        ronva = numpy.zeros(na, dtype=dta)
        ronob = numpy.zeros(nb, dtype=dtb)
        ronvb = numpy.zeros(nb, dtype=dtb)
        if self.athresh > 0.0:
            temp = cc_utils.u_Fd_on_active(
                    Fdaa, Fdab, Fdba, Fdbb, iocca, ivira, ioccb, ivirb,
                    self.ndia, self.ndba, self.ndji, self.ndai)
            ronoa += temp[0]
            ronob += temp[1]
        else:
            temp = cc_utils.u_Fd_on(
                    Fdaa, Fdab, Fdba, Fdbb, self.ndia, self.ndba, self.ndji, self.ndai)
            ronoa += temp[0]
            ronob += temp[1]

        # Add contributions from occupation number relaxation
        jitempa = numpy.zeros(nocca, dtype=dta) if self.athresh > 0.0 else numpy.zeros(na, dtype=dta)
        batempa = numpy.zeros(nvira, dtype=dta) if self.athresh > 0.0 else numpy.zeros(na, dtype=dta)
        jitempb = numpy.zeros(noccb, dtype=dtb) if self.athresh > 0.0 else numpy.zeros(nb, dtype=dtb)
        batempb = numpy.zeros(nvirb, dtype=dtb) if self.athresh > 0.0 else numpy.zeros(nb, dtype=dtb)
        cc_utils.u_d_on_oo(
                dsoa, dsob, Fa, Fb, Ia, Ib, Iabab,
                self.dia, self.dji, self.dai, self.P2, jitempa, jitempb)
        cc_utils.u_d_on_vv(
                dsva, dsvb, Fa, Fb, Ia, Ib, Iabab,
                self.dia, self.dba, self.dai, self.P2, batempa, batempb)

        if self.athresh > 0.0:
            ronoa[numpy.ix_(iocca)] += jitempa
            ronob[numpy.ix_(ioccb)] += jitempb
            ronva[numpy.ix_(ivira)] += batempa
            ronvb[numpy.ix_(ivirb)] += batempb
        else:
            ronoa += jitempa
            ronob += jitempb
            ronva += batempa
            ronvb += batempb
        self.rono = [ronoa, ronob]
        self.ronv = [ronva, ronvb]
