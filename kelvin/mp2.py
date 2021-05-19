import logging
import numpy
from cqcpy import ft_utils
from . import ft_mp
from . import zt_mp


class MP2(object):
    """MP2 driver.

    Attributes:
        sys: System object.
        T (float): Temperature.
        mu (float): Chemical potential.
        iprint (int): Print level.
        saveT (bool): Save T-amplitudes.
    """
    def __init__(self, sys, T=0, mu=0, iprint=0, saveT=False):
        self.T = T
        self.mu = mu
        self.finite_T = False if T == 0 else True
        if self.finite_T:
            self.beta = 1/T
            if not sys.verify(self.T,self.mu):
                raise Exception("Sytem temperature inconsistent with MP2 temp")
        else:
            self.beta = 1.0e20
        self.sys = sys
        self.iprint = iprint
        self.saveT = saveT

    def run(self):
        if self.finite_T:
            logging.info('Running MP2 at an electronic temperature of %f K'
                % ft_utils.HtoK(self.T))
            return self._ft_mp2()
        else:
            logging.info('Running MP2 at zero Temperature')
            return self._mp2()

    def _mp2(self):
        eo,ev = self.sys.g_energies()
        no = eo.shape[0]
        nv = ev.shape[0]
        n = no + nv

        # compute zero order energy
        En = self.sys.const_energy()
        E0 = zt_mp.mp0(eo) + En

        # compute requisite memory
        mem = no + nv + no*nv + n*n\
            + 3*no*no*nv*nv
        mem_mb = mem*8/1024.0/1024.0
        assert(mem_mb < 2000)
        logging.info('  RMP2 will use %f mb' % mem_mb)

        # get ERIs
        I = self.sys.g_aint(code=4)

        # get Fock matrix
        F = self.sys.g_fock()

        # get first order energy
        E1 = self.sys.get_mp1()

        # compute second order energy
        if self.saveT:
            E2,T1,T2 = zt_mp.mp2(eo,ev,F.vo,I.vvoo,returnT=True)
            self.T1 = T1
            self.T2 = T2
        else:
            E2 = zt_mp.mp2(eo,ev,F.vo,I.vvoo)

        # save and return energies
        self.E0 = E0
        self.E1 = E1
        self.E2 = E2
        return (E0,E1,E2)

    def _ft_mp2(self):
        mu = self.mu

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(self.beta, en, mu)

        # compute zero order quantities
        En = self.sys.const_energy()
        g0 = ft_utils.GP0(self.beta, en, mu)
        E0 = ft_mp.mp0(g0) + En

        # compute requisite memory
        n = en.shape[0]
        mem1e = 4*n*n # include memory for D1
        mem2e = 2*n*n*n*n # include memory for D2
        mem_mb = (mem1e + mem2e)*8.0/1024.0/1024.0
        assert(mem_mb < 4000)
        logging.info('  FT-RMP2 will use %f mb' % mem_mb)

        # get FT Fock matrix
        fmo = self.sys.g_fock_tot()
        fmo = fmo - numpy.diag(en)

        # compute first order energy
        E1 = self.sys.get_mp1()

        # get ERIs
        eri = self.sys.g_aint_tot()

        # compute second order energy
        if self.saveT:
            E2,T1n,T2n,T1a,T2a = ft_mp.mp2_sep(en, fo, fmo, eri, self.T, returnT=True)
            self.T1n = T1n
            self.T1a = T1a
            self.T2n = T2n
            self.T2a = T2a
        else:
            E2 = ft_mp.mp2(en, fo, fmo, eri, self.T)

        # save a return energies
        self.E0 = E0
        self.E1 = E1
        self.E2 = E2
        return (E0,E1,E2)
