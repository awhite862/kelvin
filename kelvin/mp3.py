import numpy
from cqcpy import ft_utils
from . import ft_mp
from . import zt_mp

class mp3(object):
    """MP3 driver.

    Attributes:
        sys: System object.
        T (float): Temperature.
        mu (float): Chemical potential.
        iprint (int): Print level.
    """
    def __init__(self, sys, T=0,mu=0,iprint=0):
        self.T = T
        self.mu = mu
        self.iprint = iprint
        self.finite_T = False if T == 0 else True
        if self.finite_T:
            self.beta = 1/T
            if not sys.verify(self.T,self.mu):
                raise Exception("Sytem temperature inconsistent with MP2 temp")
        else:
            self.beta = 1.0e20
        self.sys = sys

    def run(self):
        if self.finite_T:
            print('Running MP3 at an electronic temperature of %f K'
                % ft_utils.HtoK(self.T))
            return self._ft_mp3()
        else:
            if self.iprint > 0:
                print('Running MP3 at zero Temperature')
            return self._mp3()

    def _mp3(self):
        # create orbitals and energies in spin-orbital basis
        eo,ev = self.sys.g_energies()

        # compute requisite memory
        no = eo.shape[0]
        nv = ev.shape[0]
        mem1e = no*no + 3*no*nv + nv*nv  # include memory for D1
        mem2e = 4*no*no*nv*nv + nv*nv*nv*nv + 2*nv*nv*nv*no + \
                2*nv*no*no*no + no*no*no*no # include memory for D2
        mem_mb = (mem1e + mem2e)*8.0/1024.0/1024.0
        assert(mem_mb < 4000)
        if self.iprint > 0:
            print('  RMP3 will use %f mb' % mem_mb)

        # compute zero order energy
        En = self.sys.const_energy()
        E0 = zt_mp.mp0(eo) + En

        # get ERIs
        I = self.sys.g_aint()

        # get Fock matrix
        F = self.sys.g_fock()
        F.oo = F.oo - numpy.diag(eo) # subtract diagonal
        F.vv = F.vv - numpy.diag(ev) # subtract diagonal

        # get first order energy
        E1 = self.sys.get_mp1()

        # compute 2nd and 3rd order energies
        E2 = zt_mp.mp2(eo,ev,F.vo,I.vvoo)
        E3 = zt_mp.mp3(eo,ev,F,I)

        # save and return energies
        self.E0 = E0
        self.E1 = E1
        self.E2 = E2
        self.E3 = E3
        return (E0,E1,E2,E3)

    def _ft_mp3(self):
        mu = self.mu
        T = self.T

        # get energies and occupation numbers
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(self.beta, en, mu)
        fv = ft_utils.ffv(self.beta, en, mu)

        # compute requisite memory
        n = en.shape[0]
        mem1e = 5*n*n # include memory for D1
        mem2e = 3*n*n*n*n # include memory for D2
        mem_mb = (mem1e + mem2e)*8.0/1024.0/1024.0
        assert(mem_mb < 4000)
        if self.iprint > 0:
            print('  FT-RMP3 will use %f mb' % mem_mb)

        # compute zero order quantities
        En = self.sys.const_energy()
        g0 = ft_utils.GP0(self.beta, en, mu)
        E0 = ft_mp.mp0(g0) + En

        # get FT Fock matrix
        fmo = self.sys.g_fock_tot()
        fmo = fmo - numpy.diag(en)

        # compute first order energy
        E1 = self.sys.get_mp1()

        # get ERIs
        eri = self.sys.g_aint_tot()

        # compute second and third order energies
        E2 = ft_mp.mp2(en, fo, fmo, eri, T)
        #E3 = ft_mp.mp3_new(en, fo, fmo, eri, T) TODO: fix this!
        E23 = ft_mp.mp23_int(en, fo, fv, fmo, eri, T, ngrid=100)
        E3 = E23 - E2

        # save and return energies
        self.E0 = E0
        self.E1 = E1
        self.E2 = E2
        self.E3 = E3
        return (E0,E1,E2,E3)
