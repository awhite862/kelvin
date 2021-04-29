import numpy
from pyscf.fci import cistring
from pyscf.fci import fci_slow
from cqcpy import ft_utils
from . import ft_mp

class fci(object):
    """Full configuration interaction (FCI) driver.

    Attributes:
        sys: System object.
        T (float): Temperature.
        mu (float): Chemical potential.
        iprint (int): Print level.
        lam (float): Interaction scale factor.
        nalpha (int): Number of alpha electrons.
        nbeta (int): Number of beta electrons.
    """
    def __init__(self, sys, T=0, mu=0, iprint=0, lam=1.0,
        nalpha=None,nbeta=None):

        self.sys = sys
        self.T = T
        self.mu = mu
        self.iprint = iprint
        self.nalpha = nalpha
        self.nbeta = nbeta
        self.lam = 1.0
        if nalpha is not None and nbeta is not None:
            if self.T > 0.0:
                raise Exception("nalpha and nbeta ill-defined at FT")
            self.mu = None
        else:
            if T == 0.0:
                ddd = sys.g_energies_tot()
                N = len([d < mu for d in ddd])
                self.nalpha = N//2
                self.nbeta = N//2

    def run(self):
        """Run the calculation."""
        En = self.sys.const_energy()
        T = self.T
        if self.T > 0.0:
            if self.iprint > 0:
                print("Running FCI at finite T")
                print('   T = {}'.format(T))
            beta = 1.0 / self.T
            en = self.sys.g_energies_tot()
            g0 = ft_utils.GP0(beta, en, self.mu)
            E0 = ft_mp.mp0(g0) + En
            E1 = self.sys.get_mp1()
            Efci = self._ft_fci()
            return (Efci+En,Efci - E0 - E1)
        else:
            if self.iprint > 0:
                print("Running FCI at T=0")
            Efci = self._fci_fixedN(self.nalpha,self.nbeta)[0]
            E0 = self.sys.g_energies()[0].sum() + En
            E1 = self.sys.get_mp1()
            return (Efci+En,Efci - E0 - E1)

    def _fci_fixedN(self, nalpha, nbeta):
        """Run FCI at fixed N."""
        lam = self.lam
        nelec = (nalpha,nbeta)
        f = self.sys.r_fock_tot()
        f = f - numpy.diag(self.sys.r_energies_tot())
        f = numpy.diag(self.sys.r_energies_tot())
        h = self.sys.r_hcore()
        h1e = f + lam*(h - f)
        n = h1e.shape[0]
        eri = lam*self.sys.g_int_tot()[:n,:n,:n,:n]
        eri = numpy.transpose(eri,(0,2,1,3))
        h2e = fci_slow.absorb_h1e(h1e, eri, n, nelec, .5)
        na = cistring.num_strings(n, nalpha)
        nb = cistring.num_strings(n, nbeta)
        N = na*nb
        assert(N < 4000)
        H = numpy.zeros((N,N))
        I = numpy.identity(N)
        for i in range(N):
            hc = fci_slow.contract_2e(h2e,I[:,i],n,nelec)
            hc.reshape(-1)
            H[:,i] = hc

        e,v = numpy.linalg.eigh(H)
        return e

    def _ft_fci(self):
        """Run FCI at finite T."""
        T = self.T
        mu = self.mu
        beta = 1.0 / T
        Z = 0.0
        n = self.sys.g_energies_tot().shape[0]//2
        for nalpha in range(0,n + 1):
            for nbeta in range(0,n + 1):
                e = self._fci_fixedN(nalpha,nbeta)
                nel = nalpha + nbeta
                for ej in e:
                    ex = beta*(nel*mu - ej)
                    Z += numpy.exp(ex)

        return -T*numpy.log(Z)
