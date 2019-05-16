import unittest
import numpy

from lattice.hubbard import Hubbard1D
from cqcpy import ft_utils
from cqcpy import utils

from kelvin.hubbard_system import HubbardSystem
from kelvin.ccsd import ccsd

class FakeHubbardSystem(object):
    def __init__(self,sys,M=None):
        self.M = M
        self.sys = sys 

    def has_g(self):
        return self.sys.has_g()

    def has_u(self):
        return self.sys.has_u()

    def has_r(self):
        return self.sys.has_r()

    def verify(self,T,mu):
        return self.sys.verify(T,mu)

    def const_energy(self):
        return self.sys.const_energy()

    def get_mp1(self):
        E1 = self.sys.get_mp1()
        beta = 1.0/self.sys.T
        mu = self.sys.mu
        en = self.sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)
        extra = 0.5*numpy.einsum('ijij,i,j->',self.M,fo,fo)
        return E1 + extra

    def g_energies_tot(self):
        return self.sys.g_energies_tot()

    def g_fock_tot(self):
        return self.sys.g_fock_tot()

    def g_aint_tot(self):
        U = self.sys.g_aint_tot()
        return U + self.M

class FTCC2RDMTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_hubbard(self):
        U = 1.0
        T = 1.0
        beta = 1./T
        L = 2
        mu = 0.5
        Mg = numpy.zeros((2*L,2*L,2*L,2*L))
        for i in range(L):
            Mg[i,L+i,i,L+i] = -2.0
            Mg[L+i,i,L+i,i] = -2.0
        Mg = Mg - Mg.transpose((0,1,3,2))
        hub = Hubbard1D(2,1.0,U)
        Pa = numpy.zeros((2,2))
        Pb = numpy.zeros((2,2))
        Pa[0,0] = 1.0
        Pb[1,1] = 1.0
        sys = HubbardSystem(T,hub,Pa,Pb,mu=mu,orbtype='g')
        cmat = utils.block_diag(sys.ua,sys.ub)
        Mg = sys._transform1(Mg,cmat)
        cc = ccsd(sys,T=T,mu=mu,iprint=0,max_iter=80,econv=1e-11)
        E,Ecc = cc.run()
        en = sys.g_energies_tot()
        fo = ft_utils.ff(beta, en, mu)
        fv = ft_utils.ffv(beta, en, mu)
        sfo = numpy.sqrt(fo)
        sfv = numpy.sqrt(fv)
        cc._ft_ccsd_lambda()
        cc._g_ft_2rdm()
        P2 = cc.P2
        prop = 0.5*numpy.einsum('ijij,i,j->',Mg,fo,fo)
        A1 = 0.25*numpy.einsum('cdab,abcd,a,b,c,d->',P2[0],Mg,sfv,sfv,sfv,sfv)
        A2 = 0.5*numpy.einsum('ciab,abci,c,i,a,b->',P2[1],Mg,sfv,sfo,sfv,sfv)
        A3 = 0.5*numpy.einsum('bcai,aibc,b,c,a,i->',P2[2],Mg,sfv,sfv,sfv,sfo)
        A4 = 0.25*numpy.einsum('ijab,abij,i,j,a,b->',P2[3],Mg,sfo,sfo,sfv,sfv)
        A5 = 1.0*numpy.einsum('bjai,aibj,b,j,a,i->',P2[4],Mg,sfv,sfo,sfv,sfo)
        A6 = 0.25*numpy.einsum('abij,ijab,a,b,i,j->',P2[5],Mg,sfv,sfv,sfo,sfo)
        A7 = 0.5*numpy.einsum('jkai,aijk,j,k,a,i->',P2[6],Mg,sfo,sfo,sfv,sfo)
        A8 = 0.5*numpy.einsum('kaij,ijka,k,a,i,j->',P2[7],Mg,sfo,sfv,sfo,sfo)
        A9 = 0.25*numpy.einsum('klij,ijkl,k,l,i,j->',P2[8],Mg,sfo,sfo,sfo,sfo)
        E2 = A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9
        out = prop + E2

        d = 5e-4
        sysf = FakeHubbardSystem(sys,M=d*Mg)
        ccf = ccsd(sysf,T=T,mu=mu,iprint=0,max_iter=80,econv=1e-11)
        Ef,Eccf = ccf.run()

        sysb = FakeHubbardSystem(sys,M=-d*Mg)
        ccb = ccsd(sysb,T=T,mu=mu,iprint=0,max_iter=80,econv=1e-11)
        Eb,Eccb = ccb.run()
        ref = (Ef - Eb)/(2*d)
        self.assertTrue(abs(ref - out) < 1e-6)

if __name__ == '__main__':
    unittest.main()
