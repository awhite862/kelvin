import unittest
import numpy
from pyscf import gto, scf
from kelvin.td_ccsd import TDCCSD
from kelvin.ccsd import ccsd
from kelvin.scf_system import scf_system
from kelvin.ueg_system import ueg_system

class TDCCSDESNTest(unittest.TestCase):
    def test_Be_gen(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-13
        Escf = m.scf()
        T = 2.0
        mu = 0.0
        ng = 30
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,iprint=0,T=T,mu=mu,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10,singles=True)
        Ecctot,Ecc = ccsdT.run()
        ccsdT.compute_ESN()
        prop = {"tprop" : "rk4", "lprop" : "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=40)
        Eout,Eccout = tdccsdT.run()
        tdccsdT.compute_ESN()

        dE = abs((ccsdT.E - tdccsdT.E)/ccsdT.E)
        dS = abs((ccsdT.S - tdccsdT.S)/ccsdT.S)
        dN = abs((ccsdT.N - tdccsdT.N)/ccsdT.N)
        eE = "Expected: {}  Actual: {}".format(ccsdT.E, tdccsdT.E)
        eS = "Expected: {}  Actual: {}".format(ccsdT.S, tdccsdT.S)
        eN = "Expected: {}  Actual: {}".format(ccsdT.N, tdccsdT.N)
        self.assertTrue(dE < 1e-6,eE)
        self.assertTrue(dS < 1e-6,eS)
        self.assertTrue(dN < 1e-6,eN)

    def test_UEG_gen(self):
        T = 0.1
        mu = 0.1
        L = 2*numpy.pi/numpy.sqrt(1.0)
        norb = 7
        cut = 1.2
        damp = 0.2
        mi = 50
        ng = 30
        ueg = ueg_system(T,L,cut,mu=mu,norb=norb,orbtype='g')
        ccsdT = ccsd(ueg,T=T,mu=mu,iprint=0,max_iter=mi,damp=damp,ngrid=ng)
        Ecctot,Ecc = ccsdT.run()
        ccsdT.compute_ESN()
        Eref = ccsdT.E
        Sref = ccsdT.S
        Nref = ccsdT.N
        prop = {"tprop" : "rk4", "lprop" : "rk4"}
        tdccsdT = TDCCSD(ueg, prop, T=T, mu=mu, ngrid=40)
        Eout,Eccout = tdccsdT.run()
        tdccsdT.compute_ESN()
        E = tdccsdT.E
        S = tdccsdT.S
        N = tdccsdT.N

        dE = abs(Eref - E)/Eref
        dS = abs(Sref - S)/Sref
        dN = abs(Nref - N)/Nref
        eE = "Expected: {}  Actual: {}".format(Eref,E)
        eS = "Expected: {}  Actual: {}".format(Sref,S)
        eN = "Expected: {}  Actual: {}".format(Nref,N)
        self.assertTrue(dE < 5e-5,eE)
        self.assertTrue(dS < 5e-5,eS)
        self.assertTrue(dN < 5e-5,eN)

    def test_Be_gen_active(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-13
        Escf = m.scf()
        T = 0.02
        mu = 0.0
        ng = 200
        athresh = 1e-20
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,iprint=0,T=T,mu=mu,max_iter=100,damp=0.1,ngrid=ng,econv=1e-10,athresh=athresh,singles=True)
        Ecctot,Ecc = ccsdT.run()
        ccsdT.compute_ESN()
        Eref = ccsdT.E
        Sref = ccsdT.S
        Nref = ccsdT.N
        prop = {"tprop" : "rk4", "lprop" : "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=560, athresh=1e-20, saveT=True)
        Eout,Eccout = tdccsdT.run()
        tdccsdT.compute_ESN()
        E = tdccsdT.E
        S = tdccsdT.S
        N = tdccsdT.N
        dE = abs((Eref - E)/Eref)
        dS = abs((Sref - S)/Sref)
        dN = abs((Nref - N)/Nref)
        eE = "Expected: {}  Actual: {}".format(Eref,E)
        eS = "Expected: {}  Actual: {}".format(Sref,S)
        eN = "Expected: {}  Actual: {}".format(Nref,N)
        self.assertTrue(dE < 1e-5,eE)
        self.assertTrue(dS < 1e-3,eS)
        self.assertTrue(dN < 1e-5,eN)

    def test_UEG(self):
        T = 0.1
        mu = 0.1
        L = 2*numpy.pi/numpy.sqrt(1.0)
        norb = 7
        cut = 1.2
        damp = 0.2
        mi = 50
        ng = 30
        ueg = ueg_system(T,L,cut,mu=mu,norb=norb,orbtype='u')
        ccsdT = ccsd(ueg,T=T,mu=mu,iprint=0,max_iter=mi,damp=damp,ngrid=ng)
        Ecctot,Ecc = ccsdT.run()
        ccsdT.compute_ESN()
        Eref = ccsdT.E
        Sref = ccsdT.S
        Nref = ccsdT.N
        prop = {"tprop" : "rk4", "lprop" : "rk4"}
        tdccsdT = TDCCSD(ueg, prop, T=T, mu=mu, ngrid=40)
        Eout,Eccout = tdccsdT.run()
        tdccsdT.compute_ESN()
        E = tdccsdT.E
        S = tdccsdT.S
        N = tdccsdT.N

        dE = abs(Eref - E)/Eref
        dS = abs(Sref - S)/Sref
        dN = abs(Nref - N)/Nref
        eE = "Expected: {}  Actual: {}".format(Eref,E)
        eS = "Expected: {}  Actual: {}".format(Sref,S)
        eN = "Expected: {}  Actual: {}".format(Nref,N)
        self.assertTrue(dE < 5e-5,eE)
        self.assertTrue(dS < 5e-5,eS)
        self.assertTrue(dN < 5e-5,eN)

    def test_UEG_h5py(self):
        T = 0.1
        mu = 0.1
        L = 2*numpy.pi/numpy.sqrt(1.0)
        norb = 7
        cut = 1.2
        damp = 0.2
        mi = 50
        ng = 30
        ueg = ueg_system(T,L,cut,mu=mu,norb=norb,orbtype='u')
        ccsdT = ccsd(ueg,T=T,mu=mu,iprint=0,max_iter=mi,damp=damp,ngrid=ng)
        Ecctot,Ecc = ccsdT.run()
        ccsdT.compute_ESN()
        Eref = ccsdT.E
        Sref = ccsdT.S
        Nref = ccsdT.N
        prop = {"tprop" : "rk4", "lprop" : "rk4"}
        tdccsdT = TDCCSD(ueg, prop, T=T, mu=mu, ngrid=40, saveT=True, tmem="hdf5", saveL=True)
        Eout,Eccout = tdccsdT.run()
        tdccsdT.compute_ESN()
        E = tdccsdT.E
        S = tdccsdT.S
        N = tdccsdT.N

        dE = abs(Eref - E)/Eref
        dS = abs(Sref - S)/Sref
        dN = abs(Nref - N)/Nref
        eE = "Expected: {}  Actual: {}".format(Eref,E)
        eS = "Expected: {}  Actual: {}".format(Sref,S)
        eN = "Expected: {}  Actual: {}".format(Nref,N)
        self.assertTrue(dE < 5e-5,eE)
        self.assertTrue(dS < 5e-5,eS)
        self.assertTrue(dN < 5e-5,eN)

    def test_Be_active(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-13
        Escf = m.scf()
        T = 0.02
        mu = 0.0
        ng = 200
        athresh = 1e-20
        sys = scf_system(m,T,mu,orbtype='u')
        ccsdT = ccsd(sys,iprint=0,T=T,mu=mu,max_iter=100,damp=0.1,ngrid=ng,econv=1e-10,athresh=athresh,singles=True)
        Ecctot,Ecc = ccsdT.run()
        ccsdT.compute_ESN()
        Eref = ccsdT.E
        Sref = ccsdT.S
        Nref = ccsdT.N
        prop = {"tprop" : "rk4", "lprop" : "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=560, athresh=1e-20, saveT=True)
        Eout,Eccout = tdccsdT.run()
        tdccsdT.compute_ESN()
        E = tdccsdT.E
        S = tdccsdT.S
        N = tdccsdT.N
        dE = abs((Eref - E)/Eref)
        dS = abs((Sref - S)/Sref)
        dN = abs((Nref - N)/Nref)
        eE = "Expected: {}  Actual: {}".format(Eref,E)
        eS = "Expected: {}  Actual: {}".format(Sref,S)
        eN = "Expected: {}  Actual: {}".format(Nref,N)
        self.assertTrue(dE < 1e-5,eE)
        self.assertTrue(dS < 1e-3,eS)
        self.assertTrue(dN < 1e-5,eN)

    def test_UEG_r_vs_u(self):
        T = 0.1
        mu = 0.1
        L = 2*numpy.pi/numpy.sqrt(1.0)
        norb = 7
        cut = 1.2
        damp = 0.2
        mi = 50
        ng = 30
        ueg = ueg_system(T,L,cut,mu=mu,norb=norb,orbtype='u')
        prop = {"tprop" : "rk4", "lprop" : "rk4"}
        tdccsdT = TDCCSD(ueg, prop, T=T, mu=mu, ngrid=40)
        Ecctot,Ecc = tdccsdT.run()
        tdccsdT.compute_ESN()
        Eref = tdccsdT.E
        Sref = tdccsdT.S
        Nref = tdccsdT.N
        ueg = ueg_system(T,L,cut,mu=mu,norb=norb,orbtype='r')
        prop = {"tprop" : "rk4", "lprop" : "rk4"}
        tdccsdT = TDCCSD(ueg, prop, T=T, mu=mu, ngrid=40)
        Eout,Eccout = tdccsdT.run()
        tdccsdT.compute_ESN()
        E = tdccsdT.E
        S = tdccsdT.S
        N = tdccsdT.N

        dE = abs(Eref - E)/Eref
        dS = abs(Sref - S)/Sref
        dN = abs(Nref - N)/Nref
        eE = "Expected: {}  Actual: {}".format(Eref,E)
        eS = "Expected: {}  Actual: {}".format(Sref,S)
        eN = "Expected: {}  Actual: {}".format(Nref,N)
        self.assertTrue(dE < 5e-14,eE)
        self.assertTrue(dS < 5e-14,eS)
        self.assertTrue(dN < 5e-14,eN)

    def test_Be_active_r_vs_u(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-13
        Escf = m.scf()
        T = 0.02
        mu = 0.0
        ng = 200
        athresh = 1e-20
        sys = scf_system(m,T,mu,orbtype='u')
        prop = {"tprop" : "rk4", "lprop" : "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=280, athresh=1e-20, saveT=True)
        Ecctot,Ecc = tdccsdT.run()
        tdccsdT.compute_ESN()
        Eref = tdccsdT.E
        Sref = tdccsdT.S
        Nref = tdccsdT.N
        sys = scf_system(m,T,mu,orbtype='r')
        prop = {"tprop" : "rk4", "lprop" : "rk4"}
        tdccsdT = TDCCSD(sys, prop, T=T, mu=mu, ngrid=280, athresh=1e-20, saveT=True)
        Eout,Eccout = tdccsdT.run()
        tdccsdT.compute_ESN()
        E = tdccsdT.E
        S = tdccsdT.S
        N = tdccsdT.N
        dE = abs((Eref - E)/Eref)
        dS = abs((Sref - S)/Sref)
        dN = abs((Nref - N)/Nref)
        eE = "Expected: {}  Actual: {}".format(Eref,E)
        eS = "Expected: {}  Actual: {}".format(Sref,S)
        eN = "Expected: {}  Actual: {}".format(Nref,N)
        self.assertTrue(dE < 1e-5,eE)
        self.assertTrue(dS < 1e-3,eS)
        self.assertTrue(dN < 1e-5,eN)

if __name__ == '__main__':
    unittest.main()
