import unittest
import numpy
from pyscf import gto, scf, cc
from kelvin.mp3 import mp3
from kelvin.ccsd import ccsd
from kelvin.scf_system import scf_system
from kelvin.ueg_system import ueg_system
from kelvin.pueg_system import pueg_system

def fd_ESN(m, T, mu, ng, Ecctot, quad = 'lin'):
    delta = 5e-4
    muf = mu + delta
    mub = mu - delta
    sys = scf_system(m,T,muf,orbtype='g')
    ccsdT = ccsd(sys,iprint=0,T=T,mu=muf,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10, quad=quad)
    Ef,Ecf = ccsdT.run()
    sys = scf_system(m,T,mub,orbtype='g')
    ccsdT = ccsd(sys,iprint=0,T=T,mu=mub,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10, quad=quad)
    Eb,Ecb = ccsdT.run()
    
    Nx = -(Ef - Eb)/(2*delta)
    
    Tf = T + delta
    Tb = T - delta
    sys = scf_system(m,Tf,mu,orbtype='g')
    ccsdT = ccsd(sys,iprint=0,T=Tf,mu=mu,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10, quad=quad)
    Ef,Ecf = ccsdT.run()
    sys = scf_system(m,Tb,mu,orbtype='g')
    ccsdT = ccsd(sys,iprint=0,T=Tb,mu=mu,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10, quad=quad)
    Eb,Ecb = ccsdT.run()
    
    Sx = -(Ef - Eb)/(2*delta)
    Ex = Ecctot + T*Sx + mu*Nx
    
    return (Ex,Nx,Sx)

class FTDerivTest(unittest.TestCase):
    def setUp(self):
        self.Bethresh = 1e-5
        self.uegthresh = 1e-5

    def test_Be_sto3g_gen(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-13
        Escf = m.scf()
        T = 2.0
        mu = 0.0
        ng = 10
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,iprint=0,T=T,mu=mu,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10,singles=True)
        Ecctot,Ecc = ccsdT.run()
        ccsdT.compute_ESN()
        Ex,Nx,Sx = fd_ESN(m, T, mu, ng, Ecctot)
        dE = abs(ccsdT.E - Ex)/Ex
        dS = abs(ccsdT.S - Sx)/Sx
        dN = abs(ccsdT.N - Nx)/Nx
        eE = "Expected: {}  Actual: {}".format(Ex,ccsdT.E)
        eS = "Expected: {}  Actual: {}".format(Sx,ccsdT.S)
        eN = "Expected: {}  Actual: {}".format(Nx,ccsdT.N)
        self.assertTrue(dE < self.Bethresh,eE)
        self.assertTrue(dS < self.Bethresh,eS)
        self.assertTrue(dN < self.Bethresh,eN)

    def test_Be_sto3g(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-13
        Escf = m.scf()
        T = 2.0
        mu = 0.0
        ng = 10
        sys = scf_system(m,T,mu,orbtype='u')
        ccsdT = ccsd(sys,iprint=0,T=T,mu=mu,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10,singles=True)
        Ecctot,Ecc = ccsdT.run()
        ccsdT.compute_ESN()
        Ex,Nx,Sx = fd_ESN(m, T, mu, ng, Ecctot)
        dE = abs(ccsdT.E - Ex)/Ex
        dS = abs(ccsdT.S - Sx)/Sx
        dN = abs(ccsdT.N - Nx)/Nx
        eE = "Expected: {}  Actual: {}".format(Ex,ccsdT.E)
        eS = "Expected: {}  Actual: {}".format(Sx,ccsdT.S)
        eN = "Expected: {}  Actual: {}".format(Nx,ccsdT.N)
        self.assertTrue(dE < self.Bethresh,eE)
        self.assertTrue(dS < self.Bethresh,eS)
        self.assertTrue(dN < self.Bethresh,eN)

    def test_Be_sto3g_ln(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-13
        Escf = m.scf()
        T = 2.0
        mu = 0.0
        ng = 10
        sys = scf_system(m,T,mu,orbtype='u')
        ccsdT = ccsd(sys,iprint=0,T=T,mu=mu,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10,quad='ln')
        Ecctot,Ecc = ccsdT.run()
        ccsdT.compute_ESN()
        Ex,Nx,Sx = fd_ESN(m, T, mu, ng, Ecctot, quad='ln')
        dE = abs(ccsdT.E - Ex)/Ex
        dS = abs(ccsdT.S - Sx)/Sx
        dN = abs(ccsdT.N - Nx)/Nx
        eE = "Expected: {}  Actual: {}".format(Ex,ccsdT.E)
        eS = "Expected: {}  Actual: {}".format(Sx,ccsdT.S)
        eN = "Expected: {}  Actual: {}".format(Nx,ccsdT.N)
        self.assertTrue(dE < self.Bethresh,eE)
        self.assertTrue(dS < self.Bethresh,eS)
        self.assertTrue(dN < self.Bethresh,eN)

    def test_Be_sto3g_sin(self):
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        m = scf.RHF(mol)
        m.conv_tol = 1e-13
        Escf = m.scf()
        T = 2.0
        mu = 0.0
        ng = 10
        sys = scf_system(m,T,mu,orbtype='u')
        ccsdT = ccsd(sys,iprint=0,T=T,mu=mu,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10,quad='sin')
        Ecctot,Ecc = ccsdT.run()
        ccsdT.compute_ESN()
        Ex,Nx,Sx = fd_ESN(m, T, mu, ng, Ecctot, quad='sin')
        dE = abs(ccsdT.E - Ex)/Ex
        dS = abs(ccsdT.S - Sx)/Sx
        dN = abs(ccsdT.N - Nx)/Nx
        eE = "Expected: {}  Actual: {}".format(Ex,ccsdT.E)
        eS = "Expected: {}  Actual: {}".format(Sx,ccsdT.S)
        eN = "Expected: {}  Actual: {}".format(Nx,ccsdT.N)
        self.assertTrue(dE < self.Bethresh,eE)
        self.assertTrue(dS < self.Bethresh,eS)
        self.assertTrue(dN < self.Bethresh,eN)

    def test_UEG(self):
        T = 0.1
        mu = 0.1
        L = 2*numpy.pi/numpy.sqrt(1.0)
        norb = 7
        cut = 1.2
        damp = 0.2
        mi = 50
        ng = 10
        ueg = ueg_system(T,L,cut,mu=mu,norb=norb)
        ccsdT = ccsd(ueg,T=T,mu=mu,iprint=0,max_iter=mi,damp=damp,ngrid=ng)
        Ecctot,Ecc = ccsdT.run()
        ccsdT.compute_ESN()
        E = ccsdT.E
        S = ccsdT.S
        N = ccsdT.N
        delta = 1e-4
        muf = mu + delta
        mub = mu - delta
        ueg = ueg_system(T,L,cut,mu=muf,norb=norb)
        ccsdT = ccsd(ueg,iprint=0,T=T,mu=muf,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10)
        Ef,Ecf = ccsdT.run()
        ueg = ueg_system(T,L,cut,mu=mub,norb=norb)
        ccsdT = ccsd(ueg,iprint=0,T=T,mu=mub,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10)
        Eb,Ecb = ccsdT.run()

        Nx = -(Ef - Eb)/(2*delta)

        Tf = T + delta
        Tb = T - delta
        ueg = ueg_system(Tf,L,cut,mu=mu,norb=norb)
        ccsdT = ccsd(ueg,iprint=0,T=Tf,mu=mu,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10)
        Ef,Ecf = ccsdT.run()
        ueg = ueg_system(Tb,L,cut,mu=mu,norb=norb)
        ccsdT = ccsd(ueg,iprint=0,T=Tb,mu=mu,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10)
        Eb,Ecb = ccsdT.run()

        Sx = -(Ef - Eb)/(2*delta)
        Ex = Ecctot + T*Sx + mu*Nx

        dE = abs(E - Ex)
        dS = abs(S - Sx)
        dN = abs(N - Nx)
        eE = "Expected: {}  Actual: {}".format(Ex,E)
        eS = "Expected: {}  Actual: {}".format(Sx,S)
        eN = "Expected: {}  Actual: {}".format(Nx,N)
        self.assertTrue(dE < self.uegthresh,eE)
        self.assertTrue(dS < self.uegthresh,eS)
        self.assertTrue(dN < self.uegthresh,eN)

    def test_UEG_gen(self):
        T = 0.1
        mu = 0.1
        L = 2*numpy.pi/numpy.sqrt(1.0)
        norb = 7
        cut = 1.2
        damp = 0.2
        mi = 50
        ng = 10
        ueg = ueg_system(T,L,cut,mu=mu,norb=norb,orbtype='g')
        ccsdT = ccsd(ueg,T=T,mu=mu,iprint=0,max_iter=mi,damp=damp,ngrid=ng)
        Ecctot,Ecc = ccsdT.run()
        ccsdT.compute_ESN()
        E = ccsdT.E
        S = ccsdT.S
        N = ccsdT.N
        delta = 1e-4
        muf = mu + delta
        mub = mu - delta
        ueg = ueg_system(T,L,cut,mu=muf,norb=norb)
        ccsdT = ccsd(ueg,iprint=0,T=T,mu=muf,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10)
        Ef,Ecf = ccsdT.run()
        ueg = ueg_system(T,L,cut,mu=mub,norb=norb)
        ccsdT = ccsd(ueg,iprint=0,T=T,mu=mub,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10)
        Eb,Ecb = ccsdT.run()

        Nx = -(Ef - Eb)/(2*delta)

        Tf = T + delta
        Tb = T - delta
        ueg = ueg_system(Tf,L,cut,mu=mu,norb=norb)
        ccsdT = ccsd(ueg,iprint=0,T=Tf,mu=mu,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10)
        Ef,Ecf = ccsdT.run()
        ueg = ueg_system(Tb,L,cut,mu=mu,norb=norb)
        ccsdT = ccsd(ueg,iprint=0,T=Tb,mu=mu,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10)
        Eb,Ecb = ccsdT.run()

        Sx = -(Ef - Eb)/(2*delta)
        Ex = Ecctot + T*Sx + mu*Nx

        dE = abs(E - Ex)/Ex
        dS = abs(S - Sx)/Sx
        dN = abs(N - Nx)/Nx
        eE = "Expected: {}  Actual: {}".format(Ex,E)
        eS = "Expected: {}  Actual: {}".format(Sx,S)
        eN = "Expected: {}  Actual: {}".format(Nx,N)
        self.assertTrue(dE < self.uegthresh,eE)
        self.assertTrue(dS < self.uegthresh,eS)
        self.assertTrue(dN < self.uegthresh,eN)

    def test_PUEG(self):
        T = 0.1
        mu = 0.1
        L = 2*numpy.pi/numpy.sqrt(1.0)
        norb = 7
        cut = 1.2
        damp = 0.2
        mi = 50
        ng = 10
        ueg = pueg_system(T,L,cut,mu=mu,norb=norb)
        ccsdT = ccsd(ueg,T=T,mu=mu,iprint=0,max_iter=mi,damp=damp,ngrid=ng)
        Ecctot,Ecc = ccsdT.run()
        ccsdT.compute_ESN()
        E = ccsdT.E
        S = ccsdT.S
        N = ccsdT.N
        delta = 1e-4
        muf = mu + delta
        mub = mu - delta
        ueg = pueg_system(T,L,cut,mu=muf,norb=norb)
        ccsdT = ccsd(ueg,iprint=0,T=T,mu=muf,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10)
        Ef,Ecf = ccsdT.run()
        ueg = pueg_system(T,L,cut,mu=mub,norb=norb)
        ccsdT = ccsd(ueg,iprint=0,T=T,mu=mub,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10)
        Eb,Ecb = ccsdT.run()

        Nx = -(Ef - Eb)/(2*delta)

        Tf = T + delta
        Tb = T - delta
        ueg = pueg_system(Tf,L,cut,mu=mu,norb=norb)
        ccsdT = ccsd(ueg,iprint=0,T=Tf,mu=mu,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10)
        Ef,Ecf = ccsdT.run()
        ueg = pueg_system(Tb,L,cut,mu=mu,norb=norb)
        ccsdT = ccsd(ueg,iprint=0,T=Tb,mu=mu,max_iter=35,damp=0.0,ngrid=ng,econv=1e-10)
        Eb,Ecb = ccsdT.run()

        Sx = -(Ef - Eb)/(2*delta)
        Ex = Ecctot + T*Sx + mu*Nx

        dE = abs(E - Ex)/Ex
        dS = abs(S - Sx)/Sx
        dN = abs(N - Nx)/Nx
        eE = "Expected: {}  Actual: {}".format(Ex,E)
        eS = "Expected: {}  Actual: {}".format(Sx,S)
        eN = "Expected: {}  Actual: {}".format(Nx,N)
        self.assertTrue(dE < self.uegthresh,eE)
        self.assertTrue(dS < self.uegthresh,eS)
        self.assertTrue(dN < self.uegthresh,eN)


if __name__ == '__main__':
    unittest.main()
