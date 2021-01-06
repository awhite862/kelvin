import unittest
import numpy
from kelvin.fci import fci
from kelvin.ccsd import ccsd
try:
    from kelvin.hubbard_site_system import hubbard_site_system
    from kelvin.hubbard_system import HubbardSystem
    from lattice.hubbard import Hubbard1D
    from lattice.fci import FCISimple
    has_lattice = True
except:
    has_lattice = False

def compute_FCISimple(hub, nelec):
    myfci = FCISimple(hub,nelec,m_s=0)
    e,v = myfci.run()
    return e[0]

def compute_fci_kelvin(hub, nelec):
    nb = nelec//2
    na = nelec - nb
    Pa = None
    Pb = None
    sys = hubbard_site_system(0.0,hub,Pa,Pb,na=na,nb=nb)
    myfci = fci(sys,nalpha=na,nbeta=nb)
    return myfci.run()[0]

def compute_fci_kelvinT(hub, T, mu):
    Pa = None
    Pb = None
    sys = hubbard_site_system(T,hub,Pa,Pb,mu=mu)
    myfci = fci(sys,T=T,mu=mu)
    return myfci.run()[0]

@unittest.skipUnless(has_lattice, "Lattice module cannot be found")
class HubbardTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-10

    def _compare_closed(self,nelec,L,U,t=1.0):
        hub = Hubbard1D(L,t,U,boundary='c')
        Es = compute_FCISimple(hub,nelec)
        Ec = compute_fci_kelvin(hub,nelec)
        return (Es,Ec)

    def _compare_open(self,nelec,L,U,t=1.0):
        hub = Hubbard1D(L,t,U)
        Es = compute_FCISimple(hub,nelec)
        Ec = compute_fci_kelvin(hub,nelec)
        return (Es,Ec)

    def test_2_1(self):
        nelec = 2
        L = 2
        U = 1.0
        Es,Ec = self._compare_closed(nelec,L,U)
        diff = abs(Es - Ec)
        msg = "Expected: {}  Actual: {}".format(Es,Ec)
        self.assertTrue(diff < self.thresh,msg)

    def test_4_1_closed(self):
        nelec = 4
        L = 4
        U = 1.0
        Es,Ec = self._compare_closed(nelec,L,U)
        diff = abs(Es - Ec)
        msg = "Expected: {}  Actual: {}".format(Es,Ec)
        self.assertTrue(diff < self.thresh,msg)

    def test_4_1_open(self):
        nelec = 4
        L = 4
        U = 1.0
        Es,Ec = self._compare_open(nelec,L,U)
        diff = abs(Es - Ec)
        msg = "Expected: {}  Actual: {}".format(Es,Ec)
        self.assertTrue(diff < self.thresh,msg)

    def test_6_2_open(self):
        nelec = 6
        L = 6
        U = 2.0
        Es,Ec = self._compare_open(nelec,L,U)
        diff = abs(Es - Ec)
        msg = "Expected: {}  Actual: {}".format(Es,Ec)
        self.assertTrue(diff < self.thresh,msg)

    def test_ccsd_site(self):
        nelec = 2
        L = 2
        U = 1.0
        hub = Hubbard1D(L,1.0,U,boundary='c')
        Eref = compute_FCISimple(hub,nelec)

        Oa = numpy.zeros((2))
        Ob = numpy.zeros((2))
        Oa[0] = 1.0
        Ob[1] = 1.0
        Pa = numpy.einsum('i,j->ij',Oa,Oa)
        Pb = numpy.einsum('i,j->ij',Ob,Ob)
        sys = hubbard_site_system(0.0,hub,Pa,Pb,na=1,nb=1)
        cc = ccsd(sys,iprint=0,damp=0.6,max_iter=80,econv=1e-11)
        T2aa = numpy.zeros((1,1,1,1))
        T2ab = numpy.zeros((1,1,1,1))
        T2bb = numpy.zeros((1,1,1,1))
        T1a = numpy.zeros((1,1))
        T1b = numpy.zeros((1,1))
        T1a[0,0] = 1.0
        T1b[0,0] = 1.0
        T1 = (T1a,T1b)
        T2 = (T2aa,T2ab,T2bb)
        Eout,Ecc = cc.run(T1=T1,T2=T2)
        diff = abs(Eout - Eref)
        msg = "Expected: {} Actual: {}".format(Eref,Eout)
        self.assertTrue(diff < self.thresh,msg)

    def test_ccsd(self):
        nelec = 2
        L = 2
        U = 1.0
        hub = Hubbard1D(L,1.0,U,boundary='c')
        Eref = compute_FCISimple(hub,nelec)

        Oa = numpy.zeros((2))
        Ob = numpy.zeros((2))
        Oa[0] = 1.0
        Ob[1] = 1.0
        Pa = numpy.einsum('i,j->ij',Oa,Oa)
        Pb = numpy.einsum('i,j->ij',Ob,Ob)
        sys = HubbardSystem(0.0,hub,Pa,Pb,na=1,nb=1)
        cc = ccsd(sys,iprint=0,max_iter=80,econv=1e-11)
        Eout,Ecc = cc.run()
        diff = abs(Eout - Eref)
        msg = "Expected: {} Actual: {}".format(Eref,Eout)
        self.assertTrue(diff < self.thresh,msg)

    def test_ft_ccsd(self):
        nelec = 2
        L = 2
        U = 1.0
        T = 1.0
        mu = 0.0
        hub = Hubbard1D(L,1.0,U,boundary='c')
        Eref = compute_fci_kelvinT(hub,T,mu)

        Oa = numpy.zeros((2))
        Ob = numpy.zeros((2))
        Oa[0] = 1.0
        Ob[1] = 1.0
        Pa = numpy.einsum('i,j->ij',Oa,Oa)
        Pb = numpy.einsum('i,j->ij',Ob,Ob)
        sys = HubbardSystem(T,hub,Pa,Pb,mu=mu)
        cc = ccsd(sys,iprint=0,max_iter=80,econv=1e-11,T=T,mu=mu)
        Eout,Ecc = cc.run()
        diff = abs(Eout - Eref)
        msg = "Expected: {} Actual: {}".format(Eref,Eout)
        self.assertTrue(diff < 1e-4,msg)

    def test_ccsd_u_g(self):
        nelec = 2
        L = 2
        U = 1.0
        hub = Hubbard1D(L,1.0,U,boundary='c')
        Oa = numpy.zeros((2))
        Ob = numpy.zeros((2))
        Oa[0] = 1.0
        Ob[1] = 1.0
        Pa = numpy.einsum('i,j->ij',Oa,Oa)
        Pb = numpy.einsum('i,j->ij',Ob,Ob)
        sys = HubbardSystem(0.0,hub,Pa,Pb,na=1,nb=1,orbtype='g')
        cc = ccsd(sys,iprint=0,max_iter=80,econv=1e-11)
        Eoutg,Eccg = cc.run()
        sys = HubbardSystem(0.0,hub,Pa,Pb,na=1,nb=1,orbtype='u')
        cc = ccsd(sys,iprint=0,max_iter=80,econv=1e-11)
        Eoutu,Eccu = cc.run()
        diff = abs(Eoutg - Eoutu)
        msg = "General: {} Unrestricted: {}".format(Eoutg,Eoutu)
        self.assertTrue(diff < self.thresh,msg)

    def test_ft_ccsd_u_g(self):
        nelec = 2
        L = 2
        U = 1.0
        T = 1.0
        mu = 0.0
        hub = Hubbard1D(L,1.0,U,boundary='c')
        Eref = compute_fci_kelvinT(hub,T,mu)

        Oa = numpy.zeros((2))
        Ob = numpy.zeros((2))
        Oa[0] = 1.0
        Ob[1] = 1.0
        Pa = numpy.einsum('i,j->ij',Oa,Oa)
        Pb = numpy.einsum('i,j->ij',Ob,Ob)
        sys = HubbardSystem(T,hub,Pa,Pb,mu=mu,orbtype='g')
        cc = ccsd(sys,iprint=0,max_iter=80,econv=1e-11,T=T,mu=mu)
        Eoutg,Eccg = cc.run()
        sys = HubbardSystem(T,hub,Pa,Pb,mu=mu,orbtype='u')
        cc = ccsd(sys,iprint=0,max_iter=80,econv=1e-11,T=T,mu=mu)
        Eoutu,Eccu = cc.run()
        diff = abs(Eoutg - Eoutu)
        msg = "General: {} Unrestricted: {}".format(Eoutg,Eoutu)
        self.assertTrue(diff < self.thresh,msg)

if __name__ == '__main__':
    unittest.main()
