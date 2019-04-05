import unittest
import numpy
from pyscf import gto, scf
from cqcpy import utils
from kelvin.ccsd import ccsd
from kelvin.scf_system import scf_system
from kelvin import scf_utils

class FTCCReldenTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-14

    def test_Be_gen(self):
        T = 0.8
        mu = 0.04
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        
        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,T=T,mu=mu,iprint=0)
        cc = ccsdT.run()
        ccsdT.compute_ESN()
        Nref = ccsdT.N
        ccsdT._grel_ft_1rdm()
        Nout = numpy.trace(ccsdT.r1rdm)
        diff = abs(Nref - Nout)
        error = "Expected: {}  Actual: {}".format(Nref,Nout)
        self.assertTrue(diff < self.thresh,error)

    def test_Be_gen_prop(self):
        T = 2.0
        mu = 0.0
        thresh = 5e-7
        ngrid = 10
        damp = 0.1
        mi = 100
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        
        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        h = m.get_hcore()
        field = numpy.random.random(h.shape)
        field = 0.1*(field + field.transpose((1,0)))
        fmo = scf_utils.mo_tran_1e(m, field)
        fdiag = fmo.diagonal()
        na = fmo.shape[0]//2

        alpha = 8e-4
        m.get_hcore = lambda *args: h + alpha*field
        m.mo_energy += alpha*fdiag[:na]
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,T=T,mu=mu,iprint=0,econv=1e-9,ngrid=ngrid,damp=damp,max_iter=mi)
        ccf = ccsdT.run()

        m.get_hcore = lambda *args: h - alpha*field
        m.mo_energy -= 2.0*alpha*fdiag[:na]
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,T=T,mu=mu,iprint=0,econv=1e-9,ngrid=ngrid,damp=damp,max_iter=mi)
        ccb = ccsdT.run()

        ref = (ccf[0] - ccb[0])/(2*alpha)

        m.get_hcore = lambda *args: h
        m.mo_energy += alpha*fdiag[:na]
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,T=T,mu=mu,iprint=0,econv=1e-9,ngrid=ngrid,damp=damp,max_iter=mi)
        cc = ccsdT.run()
        ccsdT._grel_ft_1rdm()
        Dm = ccsdT.r1rdm + (ccsdT.n1rdm - numpy.diag(ccsdT.n1rdm.diagonal()))
        out = numpy.einsum('ij,ji->',Dm,fmo)
        diff = abs(out - ref)

        error = "Expected: {}  Actual: {}".format(ref,out)
        self.assertTrue(diff < thresh,error)

    def test_Be_gen_active(self):
        T = 0.02
        mu = 0.0
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        
        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,T=T,mu=mu,damp=0.1,ngrid=160,athresh=1e-30,iprint=0)
        cc = ccsdT.run()
        ccsdT.compute_ESN() 
        Nref = ccsdT.N
        ccsdT._grel_ft_1rdm()
        Nout = numpy.trace(ccsdT.r1rdm)
        diff = abs(Nref - Nout)
        error = "Expected: {}  Actual: {}".format(Nref,Nout)
        self.assertTrue(diff < self.thresh,error)

    def test_Be_gen_prop_active(self):
        T = 0.02
        mu = 0.0
        thresh = 5e-7
        ngrid = 160
        mi = 100
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G')
        
        m = scf.RHF(mol)
        m.conv_tol = 1e-12
        Escf = m.scf()
        h = m.get_hcore()
        field = numpy.random.random(h.shape)
        field = 0.1*(field + field.transpose((1,0)))
        fmo = scf_utils.mo_tran_1e(m, field)
        fdiag = fmo.diagonal()
        na = fmo.shape[0]//2

        alpha = 4e-4
        m.get_hcore = lambda *args: h + alpha*field
        m.mo_energy += alpha*fdiag[:na]
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,T=T,mu=mu,iprint=0,
                damp=0.1,ngrid=ngrid,athresh=1e-30,econv=1e-10,max_iter=mi)
        ccf = ccsdT.run()
        G0f = ccsdT.G0
        G1f = ccsdT.G1

        m.get_hcore = lambda *args: h - alpha*field
        m.mo_energy -= 2.0*alpha*fdiag[:na]
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,T=T,mu=mu,iprint=0,
                damp=0.1,ngrid=ngrid,athresh=1e-30,econv=1e-10,max_iter=mi)
        ccb = ccsdT.run()
        G0b = ccsdT.G0
        G1b = ccsdT.G1

        ref = (ccf[0] - ccb[0])/(2*alpha)
        Acc = (ccf[1] - ccb[1])/(2*alpha)
        A01 = (ccf[0] - ccf[1] - ccb[0] + ccb[1])/(2*alpha)
        A0 = (G0f - G0b)/(2*alpha)
        A1 = (G1f - G1b)/(2*alpha)

        m.get_hcore = lambda *args: h
        m.mo_energy += alpha*fdiag[:na]
        sys = scf_system(m,T,mu,orbtype='g')
        ccsdT = ccsd(sys,T=T,mu=mu,iprint=0,
                damp=0.1,ngrid=ngrid,athresh=1e-30,econv=1e-10,max_iter=mi)
        cc = ccsdT.run()
        ccsdT._grel_ft_1rdm()
        Dm = ccsdT.r1rdm + (ccsdT.n1rdm - numpy.diag(ccsdT.n1rdm.diagonal()))
        out = numpy.einsum('ij,ji->',Dm,fmo)
        diff = abs(out - ref)
        error = "Expected: {}  Actual: {}".format(ref,out)
        self.assertTrue(diff < thresh,error)

if __name__ == '__main__':
    unittest.main()
